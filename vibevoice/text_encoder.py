"""VibeVoice text-to-speech generation. Adapted from Microsoft VibeVoice (MIT)."""

import math
import os
import re
import types

import torch
from transformers import Qwen2Tokenizer

import comfy.audio
import comfy.k_diffusion.sampling
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
import comfy.text_encoders.llama
from comfy.text_encoders.llama import FixedKV, Llama2_, Qwen25_3BConfig

from .model import HOP_LENGTH, Connector, DiffusionHead, TokenizerDecoder, TokenizerEncoder

SAMPLE_RATE = 24000
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
EOS, SPEECH_START, SPEECH_END, SPEECH_DIFFUSION = 151643, 151652, 151653, 151654
# Row order of the pruned LM head: the only tokens VibeVoice samples.
LM_TOKENS = (EOS, SPEECH_START, SPEECH_END, SPEECH_DIFFUSION)
SYSTEM_PROMPT = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
VAE_STD = 0.625


def parse_script(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = SPEAKER_LINE.match(line)
        if match:
            lines.append((int(match.group(1)), match.group(2).strip()))
        else:
            lines.append((lines[-1][0] if lines else 1, line))
    if lines and min(speaker for speaker, _ in lines) > 0:
        lines = [(speaker - 1, line) for speaker, line in lines]
    return lines


class VibeVoiceTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.realpath(comfy.text_encoders.llama.__file__)), "qwen25_tokenizer"))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_with_weights(self, text, return_word_ids=False, voices=[], cfg_scale=1.3, steps=10, max_tokens=4500, **kwargs):
        ids = self.encode(SYSTEM_PROMPT)
        voice_spans = []
        if len(voices) > 0:
            ids += self.encode(" Voice input:\n")
            for index, audio in enumerate(voices):
                waveform = comfy.audio.resample(audio["waveform"][0].float().mean(0), audio["sample_rate"], SAMPLE_RATE)
                waveform = waveform * (10 ** (-25 / 20) / (waveform.square().mean().sqrt() + 1e-6))
                peak = waveform.abs().max()
                if peak > 1.0:
                    waveform = waveform / (peak + 1e-6)
                ids += self.encode(f" Speaker {index}:") + [SPEECH_START]
                voice_spans.append((len(ids), waveform))
                ids += [SPEECH_DIFFUSION] * math.ceil(waveform.shape[-1] / HOP_LENGTH) + [SPEECH_END] + self.encode("\n")
        ids += self.encode(" Text input:\n")
        for speaker, line in parse_script(text):
            ids += self.encode(f" Speaker {speaker}: {line}\n")
        ids += self.encode(" Speech output:\n") + [SPEECH_START]
        return {"ids": ids, "voices": voice_spans, "cfg_scale": cfg_scale, "steps": steps, "max_tokens": max_tokens}

    def state_dict(self):
        return {}


def detect_config(state_dict, prefix=""):
    hidden_size = state_dict[f"{prefix}model.norm.weight"].shape[0]
    head_dim = 128
    qkv = state_dict[f"{prefix}model.layers.0.self_attn.qkv_proj.weight"].shape[0]
    return {
        "llm": {
            "vocab_size": state_dict[f"{prefix}model.embed_tokens.weight"].shape[0],
            "hidden_size": hidden_size,
            "intermediate_size": state_dict[f"{prefix}model.layers.0.mlp.gate_up_proj.weight"].shape[0] // 2,
            "num_hidden_layers": comfy.model_detection.count_blocks(state_dict.keys(), f"{prefix}model.layers.{{}}."),
            "num_attention_heads": hidden_size // head_dim,
            "num_key_value_heads": (qkv - hidden_size) // (2 * head_dim),
            "max_position_embeddings": 65536 if hidden_size < 2048 else 32768,
        },
        "head_intermediate_size": state_dict[f"{prefix}prediction_head.layers.0.ffn.gate_up_proj.weight"].shape[0] // 2,
        "head_layers": comfy.model_detection.count_blocks(state_dict.keys(), f"{prefix}prediction_head.layers.{{}}."),
    }


class VibeVoiceTEModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}, config=None):
        super().__init__()
        operations = model_options.get("custom_operations", comfy.ops.manual_cast)
        quant = model_options.get("quantization_metadata")
        if quant is not None and "custom_operations" not in model_options:
            operations = comfy.ops.mixed_precision_ops(quant, dtype)
        llm_config = Qwen25_3BConfig(**config["llm"])
        llm_config.fixed_kv = True
        llm_config.merged_qkv = True
        llm_config.merged_mlp = True
        self.config = llm_config
        hidden_size = llm_config.hidden_size
        self.model = Llama2_(llm_config, device=device, dtype=dtype, ops=operations)
        self.model.prefetch_dynamic_vbars = True
        self.model.lm_head = operations.Linear(hidden_size, len(LM_TOKENS), bias=False, dtype=dtype, device=device)
        self.prediction_head = DiffusionHead(hidden_size, config["head_intermediate_size"], config["head_layers"], dtype=dtype, device=device, operations=operations)
        self.acoustic_connector = Connector(64, hidden_size, dtype=dtype, device=device, operations=operations)
        self.semantic_connector = Connector(128, hidden_size, dtype=dtype, device=device, operations=operations)
        self.acoustic_encoder = TokenizerEncoder(64, dtype=dtype, device=device, operations=operations)
        self.acoustic_decoder = TokenizerDecoder(64, dtype=dtype, device=device, operations=operations)
        self.semantic_encoder = TokenizerEncoder(128, dtype=dtype, device=device, operations=operations)
        self.register_buffer("speech_scaling_factor", torch.empty((), dtype=dtype, device=device))
        self.register_buffer("speech_bias_factor", torch.empty((), dtype=dtype, device=device))
        self.dtypes = {dtype}
        self.execution_device = device

    def get_dynamic_vram__units(self):
        units, last_units = self.model.get_dynamic_vram__units()
        # The diffusion head runs every sampling step of every frame; the embedding table and voice encoder only run on the prompt.
        return [self.prediction_head, (self.acoustic_connector, self.semantic_connector, self.acoustic_decoder, self.semantic_encoder), *units], [*last_units, self.model.embed_tokens, self.acoustic_encoder]

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        pass

    def load_sd(self, state_dict):
        return self.load_state_dict(state_dict, strict=False, assign=getattr(self, "can_assign_sd", False))

    def memory_estimation_function(self, tokens, device=None):
        config = self.config
        length = len(tokens["ids"]) + tokens["max_tokens"]
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
        cache = 2 * 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * length
        prefill = 2 * len(tokens["ids"]) * (config.intermediate_size * 3 + config.hidden_size * 8)
        voice = 450 * max((waveform.shape[-1] for _, waveform in tokens["voices"]), default=0)
        return (cache + max(prefill, voice)) * comfy.model_management.dtype_size(dtype)

    def _prompt_embeds(self, tokens, generator, dtype):
        device = self.execution_device
        embeds = self.model.embed_tokens(torch.tensor([tokens["ids"]], device=device), out_dtype=dtype)
        for start, waveform in tokens["voices"]:
            latents = self.acoustic_encoder(waveform.to(device, dtype)[None, None]).transpose(1, 2)
            # The acoustic tokenizer samples its latent with a per-utterance random scale.
            scale = VAE_STD * torch.randn((), generator=generator, device=generator.device)
            latents = latents + scale.to(device, dtype) * torch.randn(latents.shape, generator=generator, device=generator.device).to(device, dtype)
            latents = (latents + comfy.ops.cast_to_input(self.speech_bias_factor, latents)) * comfy.ops.cast_to_input(self.speech_scaling_factor, latents)
            embeds[:, start:start + latents.shape[1]] = self.acoustic_connector(latents)
        return embeds

    def _sample_latent(self, condition, model_sampling, sigmas, cfg_scale, generator):
        projected = self.prediction_head.cond_proj(condition)

        def denoise(x, sigma):
            model_input = model_sampling.calculate_input(sigma, x).to(condition.dtype)
            timestep = model_sampling.timestep(sigma).repeat(2)
            positive, negative = self.prediction_head(model_input.repeat(2, 1), timestep, projected).float().chunk(2)
            return model_sampling.calculate_denoised(sigma, negative + cfg_scale * (positive - negative), x)

        noise = torch.randn((1, 64), generator=generator, device=generator.device).to(condition.device)
        x = noise * (1 + sigmas[0] ** 2) ** 0.5
        # Host-side sigmas keep the sampler free of device syncs, so the whole loop can be graph captured.
        return comfy.k_diffusion.sampling.sample_dpmpp_2m(denoise, x, sigmas, disable=True)

    def generate(self, tokens, max_length=256, seed=None, **kwargs):
        device = self.execution_device
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
        rng_device = device if torch.device(device).type != "mps" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed or 0)
        model_sampling = comfy.model_base.model_sampling(types.SimpleNamespace(sampling_settings={"beta_schedule": "cosine"}), comfy.model_base.ModelType.V_PREDICTION).to(device)
        timesteps = torch.linspace(0, model_sampling.num_timesteps - 1, tokens["steps"] + 1).round().flip(0)[:-1]
        sigmas = torch.cat((model_sampling.sigma(timesteps.to(device)).cpu(), torch.zeros(1)))

        prompt = self._prompt_embeds(tokens, generator, dtype)
        prompt_length = prompt.shape[1]
        max_tokens = min(max_length, self.config.max_position_embeddings - prompt_length)
        if max_tokens < 1:
            raise ValueError("VibeVoice prompt fills the model context; shorten the text or voice samples.")
        special = self.model.embed_tokens(torch.tensor([SPEECH_START, SPEECH_END], device=device), out_dtype=dtype)

        # Row 0 is the prompt, row 1 the CFG branch that only sees speech since the last speech start.
        embeds = torch.cat((prompt, torch.zeros_like(prompt)))
        embeds[1, -1] = special[0]
        mask = torch.ones((2, prompt_length + max_tokens), device=device, dtype=torch.long)
        mask[1, :prompt_length - 1] = 0
        positions = mask[:, :prompt_length].cumsum(-1).sub_(1).clamp_min_(0)
        cache = self.model.init_kv_cache(2, prompt_length + max_tokens, device, dtype)
        hidden, _, cache = self.model(None, embeds=embeds, attention_mask=mask[:, :prompt_length], position_ids=positions, past_key_values=cache, dtype=dtype)
        hidden = hidden[:, -1:]
        vbar = getattr(self, "dynamic_vbars", {}).get(device)
        if vbar is not None:
            # Voice encoding and prefill transients lower the VBAR watermark: let decode fault the weights back.
            comfy.model_management.reset_cast_buffers()
            vbar.set_watermark(vbar.max_size)
        if isinstance(cache[0], FixedKV):
            mask = None
        positions = positions[:, -1:] + 1
        negative_length = 1
        acoustic_cache, semantic_cache = {}, {}
        chunks = []
        # The diffusion sampling is graph captured, so its input and output keep fixed addresses.
        sampler_io = {"condition": torch.empty((2, self.config.hidden_size), device=device, dtype=dtype),
                      "latent": torch.empty((1, 64), device=device, dtype=torch.float32)}

        def sample_core():
            sampler_io["latent"].copy_(self._sample_latent(sampler_io["condition"], model_sampling, sigmas, tokens["cfg_scale"], generator))

        progress = comfy.utils.ProgressBar(max_tokens)
        try:
            for step in comfy.utils.model_trange(max_tokens, desc="VibeVoice", unit="token"):
                comfy.model_management.throw_exception_if_processing_interrupted()
                token = LM_TOKENS[self.model.lm_head(hidden[0, -1]).argmax().item()]
                # Both rows consumed the last input, but the reference feeds the CFG row only inputs that
                # precede a diffusion token, and empties it on a new segment so the next speech start is its only context.
                if step > 0 and token == SPEECH_DIFFUSION:
                    negative_length += 1
                elif step > 0:
                    negative_length = 0 if token == SPEECH_START else negative_length
                    positions[1] = negative_length
                    if mask is None:
                        for kv in cache:
                            kv.seqlen[1] = negative_length
                    elif token == SPEECH_START:
                        mask[1].zero_()
                    else:
                        mask[1, prompt_length + step - 1] = 0
                if token == EOS:
                    break
                if token == SPEECH_DIFFUSION:
                    sampler_io["condition"].copy_(hidden[:, -1])
                    comfy.model_prefetch.malloc_graph_begin(device)
                    queue = comfy.model_prefetch.make_prefetch_queue([self.prediction_head], device, {"prefetch_dynamic_vbars": True})
                    comfy.model_prefetch.prefetch_queue_pop(queue, device, self.prediction_head, dtype, core=sample_core, enable_graph=True, generator=generator, malloc_scope="diffusion")
                    comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="diffusion")
                    comfy.model_prefetch.malloc_graph_end()
                    latent = sampler_io["latent"].to(dtype)
                    scaled = latent / comfy.ops.cast_to_input(self.speech_scaling_factor, latent) - comfy.ops.cast_to_input(self.speech_bias_factor, latent)
                    audio = self.acoustic_decoder(scaled[:, :, None], acoustic_cache)
                    chunks.append(audio)
                    semantic = self.semantic_encoder(audio, semantic_cache).transpose(1, 2)
                    next_embed = self.acoustic_connector(latent[:, None]) + self.semantic_connector(semantic)
                else:
                    next_embed = special[0 if token == SPEECH_START else 1].view(1, 1, -1)
                    if token == SPEECH_END:
                        acoustic_cache.clear()
                        semantic_cache.clear()
                if mask is not None:
                    mask[1, prompt_length + step] = 1
                hidden, _, cache = self.model(None, embeds=next_embed.repeat(2, 1, 1), past_key_values=cache, dtype=dtype, position_ids=positions,
                                              attention_mask=mask[:, :prompt_length + step + 1] if mask is not None else None)
                positions = positions + 1
                progress.update_absolute(step + 1)
        finally:
            # Captured graphs reference this run's buffers; let queued work (e.g. after an interrupt) finish before freeing them.
            comfy.model_management.synchronize()
            comfy.model_prefetch.cleanup_prefetch_queues()

        if len(chunks) == 0:
            raise ValueError("VibeVoice generated no speech.")
        return torch.cat(chunks, dim=-1).to(comfy.model_management.intermediate_device(), torch.float32)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class VibeVoiceTEModel_(VibeVoiceTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}, config=None):
            dtype = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, device)
            if llama_quantization_metadata is not None:
                model_options = {**model_options, "quantization_metadata": llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype, model_options=model_options, config=config)
    return VibeVoiceTEModel_
