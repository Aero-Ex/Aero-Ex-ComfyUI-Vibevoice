import math

import folder_paths
import comfy.sd
import comfy.text_encoders.hunyuan_video
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .vibevoice.text_encoder import FRAMES_PER_SECOND, SAMPLE_RATE, VibeVoiceTokenizer, detect_config, te


def load_vibevoice(path):
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
    if "prediction_head.cond_proj.weight" not in sd:
        raise ValueError("This VibeVoice checkpoint is not in the native layout. Use a converted vibevoice_*.safetensors file.")

    class Target:
        params = {"config": detect_config(sd)}
        clip = te(**comfy.text_encoders.hunyuan_video.llama_detect(sd))
        tokenizer = VibeVoiceTokenizer

    clip = comfy.sd.CLIP(Target, embedding_directory=folder_paths.get_folder_paths("embeddings"),
                         parameters=comfy.utils.calculate_parameters(sd), state_dict=[sd])
    clip.patcher.cached_patcher_init = (load_vibevoice_patcher, (path,))
    return clip


def load_vibevoice_patcher(path):
    return load_vibevoice(path).patcher


class VibeVoiceLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VibeVoiceLoader",
            display_name="Load VibeVoice Model",
            category="audio",
            description="Loads a converted single-file VibeVoice checkpoint (bf16 or int8 convrot) from models/text_encoders.",
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("text_encoders"))],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        return io.NodeOutput(load_vibevoice(folder_paths.get_full_path_or_raise("text_encoders", model_name)))


class VibeVoiceTTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VibeVoiceTTS",
            display_name="VibeVoice Text To Speech",
            category="audio",
            description="Generates speech from a script with a model from Load VibeVoice Model. Write one line per turn as \"Speaker 1: text\"; each speaker uses the voice sample connected in the same order.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True, default="Speaker 1: Hello, this is VibeVoice running natively in ComfyUI."),
                io.Autogrow.Input("voices", optional=True, template=io.Autogrow.TemplateNames(
                    input=io.Audio.Input("speaker", tooltip="Voice sample to clone for this speaker."),
                    names=[f"speaker_{i}" for i in range(1, 101)], min=0)),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg_scale", default=1.3, min=0.0, max=100.0, step=0.05),
                io.Int.Input("steps", default=10, min=1, max=1000, tooltip="Diffusion steps per speech frame."),
                io.Float.Input("max_duration", default=600.0, min=1.0, max=10000.0, step=1.0, tooltip="Maximum duration in seconds; generation usually ends earlier. Lower values reserve less VRAM for the cache.", advanced=True),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, clip, text, seed, cfg_scale, steps, max_duration, voices: io.Autogrow.Type = None) -> io.NodeOutput:
        voices = [voices[name] for name in sorted(voices or {}, key=lambda name: int(name.removeprefix("speaker_")))]
        max_tokens = math.ceil(max_duration * FRAMES_PER_SECOND) + 2
        tokens = clip.tokenize(text, voices=voices, cfg_scale=cfg_scale, steps=steps, max_tokens=max_tokens)
        waveform = clip.generate(tokens, max_length=max_tokens, seed=seed)
        return io.NodeOutput({"waveform": waveform, "sample_rate": SAMPLE_RATE})


class VibeVoiceExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [VibeVoiceLoader, VibeVoiceTTS]


async def comfy_entrypoint():
    return VibeVoiceExtension()
