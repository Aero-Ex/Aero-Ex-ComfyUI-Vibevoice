# ComfyUI-VibeVoice-Native

A native port of [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) (long-form, multi-speaker
text to speech with voice cloning) for ComfyUI. No extra pip packages: it runs on ComfyUI's own
Qwen2.5 LLM, attention, quantization (int8 convrot) and dynamic VRAM offloading.

## Nodes

- **Load VibeVoice Model** — loads a converted single-file checkpoint from `models/text_encoders/`.
- **VibeVoice Text To Speech** — generates speech from a script. Connect one voice sample per speaker
  (`speaker_1`, `speaker_2`, ...); speakers without a sample get a default voice.

## Checkpoints

Download one from [Aero-Ex/ComfyUI-Vibevoice](https://huggingface.co/Aero-Ex/ComfyUI-Vibevoice/tree/main)
into `ComfyUI/models/text_encoders/`:

- [`vibevoice_7b_bf16.safetensors`](https://huggingface.co/Aero-Ex/ComfyUI-Vibevoice/resolve/main/vibevoice_7b_bf16.safetensors) (17.6 GB)
- [`vibevoice_7b_int8_convrot.safetensors`](https://huggingface.co/Aero-Ex/ComfyUI-Vibevoice/resolve/main/vibevoice_7b_int8_convrot.safetensors) (10.5 GB, recommended for GPUs under 16 GB)

A ready-made workflow is in `workflows/vibevoice.json`.

These are the Microsoft VibeVoice-7B weights converted to the native layout (merged QKV / gate-up,
LM head pruned to the four speech tokens). The original sharded Hugging Face files are not loaded
directly — the loader stops with a clear error.

## Script format

One line per turn:

```
Speaker 1: Welcome back to the podcast.
Speaker 2: Thanks, it is great to be here.
```

Lines without a `Speaker N:` prefix continue the previous speaker. Write stage directions as plain
punctuation; anything in brackets is read aloud.

## Settings

| Input | Default | Notes |
| --- | --- | --- |
| `cfg_scale` | 1.3 | Upstream default. 1.5–2.0 is often more stable on long scripts. |
| `steps` | 10 | Diffusion steps per speech frame. |
| `max_duration` | 600 s | Upper limit only; generation ends on its own. The KV cache is sized for it, so set it close to the expected length to leave more VRAM for model weights (faster on small GPUs). |
| `seed` | 0 | Same seed, same output. |

Voice samples: 5–15 s of clean, single-speaker speech. Any sample rate and channel count works.

## Speed

On an RTX 4050 Laptop (6 GB) the int8 7B model runs at about 2 speech tokens per second
(7.5 tokens = 1 s of audio). It is bound by streaming weights over PCIe; a GPU that holds the whole
model runs it many times faster.

## Tests

From the ComfyUI folder:

```
python -m pytest --import-mode=importlib custom_nodes/ComfyUI-VibeVoice-Native/tests
```
