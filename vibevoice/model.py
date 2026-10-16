"""VibeVoice speech tokenizers, diffusion head and connectors. Adapted from Microsoft VibeVoice (MIT)."""

import torch
from torch import nn

import comfy.ops
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding


def causal_conv(conv, x, cache=None):
    kernel, stride = conv.kernel_size[0], conv.stride[0]
    pad = kernel - stride
    if cache is None:
        # Right padding keeps the last partial frame, giving ceil(length / stride) outputs.
        return conv(torch.nn.functional.pad(x, (pad, -x.shape[-1] % stride)))
    previous = cache.get(conv)
    if previous is None:
        previous = x.new_zeros(x.shape[0], x.shape[1], pad)
    x = torch.cat((previous, x), dim=-1)
    cache[conv] = x[..., x.shape[-1] - pad:].clone()
    return conv(x)


def causal_conv_transpose(conv, x, cache=None):
    kernel, stride = conv.kernel_size[0], conv.stride[0]
    length = x.shape[-1]
    if cache is not None:
        # Only the last (kernel - 1) // stride frames overlap the new output samples.
        context = (kernel - 1) // stride
        previous = cache.get(conv)
        if previous is None:
            previous = x.new_zeros(x.shape[0], x.shape[1], context)
        x = torch.cat((previous, x), dim=-1)
        cache[conv] = x[..., x.shape[-1] - context:].clone()
    out = conv(x)
    return out[..., out.shape[-1] - (kernel - stride) - length * stride:out.shape[-1] - (kernel - stride)]


class FeedForward(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.linear1 = operations.Linear(dim, dim * 4, dtype=dtype, device=device)
        self.linear2 = operations.Linear(dim * 4, dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear2(torch.nn.functional.gelu(self.linear1(x)))


class Block1D(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.norm = operations.RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.mixer = operations.Conv1d(dim, dim, 7, groups=dim, dtype=dtype, device=device)
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.ffn_norm = operations.RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.ffn = FeedForward(dim, dtype, device, operations)
        self.ffn_gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x, cache=None):
        mixed = causal_conv(self.mixer, self.norm(x.transpose(1, 2)).transpose(1, 2), cache)
        x = x.addcmul(mixed, comfy.ops.cast_to_input(self.gamma, x)[:, None])
        h = self.ffn(self.ffn_norm(x.transpose(1, 2))) * comfy.ops.cast_to_input(self.ffn_gamma, x)
        return x + h.transpose(1, 2)


RATIOS = (2, 2, 4, 5, 5, 8)
DEPTHS = (3, 3, 3, 3, 3, 3, 8)
FILTERS = 32
HOP_LENGTH = 3200


class TokenizerEncoder(nn.Module):
    def __init__(self, latent_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.downsample = nn.ModuleList([operations.Conv1d(1, FILTERS, 7, dtype=dtype, device=device)] + [
            operations.Conv1d(FILTERS * 2 ** i, FILTERS * 2 ** (i + 1), ratio * 2, stride=ratio, dtype=dtype, device=device)
            for i, ratio in enumerate(RATIOS)
        ])
        self.stages = nn.ModuleList([
            nn.ModuleList([Block1D(FILTERS * 2 ** i, dtype, device, operations) for _ in range(depth)])
            for i, depth in enumerate(DEPTHS)
        ])
        self.head = operations.Conv1d(FILTERS * 2 ** len(RATIOS), latent_dim, 7, dtype=dtype, device=device)

    def forward(self, x, cache=None):
        for conv, stage in zip(self.downsample, self.stages):
            x = causal_conv(conv, x, cache)
            for block in stage:
                x = block(x, cache)
        return causal_conv(self.head, x, cache)


class TokenizerDecoder(nn.Module):
    def __init__(self, latent_dim, dtype=None, device=None, operations=None):
        super().__init__()
        top = FILTERS * 2 ** len(RATIOS)
        self.upsample = nn.ModuleList([operations.Conv1d(latent_dim, top, 7, dtype=dtype, device=device)] + [
            operations.ConvTranspose1d(top // 2 ** i, top // 2 ** (i + 1), ratio * 2, stride=ratio, dtype=dtype, device=device)
            for i, ratio in enumerate(reversed(RATIOS))
        ])
        self.stages = nn.ModuleList([
            nn.ModuleList([Block1D(top // 2 ** i, dtype, device, operations) for _ in range(depth)])
            for i, depth in enumerate(reversed(DEPTHS))
        ])
        self.head = operations.Conv1d(FILTERS, 1, 7, dtype=dtype, device=device)

    def forward(self, x, cache=None):
        for i, (conv, stage) in enumerate(zip(self.upsample, self.stages)):
            x = causal_conv(conv, x, cache) if i == 0 else causal_conv_transpose(conv, x, cache)
            for block in stage:
                x = block(x, cache)
        return causal_conv(self.head, x, cache)


class Connector(nn.Module):
    def __init__(self, input_dim, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.fc1 = operations.Linear(input_dim, hidden_size, dtype=dtype, device=device)
        self.norm = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.fc2 = operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


class HeadFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype, device, operations):
        super().__init__()
        self.gate_up_proj = operations.Linear(hidden_size, intermediate_size * 2, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return comfy.ops.linear_input_act(self.down_proj, self.gate_up_proj(x), "swiglu")


class HeadLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype, device, operations):
        super().__init__()
        self.norm = operations.RMSNorm(hidden_size, eps=1e-5, dtype=dtype, device=device)
        self.ffn = HeadFeedForward(hidden_size, intermediate_size, dtype, device, operations)
        self.adaLN_modulation = operations.Linear(hidden_size, hidden_size * 3, bias=False, dtype=dtype, device=device)

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = self.norm(x)
        return x.addcmul(gate, self.ffn(h.addcmul(h, scale) + shift))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, latent_dim, dtype, device, operations):
        super().__init__()
        self.linear = operations.Linear(hidden_size, latent_dim, bias=False, dtype=dtype, device=device)
        self.adaLN_modulation = operations.Linear(hidden_size, hidden_size * 2, bias=False, dtype=dtype, device=device)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=1e-5)
        return self.linear(x.addcmul(x, scale) + shift)


class DiffusionHead(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_layers, latent_dim=64, dtype=None, device=None, operations=None):
        super().__init__()
        self.noisy_images_proj = operations.Linear(latent_dim, hidden_size, bias=False, dtype=dtype, device=device)
        self.cond_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.t_embedder = nn.Module()
        self.t_embedder.mlp = nn.Sequential(
            operations.Linear(256, hidden_size, bias=False, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device),
        )
        self.layers = nn.ModuleList([HeadLayer(hidden_size, intermediate_size, dtype, device, operations) for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, latent_dim, dtype, device, operations)

    def forward(self, x, timestep, projected_condition):
        # The reference embeds the timestep after rounding it to the compute dtype.
        t = self.t_embedder.mlp(timestep_embedding(timestep.to(x.dtype), 256).to(x.dtype))
        c = torch.nn.functional.silu(projected_condition + t)
        x = self.noisy_images_proj(x)
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)
