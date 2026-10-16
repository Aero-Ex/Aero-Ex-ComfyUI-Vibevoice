import pytest
import torch

import comfy.ops
from vibevoice.model import causal_conv, causal_conv_transpose
from vibevoice.text_encoder import parse_script


@pytest.mark.parametrize("kernel,stride", [(7, 1), (4, 2), (16, 8)])
def test_streaming_causal_conv_matches_full_sequence(kernel, stride):
    torch.manual_seed(0)
    conv = comfy.ops.manual_cast.Conv1d(3, 5, kernel, stride=stride)
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.normal_(conv.bias)
    x = torch.randn(1, 3, stride * 12)
    cache = {}
    streamed = torch.cat([causal_conv(conv, chunk, cache) for chunk in x.split(stride * 3, dim=-1)], dim=-1)
    torch.testing.assert_close(streamed, causal_conv(conv, x))


def test_causal_conv_keeps_partial_frame():
    conv = comfy.ops.manual_cast.Conv1d(1, 1, 4, stride=2)
    assert causal_conv(conv, torch.randn(1, 1, 7)).shape[-1] == 4


@pytest.mark.parametrize("kernel,stride", [(16, 8), (10, 5), (4, 2)])
def test_streaming_causal_conv_transpose_matches_full_sequence(kernel, stride):
    torch.manual_seed(0)
    conv = comfy.ops.manual_cast.ConvTranspose1d(3, 5, kernel, stride=stride)
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.normal_(conv.bias)
    x = torch.randn(1, 3, 6)
    cache = {}
    streamed = torch.cat([causal_conv_transpose(conv, x[..., i:i + 1], cache) for i in range(x.shape[-1])], dim=-1)
    full = causal_conv_transpose(conv, x)
    assert full.shape[-1] == x.shape[-1] * stride
    torch.testing.assert_close(streamed, full)


def test_parse_script_normalizes_one_based_speakers():
    script = "Speaker 1: Hello there.\n\nSpeaker 2:How are you?\nstill talking\nspeaker 1 : Fine."
    assert parse_script(script) == [(0, "Hello there."), (1, "How are you?"), (1, "still talking"), (0, "Fine.")]


def test_parse_script_keeps_zero_based_speakers_and_plain_text():
    assert parse_script("Speaker 0: Hi\nSpeaker 1: Yo") == [(0, "Hi"), (1, "Yo")]
    assert parse_script("Just one line.") == [(0, "Just one line.")]
