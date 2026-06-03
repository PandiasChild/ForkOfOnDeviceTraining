#!/usr/bin/env python3
"""Generate expected_conv1d.h for UnitTestConv1d (Layer-3 tests).

Produces a C header with PyTorch-derived ground-truth values for each
test fixture: forward output, plus autograd-derived dL/dx, dL/dW,
dL/db (when bias is present). Run via `uv run` (CMake wires this
automatically).

The lossGrad used by autograd is torch.ones_like(y) for every fixture,
so the C tests pass an all-ones lossGrad of matching shape to backward.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _format_float_literal(v: float) -> str:
    """Format a Python float as a valid C float literal.

    Python's f"{v:.9g}" strips the trailing ".0" from whole-number
    floats (10.0 -> "10"), which combined with the f suffix produces
    invalid C (10f is rejected by gcc as an integer suffix). repr
    always contains "." or "e" for finite floats, both legal in C
    with the f suffix.
    """
    s = repr(v)
    if s in ("inf", "-inf", "nan"):
        raise ValueError(f"non-finite gold value: {v!r}")
    return s + "f"


def emit_float_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().flatten().tolist()
    body = ", ".join(_format_float_literal(v) for v in flat)
    return (
        f"static const float {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(flat)};\n"
    )


def _run_fixture(name, x, w, b, *, stride, padding, dilation, groups):
    """Run forward + backward on a Conv1d fixture; return all arrays.

    Uses lossGrad = torch.ones_like(y) so the C-side test can simply
    initialize lossGrad as all-ones of matching shape.
    """
    x = x.clone().detach().requires_grad_(True)
    w = w.clone().detach().requires_grad_(True)
    if b is not None:
        b = b.clone().detach().requires_grad_(True)
    y = F.conv1d(x, w, bias=b, stride=stride, padding=padding,
                 dilation=dilation, groups=groups)
    loss = y.sum()
    loss.backward()
    return {
        "name": name,
        "x": x.detach(),
        "w": w.detach(),
        "b": b.detach() if b is not None else None,
        "y": y.detach(),
        "dx": x.grad.detach(),
        "dw": w.grad.detach(),
        "db": b.grad.detach() if b is not None else None,
    }


def fixture_single_channel_single_batch():
    # B=1, Cin=1, Cout=1, L=4, K=2, no bias, stride=1, dilation=1, VALID
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("singleChannelSingleBatch", x, w, None,
                        stride=1, padding=0, dilation=1, groups=1)


def fixture_single_channel_with_bias():
    # B=1, Cin=1, Cout=1, L=4, K=2, with scalar bias
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    b = torch.tensor([1.0])
    return _run_fixture("singleChannelWithBias", x, w, b,
                        stride=1, padding=0, dilation=1, groups=1)


def fixture_multi_channel_with_bias():
    # B=1, Cin=3, Cout=2, L=5, K=3, with bias
    torch.manual_seed(0)
    x = torch.arange(15.0).reshape(1, 3, 5)
    w = torch.arange(18.0).reshape(2, 3, 3) * 0.1
    b = torch.tensor([0.5, -0.5])
    return _run_fixture("multiChannelWithBias", x, w, b,
                        stride=1, padding=0, dilation=1, groups=1)


def fixture_multi_batch():
    # B=4, Cin=2, Cout=2, L=4, K=2, no bias
    torch.manual_seed(1)
    x = torch.randn(4, 2, 4)
    w = torch.randn(2, 2, 2)
    return _run_fixture("multiBatch", x, w, None,
                        stride=1, padding=0, dilation=1, groups=1)


def fixture_groups_depthwise():
    # B=1, Cin=4, Cout=4, K=2, groups=4 (depthwise), no bias
    torch.manual_seed(2)
    x = torch.randn(1, 4, 5)
    w = torch.randn(4, 1, 2)  # [Cout, Cin/groups=1, K]
    return _run_fixture("groupsDepthwise", x, w, None,
                        stride=1, padding=0, dilation=1, groups=4)


def fixture_groups_grouped():
    # B=1, Cin=4, Cout=8, K=2, groups=2, with bias (exercise grouped bias backward too)
    torch.manual_seed(3)
    x = torch.randn(1, 4, 5)
    w = torch.randn(8, 2, 2)  # [Cout, Cin/groups=2, K]
    b = torch.randn(8)
    return _run_fixture("groupsGrouped", x, w, b,
                        stride=1, padding=0, dilation=1, groups=2)


def fixture_stride_dilation():
    # B=1, Cin=1, Cout=1, L=9, K=2, stride=3, dilation=2, VALID, no bias
    x = torch.tensor([[[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("strideDilation", x, w, None,
                        stride=3, padding=0, dilation=2, groups=1)


def fixture_same_padding_symmetric():
    # B=1, Cin=1, Cout=1, L=5, K=3, stride=1, dilation=1, SAME (pad=1 each side)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    w = torch.tensor([[[1.0, 2.0, 3.0]]])
    # PyTorch SAME with K=3 stride=1 dilation=1 -> padding=1 each side -> outLen=inLen
    return _run_fixture("samePaddingSymmetric", x, w, None,
                        stride=1, padding=1, dilation=1, groups=1)


def fixture_same_padding_asymmetric():
    # B=1, Cin=1, Cout=1, L=5, K=4, stride=1, dilation=1, SAME asymmetric
    # Total pad = 4-1 = 3; padLeft=1, padRight=2 (PyTorch: right gets the extra).
    # Use F.pad explicitly to avoid PyTorch refusing padding="same" with even kernel.
    x_raw = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    w = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x = x_raw.clone().detach().requires_grad_(True)
    w_g = w.clone().detach().requires_grad_(True)
    # Pad input: padLeft=1, padRight=2 -> length 5+1+2=8
    x_padded = F.pad(x, (1, 2))
    y = F.conv1d(x_padded, w_g, bias=None, stride=1, padding=0,
                 dilation=1, groups=1)  # outLen=8-4+1=5
    loss = y.sum()
    loss.backward()
    return {
        "name": "samePaddingAsymmetric",
        "x": x.detach(),         # shape [1,1,5] — unpadded
        "w": w_g.detach(),
        "b": None,
        "y": y.detach(),         # shape [1,1,5]
        "dx": x.grad.detach(),   # gradient on the unpadded input
        "dw": w_g.grad.detach(),
        "db": None,
    }


def fixture_same_padding_with_groups():
    # B=2, Cin=4, Cout=4, K=3, groups=2, SAME, with bias.
    # Kernel=3 stride=1 dilation=1 -> padLeft=1 padRight=1 (symmetric SAME).
    torch.manual_seed(4)
    x = torch.randn(2, 4, 6)
    w = torch.randn(4, 2, 3)  # [Cout, Cin/groups=2, K]
    b = torch.randn(4)
    return _run_fixture("samePaddingWithGroups", x, w, b,
                        stride=1, padding=1, dilation=1, groups=2)


def fixture_pointwise():
    # B=2, Cin=3, Cout=4, L=5, K=1, groups=1 -> true pointwise (1x1) convolution.
    # Pointwise conv performs pure cross-channel mixing with no spatial extent:
    #   y[b, oc, t] = sum_ic x[b, ic, t] * w[oc, ic, 0] + bias[oc]
    # so outputLength == inputLength (K=1, stride=1, VALID).
    #
    # The backward pass deliberately uses a NON-UNIFORM lossGrad (random, not the
    # torch.ones_like used by every other fixture). Reason: with lossGrad == ones,
    #   dL/dW[oc, ic, 0] = sum_{b,t} x[b, ic, t]
    # is independent of oc, so every output-channel row of the gold weight gradient
    # would be identical and a bug that mis-assigns the output channel in backward
    # would pass undetected. A random lossGrad makes oc appear in dL/dW, dL/db and
    # dL/dx, pinning the channel-mixing that is the entire purpose of a pointwise conv.
    torch.manual_seed(5)
    x = torch.randn(2, 3, 5).clone().detach().requires_grad_(True)
    w = torch.randn(4, 3, 1).clone().detach().requires_grad_(True)  # [Cout, Cin/groups=Cin, K=1]
    b = torch.randn(4).clone().detach().requires_grad_(True)
    loss_grad = torch.randn(2, 4, 5)
    y = F.conv1d(x, w, bias=b, stride=1, padding=0, dilation=1, groups=1)
    y.backward(loss_grad)
    return {
        "name": "pointwise",
        "x": x.detach(),
        "w": w.detach(),
        "b": b.detach(),
        "y": y.detach(),
        "dx": x.grad.detach(),
        "dw": w.grad.detach(),
        "db": b.grad.detach(),
        "lossGrad": loss_grad.detach(),
    }


def emit_fixture(parts, fx):
    pre = f"conv1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"weight_{pre}", fx["w"]))
    if fx["b"] is not None:
        parts.append(emit_float_array(f"bias_{pre}", fx["b"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    parts.append(emit_float_array(f"expectedWeightGrad_{pre}", fx["dw"]))
    if fx["db"] is not None:
        parts.append(emit_float_array(f"expectedBiasGrad_{pre}", fx["db"]))
    # Emitted only for fixtures that supply an explicit (non-ones) lossGrad.
    if fx.get("lossGrad") is not None:
        parts.append(emit_float_array(f"lossGrad_{pre}", fx["lossGrad"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_conv1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONV1D_H\n",
        "#define ODT_EXPECTED_CONV1D_H\n",
        "#include <stdlib.h>\n\n",
    ]

    fixtures = [
        fixture_single_channel_single_batch(),
        fixture_single_channel_with_bias(),
        fixture_multi_channel_with_bias(),
        fixture_multi_batch(),
        fixture_groups_depthwise(),
        fixture_groups_grouped(),
        fixture_stride_dilation(),
        fixture_same_padding_symmetric(),
        fixture_same_padding_asymmetric(),
        fixture_same_padding_with_groups(),
        fixture_pointwise(),
    ]
    for fx in fixtures:
        emit_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_CONV1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
