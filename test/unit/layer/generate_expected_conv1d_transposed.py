#!/usr/bin/env python3
"""Generate expected_conv1d_transposed.h for UnitTestConv1dTransposed.

Produces a C header with PyTorch-derived forward outputs and
autograd-derived dL/dx, dL/dW, dL/db for each fixture. lossGrad is
torch.ones_like(y) for every fixture.

ConvTranspose1d weight shape: [Cin, Cout/groups, K] — note the order
vs. Conv1d's [Cout, Cin/groups, K].
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _format_float_literal(v: float) -> str:
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


def _run_fixture(name, x, w, b, *, stride, padding, output_padding,
                 dilation, groups):
    x = x.clone().detach().requires_grad_(True)
    w = w.clone().detach().requires_grad_(True)
    if b is not None:
        b = b.clone().detach().requires_grad_(True)
    y = F.conv_transpose1d(x, w, bias=b, stride=stride, padding=padding,
                           output_padding=output_padding,
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
    # B=1, Cin=1, Cout=1, L=3, K=2, stride=1, no bias
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("singleChannelSingleBatch", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_single_channel_with_bias():
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    b = torch.tensor([0.5])
    return _run_fixture("singleChannelWithBias", x, w, b,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_multi_channel_with_bias():
    torch.manual_seed(10)
    x = torch.arange(12.0).reshape(1, 3, 4)
    w = torch.arange(12.0).reshape(3, 2, 2) * 0.1
    b = torch.tensor([0.5, -0.5])
    return _run_fixture("multiChannelWithBias", x, w, b,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_multi_batch():
    torch.manual_seed(11)
    x = torch.randn(3, 2, 4)
    w = torch.randn(2, 2, 2)
    return _run_fixture("multiBatch", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_groups_depthwise():
    # Cin=4, Cout=4, groups=4 (depthwise transpose). Weight: [Cin, Cout/g=1, K]
    torch.manual_seed(12)
    x = torch.randn(1, 4, 4)
    w = torch.randn(4, 1, 2)
    return _run_fixture("groupsDepthwise", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=4)


def fixture_groups_grouped():
    # Cin=4, Cout=8, groups=2. Weight: [Cin, Cout/g=4, K]
    torch.manual_seed(13)
    x = torch.randn(1, 4, 4)
    w = torch.randn(4, 4, 2)
    b = torch.randn(8)
    return _run_fixture("groupsGrouped", x, w, b,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=2)


def fixture_stride2():
    # The classic upsampling ConvT use case: stride=2 doubles length-1.
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("stride2", x, w, None,
                        stride=2, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_stride2_with_output_padding():
    # outputPadding=1 trails an extra zero (plus bias if any)
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    b = torch.tensor([1.0])
    return _run_fixture("stride2WithOutputPadding", x, w, b,
                        stride=2, padding=0, output_padding=1,
                        dilation=1, groups=1)


def fixture_dilation2():
    # dilation>1: kernel taps stretched; output_length = (Lin-1)*stride + dilation*(K-1) + 1
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("dilation2", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=2, groups=1)


def emit_fixture(parts, fx):
    pre = f"convT1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"weight_{pre}", fx["w"]))
    if fx["b"] is not None:
        parts.append(emit_float_array(f"bias_{pre}", fx["b"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    parts.append(emit_float_array(f"expectedWeightGrad_{pre}", fx["dw"]))
    if fx["db"] is not None:
        parts.append(emit_float_array(f"expectedBiasGrad_{pre}", fx["db"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_conv1d_transposed.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONV1D_TRANSPOSED_H\n",
        "#define ODT_EXPECTED_CONV1D_TRANSPOSED_H\n",
        "#include <stdlib.h>\n\n",
    ]

    fixtures = [
        fixture_single_channel_single_batch(),
        fixture_single_channel_with_bias(),
        fixture_multi_channel_with_bias(),
        fixture_multi_batch(),
        fixture_groups_depthwise(),
        fixture_groups_grouped(),
        fixture_stride2(),
        fixture_stride2_with_output_padding(),
        fixture_dilation2(),
    ]
    for fx in fixtures:
        emit_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_CONV1D_TRANSPOSED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
