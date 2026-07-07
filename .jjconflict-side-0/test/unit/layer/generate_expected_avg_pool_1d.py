#!/usr/bin/env python3
"""Generate expected_avg_pool_1d.h for UnitTestAvgPool1d (Layer-3 tests).

Produces a C header with PyTorch-derived ground-truth values for each
fixture: forward output, propLoss (dL/dx). For most fixtures
lossGrad = torch.ones_like(y); withStrideAndDilation uses
torch.randn_like(y) (and emits the lossGrad as a gold array) so that
positional mutations on the stride/dilation backward path are
non-vacuous (per codebase_uniform_lossgrad_mutation_vacuity).

Divisor is `count_include_pad=True` (PyTorch default — A1 semantics
per spec §6.4 / §11). The hand-derived self-check uses kernel_size
as divisor for ALL output positions, including SAME edges where
validCount < kernel_size — same convention as the C impl.

Run via `uv run` (CMake wires this automatically).
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


def _hand_check_avg_forward(x, kernel_size, stride, padding, dilation):
    """count_include_pad=True semantics: divisor = kernel_size always.
    Padded positions are treated as zero and INCLUDED in the divisor count."""
    B, C, L = x.shape
    in_padded_L = L + 2 * padding
    eff_k = (kernel_size - 1) * dilation + 1
    out_L = (in_padded_L - eff_k) // stride + 1
    out = torch.zeros((B, C, out_L))
    for b in range(B):
        for c in range(C):
            for o in range(out_L):
                start = o * stride - padding
                accum = 0.0
                for k in range(kernel_size):
                    idx = start + k * dilation
                    if 0 <= idx < L:
                        accum += x[b, c, idx].item()
                    # else: padded position contributes 0, INCLUDED in divisor
                out[b, c, o] = accum / kernel_size
    return out


def _run_avg_fixture(name, x, *, kernel_size, stride, padding, dilation,
                     loss_grad_kind="ones"):
    x_in = x.clone().detach().requires_grad_(True)

    # PyTorch's F.avg_pool1d defaults to count_include_pad=True.
    # Note: F.avg_pool1d does NOT support `dilation` — for the
    # withStrideAndDilation fixture we manually pad+stride+dilate via a
    # hand-built reference (verified internally below).
    if dilation == 1:
        y = F.avg_pool1d(x_in, kernel_size=kernel_size, stride=stride,
                         padding=padding, count_include_pad=True)
    else:
        # F.avg_pool1d has no dilation arg; build the reference manually,
        # using count_include_pad=True semantics, then verify against numpy.
        y = _hand_check_avg_forward(x_in.detach(), kernel_size, stride,
                                     padding, dilation).clone()
        # Re-attach to autograd by recomputing in a way that keeps the
        # graph: use a manual gather + mean-by-divisor.
        y = _autograd_avg_with_dilation(x_in, kernel_size, stride, padding, dilation)

    expected_y = _hand_check_avg_forward(
        x_in.detach(), kernel_size, stride, padding, dilation,
    )
    assert torch.allclose(y.detach(), expected_y, atol=1e-5), (
        f"{name}: avg_pool1d forward disagrees with hand-derived (count_include_pad=True)"
    )

    if loss_grad_kind == "ones":
        gy = torch.ones_like(y)
    elif loss_grad_kind == "randn":
        torch.manual_seed(hash(name) & 0xFFFF)
        gy = torch.randn_like(y)
    else:
        raise ValueError(loss_grad_kind)

    y.backward(gy)

    return {
        "name": name,
        "x": x_in.detach(),
        "y": y.detach(),
        "dx": x_in.grad.detach(),
        "gy": gy.detach(),
        "loss_grad_kind": loss_grad_kind,
    }


def _autograd_avg_with_dilation(x_in, kernel_size, stride, padding, dilation):
    """Recompute avg pool with dilation via index_select chain so autograd
    can track gradients. count_include_pad=True semantics: divisor = kernel_size.

    Uses torch.zeros for padding (which keeps the autograd graph since
    out-of-range positions get 0 directly without indexing past x_in's bounds).
    """
    B, C, L = x_in.shape
    in_padded_L = L + 2 * padding
    eff_k = (kernel_size - 1) * dilation + 1
    out_L = (in_padded_L - eff_k) // stride + 1

    # Pad x_in with zeros on left/right.
    x_pad = F.pad(x_in, (padding, padding))  # shape: [B, C, L + 2*padding]

    # Build output by gathering windows.
    outputs = []
    for o in range(out_L):
        start = o * stride
        # Gather kernel_size positions, each at start + k*dilation in x_pad.
        col = torch.zeros((B, C), dtype=x_in.dtype)
        for k in range(kernel_size):
            idx = start + k * dilation
            col = col + x_pad[:, :, idx]
        outputs.append(col / kernel_size)
    y = torch.stack(outputs, dim=2)
    return y


def fixture_basic():
    x = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    return _run_avg_fixture("basic", x, kernel_size=2, stride=1, padding=0, dilation=1)


def fixture_multi_channel():
    torch.manual_seed(201)
    x = torch.randn(1, 3, 5)
    return _run_avg_fixture("multiChannel", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_multi_batch():
    torch.manual_seed(202)
    x = torch.randn(4, 2, 4)
    return _run_avg_fixture("multiBatch", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_with_stride_and_dilation():
    # Random lossGrad — Errata 3: stride/dilation backward needs non-uniform gy.
    torch.manual_seed(203)
    x = torch.randn(1, 1, 9)
    return _run_avg_fixture("withStrideAndDilation", x,
                            kernel_size=2, stride=3, padding=0, dilation=2,
                            loss_grad_kind="randn")


def fixture_with_same_padding():
    # K=3, S=1, D=1, padLeft=padRight=1 -> outLen = inLen = 5.
    # Edge windows: outPos=0 (validCount=2, padded left counts in divisor=3),
    # outPos=4 (validCount=2, padded right counts in divisor=3).
    x = torch.tensor([[[2.0, 4.0, 6.0, 8.0, 10.0]]])
    return _run_avg_fixture("withSamePadding", x, kernel_size=3, stride=1,
                            padding=1, dilation=1)


def fixture_edge_cases():
    # K=L=4, full-input window. outLen=1; output is x.mean() per channel.
    # validCount == kernel_size == 4. Tests the divisor == kernel_size path
    # with no truncation.
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    return _run_avg_fixture("edgeCases", x, kernel_size=4, stride=1,
                            padding=0, dilation=1)


def emit_fixture(parts, fx):
    pre = f"avgPool1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    if fx["loss_grad_kind"] != "ones":
        parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_avg_pool_1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_AVG_POOL_1D_H\n",
        "#define ODT_EXPECTED_AVG_POOL_1D_H\n",
        "#include <stdlib.h>\n\n",
    ]

    fixtures = [
        fixture_basic(),
        fixture_multi_channel(),
        fixture_multi_batch(),
        fixture_with_stride_and_dilation(),
        fixture_with_same_padding(),
        fixture_edge_cases(),
    ]
    for fx in fixtures:
        emit_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_AVG_POOL_1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
