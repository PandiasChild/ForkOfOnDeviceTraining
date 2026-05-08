#!/usr/bin/env python3
"""Generate expected_max_pool_1d.h for UnitTestMaxPool1d (Layer-3 tests).

Produces a C header with PyTorch-derived ground-truth values for each
test fixture: forward output, expected argmax indices (int32_t),
propLoss (dL/dx). For most fixtures lossGrad = torch.ones_like(y);
the `withStrideAndDilation` fixture uses torch.randn_like(y) (and
emits the lossGrad as a gold array) so that index-permuting mutations
on the stride/dilation backward path are non-vacuous (per
codebase_uniform_lossgrad_mutation_vacuity).

Run via `uv run` (CMake wires this automatically).
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


def emit_int32_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().to(torch.int32).flatten().tolist()
    body = ", ".join(str(v) for v in flat)
    return (
        f"static const int32_t {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(flat)};\n"
    )


def _hand_check_max_forward(x, kernel_size, stride, padding, dilation):
    """Numpy-level sanity check: for each output position, max over the
    valid window. Used as a self-check against PyTorch's MaxPool1d output."""
    import math
    B, C, L = x.shape
    in_padded_L = L + 2 * padding
    eff_k = (kernel_size - 1) * dilation + 1
    out_L = (in_padded_L - eff_k) // stride + 1
    out = torch.full((B, C, out_L), -math.inf)
    for b in range(B):
        for c in range(C):
            for o in range(out_L):
                start = o * stride - padding
                vals = []
                for k in range(kernel_size):
                    idx = start + k * dilation
                    if 0 <= idx < L:
                        vals.append(x[b, c, idx].item())
                if vals:
                    out[b, c, o] = max(vals)
                else:
                    out[b, c, o] = 0.0
    return out


def _run_max_fixture(name, x, *, kernel_size, stride, padding, dilation,
                     loss_grad_kind="ones"):
    """Run forward + autograd-backward on a MaxPool1d fixture; return all arrays.

    loss_grad_kind: "ones" -> lossGrad = torch.ones_like(y) (default);
                    "randn" -> lossGrad = torch.randn_like(y), reproducible via seed.
    """
    x_in = x.clone().detach().requires_grad_(True)

    # PyTorch's nn.functional.max_pool1d returns (output) by default; with
    # return_indices=True it returns (output, indices). The indices are 1D
    # input positions per output position (no batch/channel offset), matching
    # our argmax convention exactly.
    y, idx = F.max_pool1d(
        x_in, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, return_indices=True,
    )

    # Self-check forward against hand-computed reference.
    expected_y = _hand_check_max_forward(
        x_in.detach(), kernel_size, stride, padding, dilation,
    )
    assert torch.allclose(y.detach(), expected_y, atol=1e-5), (
        f"{name}: PyTorch max_pool1d forward disagrees with hand-derived reference"
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
        "argmax": idx.detach(),    # int64 in PyTorch; we cast to int32 in emit_int32_array
        "dx": x_in.grad.detach(),
        "gy": gy.detach(),
        "loss_grad_kind": loss_grad_kind,
    }


def fixture_basic():
    # B=1, C=1, L=4, K=2, stride=1, VALID
    x = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    return _run_max_fixture("basic", x, kernel_size=2, stride=1, padding=0, dilation=1)


def fixture_multi_channel():
    # B=1, C=3, L=5, K=2, stride=1, VALID
    torch.manual_seed(101)
    x = torch.randn(1, 3, 5)
    return _run_max_fixture("multiChannel", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_multi_batch():
    # B=4, C=2, L=4, K=2, stride=1, VALID
    torch.manual_seed(102)
    x = torch.randn(4, 2, 4)
    return _run_max_fixture("multiBatch", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_with_stride_and_dilation():
    # B=1, C=1, L=9, K=2, stride=3, dilation=2, VALID
    # Effective kernel span = (2-1)*2+1 = 3; output length = (9-3)/3+1 = 3.
    # Using random lossGrad (Errata 3) so positional mutations are non-vacuous.
    torch.manual_seed(103)
    x = torch.randn(1, 1, 9)
    return _run_max_fixture("withStrideAndDilation", x,
                            kernel_size=2, stride=3, padding=0, dilation=2,
                            loss_grad_kind="randn")


def fixture_with_same_padding():
    # B=1, C=1, L=5, K=3, stride=1, dilation=1.
    # SAME for K=3, S=1, D=1 -> total pad = 2; padLeft=1, padRight=1 (symmetric).
    # Output length = 5. Edge windows at outPos=0 (validCount=2, drops left)
    # and outPos=4 (validCount=2, drops right) — exercises Errata 4.
    x = torch.tensor([[[5.0, 1.0, 4.0, 2.0, 6.0]]])
    return _run_max_fixture("withSamePadding", x, kernel_size=3, stride=1,
                            padding=1, dilation=1)


def fixture_edge_cases():
    # B=1, C=1, L=4, K=1, stride=1, VALID. K=1 -> identity pool;
    # output equals input, argmax equals position index, propLoss = lossGrad.
    # Ensures the validCount-loop with N=1 doesn't have an off-by-one
    # (e.g., loop bound `i < validCount - 1` would compute -INFINITY here).
    x = torch.tensor([[[7.0, -2.0, 3.0, 0.5]]])
    return _run_max_fixture("edgeCases", x, kernel_size=1, stride=1,
                            padding=0, dilation=1)


def emit_fixture(parts, fx):
    pre = f"maxPool1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_int32_array(f"expectedArgmax_{pre}", fx["argmax"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    if fx["loss_grad_kind"] != "ones":
        parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_max_pool_1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_MAX_POOL_1D_H\n",
        "#define ODT_EXPECTED_MAX_POOL_1D_H\n",
        "#include <stdint.h>\n",
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

    parts.append("\n#endif // ODT_EXPECTED_MAX_POOL_1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
