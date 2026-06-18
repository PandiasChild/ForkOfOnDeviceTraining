#!/usr/bin/env python3
"""Generate expected_conv1d.h for UnitTestConv1d (Layer-3 tests).

Produces a C header with PyTorch-derived ground-truth values for each
test fixture: forward output, plus autograd-derived dL/dx, dL/dW,
dL/db (when bias is present). Run via `uv run` (CMake wires this
automatically).

Most FLOAT fixtures use lossGrad = torch.ones_like(y); the pointwise and
explicit-padding fixtures (and all SYM grad fixtures) use a random
(torch.randn) lossGrad to avoid output-channel vacuity in the gradients.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
import sym_gold
from sym_gold import (
    round_half_away,
    stable_dequant,
    emit_int32_array,
    emit_float_scalar,
    emit_int32_scalar,
)

QMAX_F32 = torch.tensor(32767.0, dtype=torch.float32)
QMIN_F32 = torch.tensor(-32768.0, dtype=torch.float32)

QMAX_I12 = torch.tensor(2047.0, dtype=torch.float32)
QMIN_I12 = torch.tensor(-2048.0, dtype=torch.float32)


def quantize_sym_i12(x):
    """int12 operand quantize (qMaxBits=12): absmax->scale=absmax/2047, round-clamp half-away."""
    absmax = x.abs().max().item()
    scale = 1.0 if absmax == 0.0 else absmax / 2047.0
    q = round_half_away(torch.clamp(x / scale, -2048.0, 2047.0))
    return q.to(torch.int32), scale


def stable_dequant_i12(x):
    q, s = quantize_sym_i12(x.to(torch.float64))
    deq32 = (q.to(torch.float64) * s).to(torch.float32)
    q2, _ = quantize_sym_i12(deq32.to(torch.float64))
    assert torch.equal(q, q2), "int12 fixture not dequant round-trip stable"
    return q, s, deq32


def f32_scale_i12(deq32: torch.Tensor) -> float:
    """int12 scale the C runtime derives from the emitted fixture: absmax_f32/2047 in float32."""
    absmax = deq32.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return 1.0
    return (absmax / QMAX_I12).item()


# stable_dequant (sym_gold.py) returns its scale in float64; the C runtime
# (convertFloatTensorToSymInt32Tensor) derives the scale in float32 from the
# emitted fixture. Re-derive in float32 here so the emulation's scales match C
# bit-for-bit; the round-trip-stable fixture guarantees the same mantissas.
def f32_scale(deq32: torch.Tensor) -> float:
    """Scale the C runtime derives from the EMITTED float32 fixture:
    float32(absmax(deq32)) / 32767, all in float32 (convertFloatTensorToSymInt32Tensor)."""
    absmax = deq32.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return 1.0
    return (absmax / QMAX_F32).item()


def f32_mul(a: float, b: float) -> float:
    """One IEEE-754 float32 multiply (matches the C `aScale * bScale`)."""
    return (torch.tensor(a, dtype=torch.float32) * torch.tensor(b, dtype=torch.float32)).item()


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
        "_params": {"stride": stride, "padding": padding, "dilation": dilation,
                    "groups": groups, "has_bias": b is not None},
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
        "_params": {"stride": 1, "padding": 0, "dilation": 1, "groups": 1, "has_bias": True},
    }


def fixture_explicit_padding():
    # ECG enc1 geometry (issue #177): Conv1d K=7, stride=2, EXPLICIT symmetric
    # padding=3 — a stride>1 conv whose integer padding SAME cannot express.
    # Exercises forward AND backward; the backward delegates to the transposed-conv
    # adjoint, which must also honour the explicit pad (otherwise training crashes).
    # NON-UNIFORM lossGrad so dL/dW pins the output channel: a uniform lossGrad
    # makes every weight-grad row identical across oc (see fixture_pointwise).
    torch.manual_seed(7)
    x = torch.randn(1, 2, 10).clone().detach().requires_grad_(True)
    w = torch.randn(3, 2, 7).clone().detach().requires_grad_(True)
    b = torch.randn(3).clone().detach().requires_grad_(True)
    # output length = (10 + 2*3 - 7)//2 + 1 = 5
    loss_grad = torch.randn(1, 3, 5)
    y = F.conv1d(x, w, bias=b, stride=2, padding=3, dilation=1, groups=1)
    y.backward(loss_grad)
    return {
        "name": "explicitPadding",
        "x": x.detach(),
        "w": w.detach(),
        "b": b.detach(),
        "y": y.detach(),
        "dx": x.grad.detach(),
        "dw": w.grad.detach(),
        "db": b.grad.detach(),
        "lossGrad": loss_grad.detach(),
        "_params": {"stride": 2, "padding": 3, "dilation": 1, "groups": 1, "has_bias": True},
    }


def fixture_stride_dilation_sym():
    # SYM twin of strideDilation with a RANDOM lossGrad (uniform-lossGrad vacuity).
    torch.manual_seed(11)
    x = torch.tensor([[[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0]]]).requires_grad_(True)
    w = torch.tensor([[[2.0, 4.0]]]).requires_grad_(True)
    loss_grad = torch.randn(1, 1, 3)
    y = F.conv1d(x, w, bias=None, stride=3, padding=0, dilation=2, groups=1)
    y.backward(loss_grad)
    return {"name": "strideDilationSym", "x": x.detach(), "w": w.detach(), "b": None,
            "y": y.detach(), "dx": x.grad.detach(), "dw": w.grad.detach(), "db": None,
            "lossGrad": loss_grad.detach(),
            "_params": {"stride": 3, "padding": 0, "dilation": 2, "groups": 1, "has_bias": False}}


def fixture_groups_grouped_sym():
    # SYM twin of groupsGrouped with a RANDOM lossGrad.
    torch.manual_seed(13)
    x = torch.randn(1, 4, 5).requires_grad_(True)
    w = torch.randn(8, 2, 2).requires_grad_(True)   # [Cout, Cin/groups=2, K]
    b = torch.randn(8).requires_grad_(True)
    loss_grad = torch.randn(1, 8, 4)
    y = F.conv1d(x, w, bias=b, stride=1, padding=0, dilation=1, groups=2)
    y.backward(loss_grad)
    return {"name": "groupsGroupedSym", "x": x.detach(), "w": w.detach(), "b": b.detach(),
            "y": y.detach(), "dx": x.grad.detach(), "dw": w.grad.detach(), "db": b.grad.detach(),
            "lossGrad": loss_grad.detach(),
            "_params": {"stride": 1, "padding": 0, "dilation": 1, "groups": 2, "has_bias": True}}


SYM_FIXTURES = [
    fixture_single_channel_single_batch(),   # reused FLOAT: VALID, no bias, ones (allowed)
    fixture_single_channel_with_bias(),      # reused FLOAT: VALID, bias, ones (allowed)
    fixture_pointwise(),                      # reused FLOAT: K=1, bias, RANDOM lossGrad
    fixture_groups_grouped_sym(),            # new _sym twin: groups=2, bias, RANDOM lossGrad
    fixture_stride_dilation_sym(),           # new _sym twin: stride=3 dilation=2, no bias, RANDOM
    fixture_explicit_padding(),              # reused FLOAT: K=7 stride=2 EXPLICIT pad=3, bias, RANDOM
]


def _conv_params(fx):
    """Recover the conv hyper-params a fixture ran with (mirrors its _run_fixture call)."""
    return fx["_params"]  # {stride, padding, dilation, groups, has_bias}


def _quant_inputs(fx):
    """int12-quantize x/w/(b)/lossGrad operands; return mantissas, float32 scales, float fixtures.
    Operands use int12 (qMaxBits=12) so int12*int12 products stay within int32 headroom.
    Grad accumulators (weightGrad) stay int16 via _requant_absmax_f32."""
    xq, _, x_deq = stable_dequant_i12(fx["x"])
    wq, _, w_deq = stable_dequant_i12(fx["w"])
    s_in = f32_scale_i12(x_deq)
    s_w = f32_scale_i12(w_deq)
    bq = b_deq = s_b = None
    if fx["b"] is not None:
        bq, _, b_deq = stable_dequant_i12(fx["b"])
        s_b = f32_scale_i12(b_deq)
    gy = fx["lossGrad"] if fx.get("lossGrad") is not None else torch.ones_like(fx["y"])
    gyq, _, gy_deq = stable_dequant_i12(gy)
    s_loss = f32_scale_i12(gy_deq)
    return dict(xq=xq, wq=wq, bq=bq, gyq=gyq, s_in=s_in, s_w=s_w, s_b=s_b, s_loss=s_loss,
                x_deq=x_deq, w_deq=w_deq, b_deq=b_deq, gy_deq=gy_deq)


def _int_mac(q, p):
    """Exact integer forward + autograd grads via float64 conv of integer mantissas.
    Returns (fwd_mac[B,Cout,L] int64, dw_mac int64 [gather], dx_mac int64 [scatter])."""
    xqf = q["xq"].to(torch.float64).requires_grad_(True)
    wqf = q["wq"].to(torch.float64).requires_grad_(True)
    y = F.conv1d(xqf, wqf, bias=None, stride=p["stride"], padding=p["padding"],
                 dilation=p["dilation"], groups=p["groups"])
    y.backward(q["gyq"].to(torch.float64))
    fwd_mac = y.detach().round().to(torch.int64)
    dw_mac = wqf.grad.round().to(torch.int64)
    dx_mac = xqf.grad.round().to(torch.int64)
    return fwd_mac, dw_mac, dx_mac


def _round_away_f32(x: torch.Tensor) -> torch.Tensor:
    return round_half_away(x.to(torch.float32))


def _requant_absmax_f32(mac_int: torch.Tensor, in_scale: float):
    """addSymInt32TensorsInplace into a zero grad: dequant (f32) -> absmax (f32) ->
    scale=absmax/32767 (f32) -> round_half_away(clamp(.../scale)). Returns (q int32, scale)."""
    deq = mac_int.to(torch.float32) * torch.tensor(in_scale, dtype=torch.float32)
    absmax = deq.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return torch.zeros_like(mac_int, dtype=torch.int32), 1.0
    scale = (absmax / QMAX_F32)
    q = round_half_away(torch.clamp(deq / scale, QMIN_F32, QMAX_F32))
    return q.to(torch.int32), scale.item()


def _autograd_ref_f64(q, p):
    """float64 torch-autograd on the EXACT dequantized inputs (q*s in f64) — the true
    grads the SYM dequants must track within analytic bounds."""
    x64 = (q["xq"].to(torch.float64) * q["s_in"]).requires_grad_(True)
    w64 = (q["wq"].to(torch.float64) * q["s_w"]).requires_grad_(True)
    b64 = None
    if q["bq"] is not None:
        b64 = (q["bq"].to(torch.float64) * q["s_b"]).requires_grad_(True)
    y = F.conv1d(x64, w64, bias=b64, stride=p["stride"], padding=p["padding"],
                 dilation=p["dilation"], groups=p["groups"])
    y.backward(q["gyq"].to(torch.float64) * q["s_loss"])
    return (y.detach(), x64.grad, w64.grad, b64.grad if b64 is not None else None)


def emulate_sym_conv(fx):
    """Full SYM emulation of conv1dForward + conv1dBackward for one fixture.
    Returns a dict of int32 mantissas + float32 scales + float dequant references + tols."""
    # Rounding-mode canary: half-AWAY must round 0.5 -> 1 and -0.5 -> -1 (NOT
    # half-to-even's 0). Fires unconditionally if _round_away_f32 is ever a
    # half-even rounder — the per-fixture tolerances cannot catch this on
    # non-tie fixtures, so this guards the rounding path directly.
    _canary = _round_away_f32(torch.tensor([0.5, -0.5], dtype=torch.float32))
    assert _canary.tolist() == [1.0, -1.0], \
        "round_half_away is not half-away-from-zero — rounding mode wrong"
    p = _conv_params(fx)
    q = _quant_inputs(fx)
    fwd_mac, dw_mac, dx_mac = _int_mac(q, p)
    out_scale = f32_mul(q["s_in"], q["s_w"])          # forward output scale
    dx_scale = f32_mul(q["s_loss"], q["s_w"])         # propLoss scale = s_loss * s_w
    inter_scale = f32_mul(q["s_loss"], q["s_in"])     # weightGrad intermediate scale

    # forward: int MAC + per-channel bias seed (float32 refold)
    out_channels = fwd_mac.shape[1]
    if q["bq"] is not None:
        seed = _round_away_f32(q["bq"].to(torch.float32)
                               * torch.tensor(q["s_b"], dtype=torch.float32)
                               / torch.tensor(out_scale, dtype=torch.float32)).to(torch.int64)
        fwd_q = (fwd_mac + seed.reshape(1, out_channels, 1)).to(torch.int32)
        fwd_mtol = 1
    else:
        fwd_q = fwd_mac.to(torch.int32)
        fwd_mtol = 0
    fwd_deq = fwd_q.to(torch.float32) * torch.tensor(out_scale, dtype=torch.float32)

    # dx (propLoss): raw scatter, no requant, no bias
    dx_q = dx_mac.to(torch.int32)
    dx_deq = dx_q.to(torch.float32) * torch.tensor(dx_scale, dtype=torch.float32)

    # weightGrad: strategy A requant of the int gather from a zero grad
    dw_q, dw_scale = _requant_absmax_f32(dw_mac, inter_scale)

    # biasGrad: int sum over (batch, outLen) per oc, fixed-scale rescale (s_bg = 1.0, zero-init grad).
    # The division below mirrors rescaleIntoAccumulatorScale(sum_int, s_loss, s_bg=1.0, HALF_AWAY) =
    # roundByMode(sum_int*s_loss/1.0, HALF_AWAY) — the exact op Task 6's C calls. The #189 ODT_SEED_GUARD
    # is inactive at these fixture scales (no overflow), so the guarded helper and this raw cast agree.
    db_q = db_scale = None
    if q["bq"] is not None:
        sum_int = q["gyq"].sum(dim=(0, 2))            # exact int per output channel
        db_q = _round_away_f32(sum_int.to(torch.float32)
                               * torch.tensor(q["s_loss"], dtype=torch.float32)
                               / torch.tensor(1.0, dtype=torch.float32)).to(torch.int32)
        db_scale = 1.0

    # float64 autograd references for the dequant checks
    ref_y, ref_dx, ref_dw, ref_db = _autograd_ref_f64(q, p)

    # ---- self-checks (fail at generation time, not in C) ----
    # float32 precision: when fwd_q or dx_q mantissas exceed 2^24, the int32->float32 cast
    # loses low bits. The C dequant (int32_val * scale) in float32 has the same precision
    # as our emulation, so the tolerance must include this f32 ULP term.
    _eps32 = float(torch.finfo(torch.float32).eps)
    fwd_f32_prec = fwd_q.abs().max().item() * out_scale * _eps32
    dx_f32_prec = dx_q.abs().max().item() * dx_scale * _eps32

    # forward dequant tracks the true (quantized-input) conv within quantization noise + f32 ULP
    fwd_err = (fwd_deq.to(torch.float64) - ref_y).abs().max().item()
    fwd_tol = 8.0 * out_scale + fwd_f32_prec + 1e-9
    assert fwd_err <= fwd_tol, f"{fx['name']}: fwd emulation err {fwd_err} > tol {fwd_tol}"
    # dx dequant tracks autograd dx (+ f32 ULP term for large mantissas)
    dx_err = (dx_deq.to(torch.float64) - ref_dx).abs().max().item()
    dx_tol = 8.0 * dx_scale + dx_f32_prec + 1e-9
    assert dx_err <= dx_tol, f"{fx['name']}: dx emulation err {dx_err} > tol {dx_tol}"
    # weightGrad dequant tracks autograd dw (increment-quant + requant + drift)
    dw_err = (dw_q.to(torch.float64) * dw_scale - ref_dw.reshape(-1).reshape(dw_q.shape)
              ).abs().max().item()
    assert dw_err <= (0.5 * inter_scale + 2.5 * dw_scale) * 1.5 + 1e-9, \
        f"{fx['name']}: dw emulation err {dw_err}"
    if db_q is not None:
        db_err = (db_q.to(torch.float64) * db_scale - ref_db.reshape(-1)).abs().max().item()
        assert db_err <= (0.5 * q["s_loss"] + 2.5 * 1.0) * 1.5 + 1e-9, \
            f"{fx['name']}: db emulation err {db_err}"

    return dict(
        name=fx["name"], has_bias=q["bq"] is not None,
        x_deq=q["x_deq"], w_deq=q["w_deq"], b_deq=q["b_deq"], gy_deq=q["gy_deq"],
        fwd_q=fwd_q, fwd_scale=out_scale, fwd_deq=ref_y.to(torch.float32),
        fwd_mtol=fwd_mtol, fwd_dtol=8.0 * out_scale + fwd_f32_prec + 1e-6,
        dx_q=dx_q, dx_scale=dx_scale, dx_deq=ref_dx.to(torch.float32),
        dx_mtol=0, dx_dtol=8.0 * dx_scale + dx_f32_prec + 1e-6,
        dw_q=dw_q, dw_scale=dw_scale, dw_deq=ref_dw.to(torch.float32),
        dw_mtol=2, dw_dtol=(0.5 * inter_scale + 2.5 * dw_scale) * 1.5 + 1e-6,
        db_q=db_q, db_scale=db_scale,
        db_deq=ref_db.to(torch.float32) if db_q is not None else None,
        db_mtol=1, db_dtol=(0.5 * q["s_loss"] + 2.5 * 1.0) * 1.5 + 1e-6,
    )


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


def emit_sym_fixture(parts, fx):
    g = emulate_sym_conv(fx)
    pre = f"conv1dSym_{g['name']}"
    parts.append(emit_float_array(f"input_{pre}", g["x_deq"]))
    parts.append(emit_float_array(f"weight_{pre}", g["w_deq"]))
    if g["has_bias"]:
        parts.append(emit_float_array(f"bias_{pre}", g["b_deq"]))
    parts.append(emit_float_array(f"lossGrad_{pre}", g["gy_deq"]))
    # forward
    parts.append(emit_int32_array(f"expectedForward_{pre}", g["fwd_q"]))
    parts.append(emit_float_scalar(f"expectedForwardScale_{pre}", g["fwd_scale"]))
    parts.append(emit_int32_scalar(f"forwardMantissaTol_{pre}", g["fwd_mtol"]))
    parts.append(emit_float_array(f"expectedForwardDequant_{pre}", g["fwd_deq"]))
    parts.append(emit_float_scalar(f"forwardDequantTol_{pre}", g["fwd_dtol"]))
    # dx / propLoss
    parts.append(emit_int32_array(f"expectedPropLoss_{pre}", g["dx_q"]))
    parts.append(emit_float_scalar(f"expectedPropLossScale_{pre}", g["dx_scale"]))
    parts.append(emit_int32_scalar(f"propLossMantissaTol_{pre}", g["dx_mtol"]))
    parts.append(emit_float_array(f"expectedPropLossDequant_{pre}", g["dx_deq"]))
    parts.append(emit_float_scalar(f"propLossDequantTol_{pre}", g["dx_dtol"]))
    # weightGrad
    parts.append(emit_int32_array(f"expectedWeightGrad_{pre}", g["dw_q"]))
    parts.append(emit_float_scalar(f"expectedWeightGradScale_{pre}", g["dw_scale"]))
    parts.append(emit_int32_scalar(f"weightGradMantissaTol_{pre}", g["dw_mtol"]))
    parts.append(emit_float_array(f"expectedWeightGradDequant_{pre}", g["dw_deq"]))
    parts.append(emit_float_scalar(f"weightGradDequantTol_{pre}", g["dw_dtol"]))
    # biasGrad
    if g["has_bias"]:
        parts.append(emit_int32_array(f"expectedBiasGrad_{pre}", g["db_q"]))
        parts.append(emit_float_scalar(f"expectedBiasGradScale_{pre}", g["db_scale"]))
        parts.append(emit_int32_scalar(f"biasGradMantissaTol_{pre}", g["db_mtol"]))
        parts.append(emit_float_array(f"expectedBiasGradDequant_{pre}", g["db_deq"]))
        parts.append(emit_float_scalar(f"biasGradDequantTol_{pre}", g["db_dtol"]))


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
        fixture_explicit_padding(),
    ]
    for fx in fixtures:
        emit_fixture(parts, fx)

    for fx in SYM_FIXTURES:
        emit_sym_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_CONV1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
