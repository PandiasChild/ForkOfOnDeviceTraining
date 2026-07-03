# Tensor — quantization dtype semantics

Conventions for `src/tensor/**` — dtypes, quantization configs, and the
conversion matrix. Path-scoped for Claude via `.claude/rules/tensor.md`.

## SYM_INT32 is a compute format, not storage (#261)

`SYM_INT32` (int32 mantissa + one per-tensor float scale) is the framework's
**integer-compute** representation — the only integer-math path the kernels use.
It is **not** a storage format: it costs the same 4 bytes/element as `FLOAT32`
but is a single-scale fixed-point approximation, so as storage it is dominated by
both `FLOAT32` (same size, better fidelity — a per-value exponent keeps the small
magnitudes a single scale loses) and `SYM`/`ASYM` (which sub-byte-pack). The
integer math is a **transient**; nothing durable should be persisted `SYM_INT32`
to "save memory" — it saves nothing and adds error.

This bites hardest for **gradients**. Persistent parameter grads should be stored
`FLOAT32` (fidelity, same size) or `SYM`/`ASYM` (real compression); the integer
step stays transient `SYM_INT32`. The only legitimate `SYM_INT32` grads are the
transient dx/agrad operand-wires during backprop (int12, freed after the pass).

As of PR1c, the factory default for parameter grads (Linear/LayerNorm/Conv1d/
Conv1dTransposed) IS `FLOAT32` — the NULL-knob fallback that used to derive from
`propLossQ` (silently landing on `SYM_INT32` for a uniform-SYM profile) is now a
hard-pinned `FLOAT32`, closing the gap described above by default. `SYM_INT32`
parameter grads remain available and legitimate only via the explicit
`weightGradStorage`/`biasGradStorage` knob on `layerQuant_t` (#261).

## SYM ↔ * conversion bridge (#227)

`SYM` is the sub-byte bit-packed **storage** dtype; `SYM_INT32` is the int32-slot
**compute** dtype. The MCU lifecycle is store-packed (`SYM`) → unpack to int32
(`SYM_INT32`) → compute → repack. `conversionMatrix`
(`src/tensor/TensorConversion.c`) fills these cells: PR-B implements the **unpack
row** (`SYM → {SYM_INT32, FLOAT32, INT32, ASYM}`); the pack column (`* → SYM`) is
PR-C.

**Sign-extend on unpack.** `byteConversion` is a pure bit-copy that ZERO-FILLS on
widen, so a packed signed mantissa (e.g. `−3` at qBits=6 = `0b111101`) would read
back as `61`. Every `SYM →` cell routes through the shared
`unpackSignExtend(src, srcBits, dst, n)` helper, which widens then sign-extends the
two's-complement payload from `srcBits` (`(v ^ signBit) − signBit`). ASYM codes are
non-negative, so the ASYM **pack** path does not sign-extend.

**`int_repr` vs `dequantize` (deliberate, documented asymmetry).** A conversion
whose destination is `INT32` emits the integer **codes** and drops the scale
(`int_repr`); a conversion whose destination is `FLOAT32` emits the **values** with
the scale applied (`dequantize`). This mirrors PyTorch `int_repr()` vs
`dequantize()` and is consistent across both source dtypes: `SYM → INT32` and
`SYM_INT32 → INT32` are both `int_repr`; `SYM → FLOAT32` and `SYM_INT32 → FLOAT32`
are both `dequantize`. No value-rounding `→INT32` variant exists (YAGNI;
near-useless for `scale ≪ 1`).

**Rescale on the symmetric↔asymmetric transition.** `SYM → ASYM` always rescales
(dequantize → derive a fresh asym `scale`+`zeroPoint` from min/max → requantize →
pack): a symmetric code grid cannot hold an off-center `+zeroPoint` band at the
carried scale, independent of width.

**Asymmetric quantization convention (#243).** Every `* → ASYM` cell builds a float
buffer (from its own preamble) and routes through one shared helper,
`quantizeFloatToAsym` (`src/tensor/TensorConversion.c`) — the single source of truth.
Standard affine: `scale = (max − min) / (2^qBits − 1)`, `zeroPoint = round(min/scale)`,
`code = clamp(round(v/scale − zeroPoint), 0, 2^qBits − 1)` (HALF_AWAY). Dequant is
`(code + zeroPoint)·scale` — note the **additive** `zeroPoint` (ODT's sign convention,
the inverse of PyTorch's `q − zeroPoint`). A constant tensor (`min == max`) uses
`scale = (min != 0) ? |min| : 1` to avoid divide-by-zero. The denominator is
`2^qBits − 1`, **not** `2^qBits` — the latter is an off-by-one that leaves the top code
unreachable. New asym-producing converters MUST call this helper and never re-derive the
grid inline: hand-rolled copies are exactly how the four `*→ASYM` converters drifted
before #243. The float→SYM pack sibling is `packFloatBufferAsSym`.
