# HAR Classifier — PyTorch + C Parity Demo

Trains a 6-class human-activity classifier on the UCI HAR dataset using the
1D-CNN layers exposed by both PyTorch (reference) and the ODT C framework. The
C model is built with the factory layer API (`conv1dLayerInit` + `layerQuant_t`)
and loads PyTorch weights through `StateDictApi`.

One binary, two verification modes:

- **Bit-parity** (what CI runs): `BIT_PARITY=1` loads PyTorch's trained weights
  into the C model and runs inference only — the C predictions must be
  **bit-identical** to PyTorch's. Deterministic and exact.
- **Train-from-scratch demo**: with no env var the C model trains from its own
  random init; `compare.py` checks final-state parity within tolerance and emits
  plots. Independent init, so it verifies *convergence*, not bits.

## Run it

```bash
# 1. Prepare data (downloads ~58 MB the first time; cached under data/raw/)
uv run python examples/har_classifier/prepare_data.py

# 2. Train the PyTorch reference + export weights (~30s on CPU)
uv run python examples/har_classifier/train_pytorch.py

# 3. Build the C trainer
cmake --preset examples
cmake --build --preset examples --target train_c_har_classifier

# 4a. Bit-parity check (exact — this is the CI gate)
BIT_PARITY=1 ./build/examples/examples/har_classifier/train_c_har_classifier
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/har_classifier/outputs/pytorch_predictions.npy \
  --c examples/har_classifier/outputs/c_predictions.npy --dtype int32

# 4b. …or the train-from-scratch demo + plots (several minutes)
./build/examples/examples/har_classifier/train_c_har_classifier
uv run python examples/har_classifier/compare.py
```

## Outputs

After the train-from-scratch demo, `examples/har_classifier/` contains:
- `data/{train,val,test}_{x,y}.npy`
- `logs/{pytorch,c}.json`
- `outputs/{pytorch,c}_predictions.npy`
- `plots/{loss_curves,accuracy_curves,confusion_matrix_pt,confusion_matrix_c}.png`

## Model

- Input: `[9, 128]` (9 IMU channels, 2.56 s window @ 50 Hz)
- 2 × `Conv1d → ReLU → MaxPool1d` blocks, then `Conv1d → ReLU`
- Global `AvgPool1d` → `Flatten → Linear → Softmax → CrossEntropy`
- ~10 K parameters

## Parity tolerance (train-from-scratch demo)

| Metric | Tolerance |
|---|---|
| test_acc  | ±2.5 pp absolute |
| test_loss | ±0.15 nats absolute |

The demo's two implementations use independent random init; the loss tolerance
is empirically calibrated. Bit-parity mode requires exact equality instead.
See `examples/_shared/DETERMINISM.md` for the full determinism contract.

## Packed-SYM weight-quantization memory study

A second trainer, `train_c_har_classifier_sym` (source `train_c_sym.c`), trains
the *same* model with weights + biases stored **packed sub-byte SYM@x** while the
whole backward pass, gradients, and optimizer momentum stay FLOAT32. It quantifies
the on-device memory cost/benefit of weight quantization across widths, against the
FLOAT32 trainer as the reference.

Key design points (see `docs/CONVENTIONS.md` and the source comments):

- **Weights + biases** are packed `SYM@x` (`ceil(x·N/8)` bytes → @12 = 62.5%,
  @8 = 75%, @4 = 87.5% smaller than FLOAT32's `4N`). Everything else — gradients,
  momentum, activation wires — is FLOAT32.
- **Momentum is FLOAT32**, decoupled from the packed-SYM params via the optimizer's
  own-config knob (`sgdMCreateOptim(..., momentumQuant)`). A packed-SYM momentum
  would re-quantize the velocity to the same coarse levels as the weights.
- **Stochastic rounding** (`SR_HALF_AWAY`) on the packed-SYM param storage lets a
  FLOAT32 gradient step smaller than one SYM level move the weight *in expectation*
  (the #279 dead-zone escape). `SYM_ROUNDING=det` forces deterministic rounding to
  A/B that claim — which is still a **hypothesis** until the ≥10-seed sweep confirms it.

### Build with memory profiling

Memory instrumentation is compiled in only under the `examples_memprofile` preset
(`-DODT_MEM_PROFILE`); the plain `examples` preset (the CI bit-parity build) is
byte-identical bare calloc/free.

```bash
cmake --preset examples_memprofile
cmake --build --preset examples_memprofile --target \
    train_c_har_classifier train_c_har_classifier_sym
```

Both binaries are env-configured: `SEED`, `EPOCHS`, `LR`, `MOMENTUM`, `LOG_PATH`
(+ `SYM_BITS`, `SYM_ROUNDING` for the SYM binary). Each writes an extended RunLog
JSON whose `memory` block carries per-category analytic bytes, instrumented
heap/stack/RSS peaks, and the **reconciliation gap** (`heap_peak − mcu_total`,
≈ the host-resident dataset the MCU would stream — recorded, never massaged).

### Offline sweep + honest aggregation

```bash
# Full study: {float, sym@12, sym@10, sym@8, sym@6, sym@4} × seed 1..10 = 60 runs.
# LONG (~40 h offline; NOT wired into CI). Use --configs/--seeds/--epochs to smoke.
uv run examples/har_classifier/run_matrix.py                 # full
uv run examples/har_classifier/run_matrix.py --configs float sym8 --seeds 1 2 --epochs 2  # smoke

# Aggregate 10-seed mean±std + comparison table + 3 pictograms.
uv run examples/har_classifier/compare_memory.py --plots
```

**Read the numbers honestly.** The training loop streams the macro-batch one sample
at a time (**micro-batch B=1**; gradients accumulate at the optimizer), so the
concurrent activation peak is *one sample's* worth — not 64×. That makes the on-device
footprint roughly balanced: at FLOAT32 (~181 KB total) params, grads, and momentum are
each ~22% and activations ~31%. At SYM@8 the weight *category* shrinks **75%** (40 KB →
10 KB), which translates to a **material ~17% drop in total on-device training RAM**
(~181 KB → ~151 KB; ~19% at SYM@4). `compare_memory.py` reports the weight-category
drop and the total-footprint drop **separately** — they answer different questions. The
next wins are grads and momentum (each another ~22%), reachable via the optimizer's
per-config quant knob. Headline claims come **only** from the ≥10-seed aggregate; a
`--min-seeds` guard warns loudly on smoke-sized runs.

Sweep artifacts (`logs/`, `outputs/memory_summary.json`, `plots/har_mem_*.png`) are
gitignored like every other example's results — regenerate them locally.
