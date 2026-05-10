# HAR Classifier — PyTorch + C Parity Demo

Trains a 6-class human-activity classifier on the UCI HAR dataset using the
1D-CNN layers exposed by both PyTorch (reference) and the ODT C framework.

## Run it

```bash
# 1. Prepare data (downloads ~58 MB the first time; cached under data/raw/)
uv run python examples/har_classifier/prepare_data.py

# 2. Train PyTorch reference (~30s on CPU)
uv run python examples/har_classifier/train_pytorch.py

# 3. Build + run C training (~2.5 min)
cmake --preset examples
cmake --build --preset examples --target train_c_har_classifier
./build/examples/examples/har_classifier/train_c_har_classifier

# 4. Compare runs and emit plots (exits non-zero if parity fails)
uv run python examples/har_classifier/compare.py
```

## Outputs

After all four steps, `examples/har_classifier/` contains:
- `data/{train,val,test}_{x,y}.npy`
- `logs/{pytorch,c}.json`
- `outputs/{pytorch,c}_predictions.npy`
- `plots/{loss_curves,accuracy_curves,confusion_matrix_pt,confusion_matrix_c}.png`

## Model

- Input: `[9, 128]` (9 IMU channels, 2.56 s window @ 50 Hz)
- 3 × `Conv1d → ReLU → MaxPool1d` blocks
- Global `AvgPool1d` → `Flatten → Linear → Softmax → CrossEntropy`
- ~10 K parameters

## Parity tolerance

| Metric | Tolerance |
|---|---|
| test_acc  | ±2.5 pp absolute |
| test_loss | ±0.15 nats absolute |

Both implementations use independent random init; the loss tolerance is
empirically calibrated. See `examples/_shared/DETERMINISM.md` for the full
determinism contract.
