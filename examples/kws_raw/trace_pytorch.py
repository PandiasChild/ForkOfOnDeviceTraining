"""Per-layer trace of one controlled SGD step, mirroring kws_raw/trace_c.c.

Loads the SAME exported state_dict the C BIT_PARITY path loads, runs ONE batched
forward + backward + optimizer.step() on the fixed batch test_x[start:start+B],
and dumps every probe to dump_pt/step000/<probe>.<phase>[.sNN].npy with names
matching probe_manifest.h. PyTorch's mean-reduction backward carries a 1/B that
C's per-sample backward does not, so the unscaled tiers (act-grad, loss-grad,
grad_raw) are multiplied by B to match the C dumps.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from examples.kws_raw.train_pytorch import KwsRawCnn  # reuse the model  # noqa: E402

LR, MOMENTUM = 0.005, 0.9
# Forward probe names in C buildModel order (must equal probe_manifest.h) — 17-layer
# per-conv-LayerNorm model:
FWD_PROBES = ["pool0","conv1","ln1","relu1","pool1","conv2","ln2","relu2","pool2",
              "conv3","ln3","relu3","pool3","adaptpool","flatten","fc","softmax"]
PARAM_LAYERS = ["conv1","ln1","conv2","ln2","conv3","ln3","fc"]


def save(d: Path, probe: str, phase: str, t) -> None:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    np.save(d / f"{probe}.{phase}.npy", np.asarray(t, dtype=np.float32))


def forward_traced(model: KwsRawCnn, x: torch.Tensor, acts: dict) -> torch.Tensor:
    acts["pool0"] = (h := model.pool0(x))
    acts["conv1"] = (h := model.conv1(h)); acts["ln1"] = (h := model.ln1(h))
    acts["relu1"] = (h := F.relu(h)); acts["pool1"] = (h := F.max_pool1d(h, 4))
    acts["conv2"] = (h := model.conv2(h)); acts["ln2"] = (h := model.ln2(h))
    acts["relu2"] = (h := F.relu(h)); acts["pool2"] = (h := F.max_pool1d(h, 4))
    acts["conv3"] = (h := model.conv3(h)); acts["ln3"] = (h := model.ln3(h))
    acts["relu3"] = (h := F.relu(h)); acts["pool3"] = (h := F.max_pool1d(h, 4))
    acts["adaptpool"] = (h := F.adaptive_avg_pool1d(h, 1))
    acts["flatten"] = (h := h.flatten(start_dim=1))
    acts["fc"] = (logits := model.fc(h))
    acts["softmax"] = F.softmax(logits, dim=1)
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--act-samples", type=int, default=4)
    ap.add_argument("--classes", type=int, default=6)
    args = ap.parse_args()
    tag = f"{args.classes}class"
    data = HERE / "data" / tag
    weights = HERE / "weights" / tag

    test_x = np.load(data / "test_x.npy"); test_y = np.load(data / "test_y.npy")
    model = KwsRawCnn(args.classes)
    sd = {}
    for name in PARAM_LAYERS:
        sd[f"{name}.weight"] = torch.from_numpy(np.load(weights / f"{name}.weight.npy"))
        sd[f"{name}.bias"] = torch.from_numpy(np.load(weights / f"{name}.bias.npy"))
    model.load_state_dict(sd, strict=True)

    out = HERE / "dump_pt" / "step000"; out.mkdir(parents=True, exist_ok=True)
    sl = slice(args.sample_start, args.sample_start + args.batch)
    x = torch.from_numpy(test_x[sl].astype(np.float32))
    y = torch.from_numpy(test_y[sl].astype(np.int64))
    B = x.shape[0]              # effective batch (slice truncates at the dataset end)
    K = min(args.act_samples, B)
    if B == 0:
        raise SystemExit(f"--sample-start {args.sample_start} >= test size {len(test_x)}")

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    for name in PARAM_LAYERS:
        layer = getattr(model, name)
        save(out, name, "w_before.weight", layer.weight)
        save(out, name, "w_before.bias", layer.bias)

    acts: dict = {}
    logits = forward_traced(model, x, acts)
    for t in acts.values():
        if t.requires_grad:
            t.retain_grad()  # keep act-grads for the backward dump

    loss = F.cross_entropy(logits, y)  # reduction='mean' over the batch (÷B)
    opt.zero_grad()
    loss.backward()

    # tier 1: per-sample activation slices; keep the leading batch-dim-1 to match C [1,..]
    for probe, t in acts.items():
        a = t.detach()
        for s in range(K):
            save(out, probe, f"fwd.s{s:03d}", a[s:s + 1])
    # tier 2 + loss-grad: per-sample, ×B to undo the mean reduction (match C's unscaled grads)
    for probe, t in acts.items():
        if t.grad is None:
            continue
        for s in range(K):
            save(out, probe, f"agrad.s{s:03d}", t.grad[s:s + 1] * B)
    for s in range(K):
        save(out, "loss", f"lossgrad.s{s:03d}", acts["fc"].grad[s:s + 1] * B)

    # tier 3: grad_raw == sum (param.grad × B), grad_scaled == mean (param.grad)
    for name in PARAM_LAYERS:
        layer = getattr(model, name)
        save(out, name, "grad_raw.weight", layer.weight.grad * B)
        save(out, name, "grad_raw.bias", layer.bias.grad * B)
        save(out, name, "grad_scaled.weight", layer.weight.grad)
        save(out, name, "grad_scaled.bias", layer.bias.grad)

    opt.step()
    for name in PARAM_LAYERS:
        layer = getattr(model, name)
        save(out, name, "w_after.weight", layer.weight)
        save(out, name, "w_after.bias", layer.bias)
    print(f"trace_pytorch: mean_loss={loss.item():.6f} -> {out}")


if __name__ == "__main__":
    main()
