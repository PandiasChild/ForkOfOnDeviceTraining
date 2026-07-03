import json
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#wir haben hier with und without delta im vergleich
#es ist pro epoche der loss.
#kann man auch mit accuracy machen.
#außerdem fehlt noch x achse delta reduction mit y achse dann die accuracy


# -----------------------------
# Dateien einlesen
# -----------------------------
without_files = sorted(glob.glob("logs/without_delta/trial_*.log"))
with_files = sorted(glob.glob("logs/with_delta/trial_*.log"))

data = {}

for filename in without_files + with_files:
    with open(filename, "r") as f:
        log = json.load(f)

    cfg = log["config"]
    trial = cfg["trial_number"]
    delta = cfg["delta_status"]

    epochs = [e["epoch"] for e in log["epochs"]]
    losses = [e["train_loss"] for e in log["epochs"]]
    # Für Validation-Loss stattdessen:
    # losses = [e["val_loss"] for e in log["epochs"]]

    if trial not in data:
        data[trial] = {}

    data[trial][delta] = (epochs, losses)

# -----------------------------
# Farben erzeugen
# -----------------------------
trials = sorted(data.keys())
n_trials = len(trials)

# gleiche Helligkeit für gleiche Trialnummer
blues = cm.Blues(np.linspace(0.35, 0.95, n_trials))
reds = cm.Reds(np.linspace(0.35, 0.95, n_trials))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))

legend_done = {0: False, 1: False}

for i, trial in enumerate(trials):

    if 0 in data[trial]:
        epochs, losses = data[trial][0]
        plt.plot(
            epochs,
            losses,
            color=blues[i],
            linewidth=2,
            label="without delta" if not legend_done[0] else None
        )
        legend_done[0] = True

    if 1 in data[trial]:
        epochs, losses = data[trial][1]
        plt.plot(
            epochs,
            losses,
            color=reds[i],
            linewidth=2,
            linestyle="--",
            label="with delta" if not legend_done[1] else None
        )
        legend_done[1] = True

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss über die Epochen")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()