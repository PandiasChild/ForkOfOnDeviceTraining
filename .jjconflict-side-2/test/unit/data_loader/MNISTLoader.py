import torch
import numpy as np
from torchvision import datasets, transforms
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = 10


def export_mnist(split, out_prefix):
    assert split in ("train", "test")

    dataset = datasets.MNIST(
        root="./data",
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )

    N = len(dataset)

    images = np.empty((N, 1, 28, 28), dtype=np.float32)
    labels = np.zeros((N, NUM_CLASSES), dtype=np.float32)

    for i in range(N):
        x, y = dataset[i]
        images[i] = x.numpy()
        labels[i, y] = 1.0

    np.save(os.path.join(BASE_DIR, f"{out_prefix}_{split}_x.npy"), images)
    np.save(os.path.join(BASE_DIR, f"{out_prefix}_{split}_y.npy"), labels)


if __name__ == "__main__":
    export_mnist("train", "mnist")
    export_mnist("test", "mnist")
