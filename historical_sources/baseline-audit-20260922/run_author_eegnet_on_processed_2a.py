"""Author-code EEGNet sanity probe on the existing BCIC-IV-2a NumPy arrays.

This runs FBCNet commit de1bbdd's EEGNet and two-stage trainer unchanged. The
only adapter replaces the authors' per-trial pickle loader with the project's
already exported NumPy arrays. It is an author-recipe probe, not a byte-exact
reproduction of the authors' GDF preprocessing or 2019 software environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.labels = [[str(i), "", int(label)] for i, label in enumerate(self.y)]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict:
        return {"data": torch.from_numpy(self.x[index]), "label": int(self.y[index])}

    def combineDataset(self, other: "ArrayDataset") -> None:
        self.x = np.concatenate((self.x, other.x), axis=0)
        self.y = np.concatenate((self.y, other.y), axis=0)
        self.labels.extend(other.labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.subject <= 9:
        raise ValueError("subject must be 1..9")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir / "codes" / "centralRepo"))
    from baseModel import baseModel  # type: ignore
    from networks import eegNet  # type: ignore

    prefix = f"A{args.subject:02d}"
    x_train = np.load(args.data_dir / f"{prefix}T_data.npy")
    y_train = np.load(args.data_dir / f"{prefix}T_label.npy").reshape(-1) - 1
    x_test = np.load(args.data_dir / f"{prefix}E_data.npy")
    y_test = np.load(args.data_dir / f"{prefix}E_label.npy").reshape(-1) - 1
    assert x_train.shape == x_test.shape == (288, 22, 1000)
    assert sorted(np.unique(y_train).tolist()) == [0, 1, 2, 3]
    assert sorted(np.unique(y_test).tolist()) == [0, 1, 2, 3]

    # ho.py takes the first ceil(80%) of Session 1 for stage 1 training.
    split = math.ceil(len(x_train) * 0.8)
    train = ArrayDataset(x_train[:split], y_train[:split])
    val = ArrayDataset(x_train[split:], y_train[split:])
    test = ArrayDataset(x_test, y_test)

    # ho.py creates the network from its stored initialization with this seed.
    np.random.seed(20190821)
    torch.manual_seed(20190821)
    net = eegNet(nChan=22, nTime=1000, nClass=4, dropoutP=0.5)
    model = baseModel(net=net, batchSize=16, nGPU=args.gpu)
    max_epochs = 1 if args.smoke else 1500
    patience = 1 if args.smoke else 200
    start = time.monotonic()
    model.train(
        train,
        val,
        test,
        classes=[0, 1, 2, 3],
        sampler="RandomSampler",
        lr=1e-3,
        stopCondi={"c": {"Or": {
            "c1": {"MaxEpoch": {"maxEpochs": max_epochs, "varName": "epoch"}},
            "c2": {"NoDecrease": {"numEpochs": patience, "varName": "valInacc"}},
        }}},
        loadBestModel=True,
        bestVarToCheck="valInacc",
        continueAfterEarlystop=not args.smoke,
    )
    details = model.expDetails[0]["results"]
    summary = {
        "source_commit": "de1bbdd8a54cb1e466830e3d47070e0e56761a37",
        "subject": args.subject,
        "smoke": args.smoke,
        "dataset": "local MOABB-exported NumPy arrays, not author GDF parser",
        "torch": torch.__version__,
        "gpu": str(model.device),
        "elapsed_seconds": time.monotonic() - start,
        "stage1_train_count": split,
        "stage1_validation_count": len(x_train) - split,
        "test_accuracy": details["test"]["acc"],
        "train_accuracy": details["trainBest"]["acc"],
        "validation_accuracy": details["valBest"]["acc"],
    }
    out_file = args.out_dir / f"{prefix}_{'smoke' if args.smoke else 'full'}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
