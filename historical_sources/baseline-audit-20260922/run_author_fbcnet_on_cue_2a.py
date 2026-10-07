"""Author FBCNet architecture/trainer probe on audited cue-aligned MAT arrays.

The author's GDF conversion is unavailable on this server. Its 2 s offset is
applied in the separately verified MAT export. This adapter reproduces the
author's nine causal Chebyshev-II sub-bands on each four-second trial, then
uses the pinned FBCNet model and two-stage trainer without changing them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset


BANDS = ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
         (24, 28), (28, 32), (32, 36), (36, 40))


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


def filter_trials(x: np.ndarray, bank) -> np.ndarray:
    output = np.empty((*x.shape, len(BANDS)), dtype=np.float32)
    for index, band in enumerate(BANDS):
        output[..., index] = bank.bandpassFilter(
            x, band, 250, filtAllowance=2, axis=2, filtType="filter"
        ).astype(np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.subject not in range(1, 10):
        raise ValueError("subject must be 1..9")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir / "codes" / "centralRepo"))
    from baseModel import baseModel  # type: ignore
    from networks import FBCNet  # type: ignore
    from transforms import filterBank  # type: ignore

    prefix = f"A{args.subject:02d}"
    x_train = np.load(args.data_dir / f"{prefix}T_data.npy")
    y_train = np.load(args.data_dir / f"{prefix}T_label.npy").reshape(-1) - 1
    x_test = np.load(args.data_dir / f"{prefix}E_data.npy")
    y_test = np.load(args.data_dir / f"{prefix}E_label.npy").reshape(-1) - 1
    assert x_train.shape == x_test.shape == (288, 22, 1000)
    assert sorted(np.unique(y_train).tolist()) == [0, 1, 2, 3]
    assert sorted(np.unique(y_test).tolist()) == [0, 1, 2, 3]
    bank = filterBank(BANDS, fs=250, filtType="filter")
    started = time.monotonic()
    train_filtered = filter_trials(x_train, bank)
    test_filtered = filter_trials(x_test, bank)
    filter_seconds = time.monotonic() - started
    assert train_filtered.shape == test_filtered.shape == (288, 22, 1000, 9)
    assert np.isfinite(train_filtered).all() and np.isfinite(test_filtered).all()

    split = math.ceil(len(x_train) * 0.8)
    train = ArrayDataset(train_filtered[:split], y_train[:split])
    val = ArrayDataset(train_filtered[split:], y_train[split:])
    test = ArrayDataset(test_filtered, y_test)
    np.random.seed(20190821)
    torch.manual_seed(20190821)
    net = FBCNet(nChan=22, nTime=1000, nClass=4, nBands=9, m=32,
                 temporalLayer="LogVarLayer", doWeightNorm=True)
    model = baseModel(net=net, batchSize=16, nGPU=args.gpu)
    max_epochs = 1 if args.smoke else 1500
    patience = 1 if args.smoke else 200
    start = time.monotonic()
    model.train(
        train, val, test,
        classes=[0, 1, 2, 3], sampler="RandomSampler", lr=1e-3,
        stopCondi={"c": {"Or": {
            "c1": {"MaxEpoch": {"maxEpochs": max_epochs, "varName": "epoch"}},
            "c2": {"NoDecrease": {"numEpochs": patience, "varName": "valInacc"}},
        }}},
        loadBestModel=True, bestVarToCheck="valInacc",
        continueAfterEarlystop=not args.smoke,
    )
    details = model.expDetails[0]["results"]
    summary = {
        "source_commit": "de1bbdd8a54cb1e466830e3d47070e0e56761a37",
        "model": "author FBCNet",
        "subject": args.subject,
        "smoke": args.smoke,
        "dataset": "cue-aligned MOABB MAT export; not author's GDF parser",
        "torch": torch.__version__,
        "gpu": str(model.device),
        "filter_seconds": filter_seconds,
        "train_seconds": time.monotonic() - start,
        "stage1_train_count": split,
        "stage1_validation_count": len(x_train) - split,
        "test_accuracy": details["test"]["acc"],
        "train_accuracy": details["trainBest"]["acc"],
        "validation_accuracy": details["valBest"]["acc"],
    }
    output = args.out_dir / f"{prefix}_{'smoke' if args.smoke else 'full'}_fbcnet.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
