"""Numerically compare local FFT-IIR FBCNet filter with pinned author lfilter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


BANDS = ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
         (24, 28), (28, 32), (32, 36), (36, 40))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    sys.path.insert(0, str(args.author / "codes" / "centralRepo"))
    from model.baselines import _CausalChebyshevFilterBank  # type: ignore
    from transforms import filterBank  # type: ignore

    source = np.load(args.data, mmap_mode="r")[:3].copy()
    bank = filterBank(BANDS, fs=250, filtType="filter")
    expected = np.stack([
        bank.bandpassFilter(source, band, 250, filtAllowance=2, axis=2, filtType="filter")
        for band in BANDS
    ], axis=1)
    local = _CausalChebyshevFilterBank(250, BANDS, 1000)
    with torch.no_grad():
        observed = local(torch.from_numpy(source[:, None]).float()).numpy()
    difference = observed.astype(np.float64) - expected
    rows = []
    for index, band in enumerate(BANDS):
        a = expected[:, index].ravel()
        b = observed[:, index].ravel()
        rows.append({"band": list(band), "max_abs_diff_uv": float(np.max(np.abs(b - a))),
                     "rmse_uv": float(np.sqrt(np.mean(np.square(b - a)))),
                     "correlation": float(np.corrcoef(a, b)[0, 1]),
                     "expected_sd_uv": float(np.std(a))})
    report = {"source": str(args.data), "n_trials": len(source),
              "overall_rmse_uv": float(np.sqrt(np.mean(np.square(difference)))),
              "bands": rows}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
