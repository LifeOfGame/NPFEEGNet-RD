"""Patch only the isolated local FBCNet filter to match author coefficients."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    path = args.source
    text = path.read_text(encoding="utf-8")
    old = '''            order, critical = signal.cheb2ord(
                passband,
                stopband,
                passband_ripple_db,
                stopband_attenuation_db,
            )
            numerator, denominator = signal.cheby2(
                order,
                stopband_attenuation_db,
                critical,
                btype="bandpass",
            )'''
    new = '''            order, _critical = signal.cheb2ord(
                passband,
                stopband,
                passband_ripple_db,
                stopband_attenuation_db,
            )
            # The pinned FBCNet author's filterBank uses fStop, not ws.
            numerator, denominator = signal.cheby2(
                order,
                stopband_attenuation_db,
                stopband,
                btype="bandpass",
            )'''
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exact one-site patch, found {text.count(old)}")
    before = digest(path)
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched={path} before_sha256={before} after_sha256={digest(path)}")


if __name__ == "__main__":
    main()
