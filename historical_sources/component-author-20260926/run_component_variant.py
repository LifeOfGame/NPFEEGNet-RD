"""Use the pinned OpenBMI engine with a predeclared independent component model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ENGINE_SHA = "dd010b3c64a1ec25e10b1fe539e439a0e1f3305942327e4baea49ef7e6485e15"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=["spectral_only", "no_spectral_prior"], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repro.frozen_engine_loader import load_openbmi_engine
    from component_variants import SpectralOnly, NoSpectralPrior

    training = load_openbmi_engine("openbmi_component_" + args.variant)
    digest = hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest()
    if digest != ENGINE_SHA:
        raise RuntimeError("Pinned OpenBMI engine hash changed")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["network"] != "NPFEEGNet":
        raise ValueError("Expected NPFEEGNet registry key")
    if config.get('fixed_epoch_training'):
        lock = json.loads(Path(config['protocol_lock_file']).read_text())
        if lock.get('status') != 'ready_for_session2':
            raise RuntimeError('Session-2 protocol not frozen')
        for path, checksum in lock['sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f'Frozen artifact changed: {path}')
    training.NPFEEGNet = {
        "spectral_only": SpectralOnly,
        "no_spectral_prior": NoSpectralPrior,
    }[args.variant]
    print(f"variant={args.variant} engine_sha256={digest}", flush=True)
    training.main(config)


if __name__ == "__main__":
    main()
