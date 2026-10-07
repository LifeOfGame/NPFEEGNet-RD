"""Portable local-MAT OpenBMI export with preserved historical extraction.

The extraction functions are retained from the project's authored OpenBMI
preprocessor. No download, MOABB import, legacy engine, or raw data is bundled.
Onsets, channel order, resampling and integer labels retain historical behavior.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

OPENBMI_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
)

PHASE_KEYS = ("EEG_MI_train", "EEG_MI_test")

TARGET_SAMPLING_RATE = 250.0

EPOCH_SECONDS = (0.0, 4.0)

EXPECTED_TRIALS_PER_PHASE = 100

EXPECTED_TRIALS_PER_CLASS_PER_PHASE = 50

def _matlab_string(value: Any) -> str:
    squeezed = np.squeeze(value)
    if squeezed.ndim == 0:
        return str(squeezed.item()).strip()
    if squeezed.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in squeezed.flat).strip()
    return str(squeezed.tolist()).strip()

def channel_names(eeg_struct: np.void) -> list[str]:
    names = [_matlab_string(item) for item in np.ravel(eeg_struct["chan"])]
    if len(names) != len(set(names)):
        raise RuntimeError("OpenBMI MAT contains duplicate EEG channel names")
    return names

def extract_phase(
    eeg_struct: np.void,
    selected_channels: tuple[str, ...] = OPENBMI_CHANNELS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract one 100-trial phase without filtering or label-dependent steps."""
    signal = np.asarray(eeg_struct["x"])
    onsets = np.asarray(eeg_struct["t"]).squeeze().astype(np.int64)
    labels = np.asarray(eeg_struct["y_dec"]).squeeze().astype(np.int64)
    sampling_rate = float(np.asarray(eeg_struct["fs"]).squeeze())
    names = channel_names(eeg_struct)
    missing = sorted(set(selected_channels) - set(names))
    if missing:
        raise RuntimeError(f"OpenBMI file is missing frozen channels: {missing}")
    indices = np.asarray([names.index(name) for name in selected_channels])

    if signal.ndim != 2 or signal.shape[1] != len(names):
        raise RuntimeError(
            f"Expected OpenBMI continuous data (samples, channels), got {signal.shape}"
        )
    if len(onsets) != EXPECTED_TRIALS_PER_PHASE or len(labels) != len(onsets):
        raise RuntimeError(
            f"Expected {EXPECTED_TRIALS_PER_PHASE} trials, got "
            f"onsets={len(onsets)}, labels={len(labels)}"
        )
    if set(np.unique(labels).tolist()) != {1, 2}:
        raise RuntimeError(f"Expected OpenBMI labels 1/2, got {np.unique(labels)}")
    class_counts = np.bincount(labels, minlength=3)[1:3]
    if not np.all(class_counts == EXPECTED_TRIALS_PER_CLASS_PER_PHASE):
        raise RuntimeError(
            "OpenBMI phase is not 50/50 balanced: "
            f"class counts={class_counts.tolist()}"
        )

    source_samples = int(round((EPOCH_SECONDS[1] - EPOCH_SECONDS[0]) * sampling_rate))
    windows = []
    for onset in onsets:
        start = int(onset + round(EPOCH_SECONDS[0] * sampling_rate))
        stop = start + source_samples
        if start < 0 or stop > signal.shape[0]:
            raise RuntimeError(
                f"OpenBMI epoch [{start}, {stop}) exceeds signal length {signal.shape[0]}"
            )
        # The MAT signal is already expressed in microvolts.  Cast before
        # stacking so a large continuous float64 recording is not duplicated.
        windows.append(
            np.asarray(signal[start:stop, indices].T, dtype=np.float32)
        )
    data = np.stack(windows, axis=0)

    ratio = Fraction(TARGET_SAMPLING_RATE / sampling_rate).limit_denominator(10000)
    data = resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
    data = np.asarray(data, dtype=np.float32)
    expected_samples = int(
        round((EPOCH_SECONDS[1] - EPOCH_SECONDS[0]) * TARGET_SAMPLING_RATE)
    )
    if data.shape != (
        EXPECTED_TRIALS_PER_PHASE,
        len(selected_channels),
        expected_samples,
    ):
        raise RuntimeError(f"Unexpected OpenBMI epoch shape after resampling: {data.shape}")
    if not np.isfinite(data).all():
        raise RuntimeError("OpenBMI epochs contain NaN or infinity")

    metadata = {
        "shape": list(data.shape),
        "source_sampling_rate": sampling_rate,
        "target_sampling_rate": TARGET_SAMPLING_RATE,
        "class_counts": class_counts.tolist(),
        "label_mapping": {"1": "right_hand", "2": "left_hand"},
    }
    return data, labels, metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat',type=Path,required=True)
    parser.add_argument('--session',type=int,choices=(1,2),required=True)
    parser.add_argument('--subject',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path,nargs='+',help='Completed selection records, required before Session 2 is opened')
    args=parser.parse_args()
    if not 1<=args.subject<=54:raise ValueError('Subject must be 1-54')
    if args.mat.name!=f'sess{args.session:02d}_subj{args.subject:02d}_EEG_MI.mat':raise ValueError('MAT filename does not match subject/session')
    locks=[]
    if args.session==2:
        if not args.selection:raise ValueError('Session 2 requires completed source selection')
        for path in args.selection:
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('protocol')!='npfeegnet-maintained-v2' or record.get('stage')!='selection' or record.get('status')!='complete' or record.get('test_accessed'):
                raise ValueError('Invalid source-only selection record')
            if not any(Path(p).name==f'O{args.subject:02d}T_data.npy' for p in record['training_files']):raise ValueError('Selection belongs to another subject/input schema')
            for name,digest in record['source_sha256'].items():
                p=(ROOT/name).resolve()
                if not p.is_relative_to(ROOT) or sha(p)!=digest:raise ValueError('Training source changed after selection')
            for name,digest in record['training_files'].items():
                if sha(name)!=digest:raise ValueError('Source training input changed after selection')
            locks.append({'selection_sha256':sha(path),'selected_epoch':record['selected_epoch']})
    elif args.selection:raise ValueError('Session 1 does not consume evaluation locks')
    suffix='T' if args.session==1 else 'E';stem=f'O{args.subject:02d}{suffix}'
    outputs=[args.output/f'{stem}_{kind}.npy' for kind in ('data','label','group')]
    manifest_path=args.output/f'O{args.subject:02d}_session{args.session}_manifest.json'
    if any(p.exists() for p in outputs+[manifest_path]):raise FileExistsError('Refusing to overwrite prepared data')
    payload=loadmat(args.mat)
    arrays=[];labels=[];groups=[];phases=[]
    for phase,key in enumerate(PHASE_KEYS):
        X,y,info=extract_phase(payload[key][0,0])
        arrays.append(X);labels.append(y);groups.append(np.full(len(y),phase,dtype=np.int64));phases.append(dict(info,phase=key))
    args.output.mkdir(parents=True,exist_ok=True)
    for path,array in zip(outputs,[np.concatenate(arrays),np.concatenate(labels)[:,None],np.concatenate(groups)]):np.save(path,array)
    manifest={'scope':'Maintained export preserving historical OpenBMI extraction; not a new prospective study','subject':args.subject,'session':args.session,'source_sha256':sha(args.mat),'preprocessor_sha256':sha(Path(__file__)),'channels':list(OPENBMI_CHANNELS),'epoch_seconds':list(EPOCH_SECONDS),'scale':'microvolts','sampling_rate':TARGET_SAMPLING_RATE,'group_definition':'offline training phase=0; online feedback phase=1','phases':phases,'source_selection_locks':locks,'files':{p.name:sha(p) for p in outputs}}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'subject':args.subject,'session':args.session,'files':manifest['files']}))

if __name__=='__main__':main()
