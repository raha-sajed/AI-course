"""Reproduce with: python HomeWork1/signal_analysis.py."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def segment_signal(signal: np.ndarray, length: int = 128, overlap: int = 64) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if length < 1 or overlap < 0 or overlap >= length:
        raise ValueError("Require length > 0 and 0 <= overlap < length.")
    if signal.size < length or not np.isfinite(signal).all():
        raise ValueError("Signal must be finite and contain at least one complete window.")
    step = length - overlap
    starts = np.arange(0, signal.size - length + 1, step)
    return np.lib.stride_tricks.sliding_window_view(signal, length)[::step].copy(), starts


def extract_features(windows: np.ndarray, starts: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "start_sample": starts,
        "mean": windows.mean(axis=1),
        "std": windows.std(axis=1, ddof=0),
        "rms": np.sqrt(np.mean(windows ** 2, axis=1)),
    })


def run(data: Path, output: Path, key: str = "X201_DE_time", length: int = 128, overlap: int = 64, sampling_rate: float | None = None) -> dict:
    if sampling_rate is not None and sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    payload = loadmat(data)
    if key not in payload:
        raise ValueError(f"Signal key {key!r} not found in {data.name}.")
    signal = np.asarray(payload[key], dtype=float).reshape(-1)
    if not np.isfinite(signal).all():
        raise ValueError("The source signal contains non-finite values.")
    centered = signal - signal.mean()
    windows, starts = segment_signal(centered, length, overlap)
    features = extract_features(windows, starts)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "segment_features.csv", index=False, float_format="%.10g")
    spectrum = np.abs(np.fft.rfft(centered)) / len(centered)
    frequency = np.fft.rfftfreq(len(centered), d=1 / sampling_rate if sampling_rate else 1)
    # Exclude DC when reporting the strongest periodic component.
    dominant = int(np.argmax(spectrum[1:]) + 1)
    summary = {
        "revision_owner": "raha-sajed", "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "signal_key": key, "n_samples": len(signal), "window_length": length,
        "overlap": overlap, "hop_length": length - overlap, "n_windows": len(windows),
        "discarded_tail_samples": int(len(signal) - (starts[-1] + length)),
        "preprocessing": "Subtract the mean of the entire signal; descriptive analysis, no predictive split",
        "std_ddof": 0, "sampling_rate_hz": sampling_rate,
        "sampling_rate_status": "Explicitly supplied; verify against the original source" if sampling_rate else "Unknown; no physical time or Hz is inferred",
        "dominant_frequency": float(frequency[dominant]),
        "frequency_unit": "Hz" if sampling_rate else "cycles/sample",
        "feature_summary": features[["mean", "std", "rms"]].describe().to_dict(),
        "all_features_finite": bool(np.isfinite(features.to_numpy()).all()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    shown = min(3000, len(signal))
    axes[0].plot(np.arange(shown), centered[:shown], color="#1e3a8a", linewidth=0.7)
    axes[0].set(title="Mean-centered signal: first 3,000 samples", xlabel="Sample index", ylabel="Amplitude (source units)")
    for idx in np.arange(0, len(windows), 13)[:10]:
        axes[1].plot(np.arange(length), windows[idx], alpha=0.7, linewidth=0.9, label=f"Segment {idx}")
    axes[1].set(title="Ten segments, selected every 13 windows", xlabel="Sample within window", ylabel="Amplitude (source units)")
    axes[1].legend(ncol=5, fontsize=8, frameon=False)
    axes[2].plot(features.start_sample, features.rms, color="#0d9488", linewidth=0.7)
    axes[2].set(title="RMS across all complete windows", xlabel="Window start sample", ylabel="RMS (source units)")
    fig.savefig(output / "signal_analysis.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.plot(frequency[1:], spectrum[1:], color="#2563eb", linewidth=0.6)
    ax.set(title="Magnitude spectrum after mean removal", xlabel=summary["frequency_unit"], ylabel="DFT magnitude / sample count")
    fig.savefig(output / "frequency_spectrum.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("n_samples", "n_windows", "discarded_tail_samples", "all_features_finite")}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "HomeWork1/201.mat")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--key", default="X201_DE_time")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--sampling-rate", type=float, help="Only supply a rate verified against the signal source")
    args = parser.parse_args()
    run(args.data, args.output, args.key, args.length, args.overlap, args.sampling_rate)


if __name__ == "__main__":
    main()

