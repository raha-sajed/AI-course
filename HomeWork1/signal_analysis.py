"""Reproduce the CWRU exercise: python HomeWork1/signal_analysis.py."""
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
# The assignment specifies 48 kHz; CWRU lists 201.mat under 48k drive-end faults.
DEFAULT_SAMPLING_RATE = 48_000.0
SOURCE = "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data"


def segment_signal(signal: np.ndarray, length: int = 128, overlap: int = 64) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if length < 1 or overlap < 0 or overlap >= length:
        raise ValueError("Require length > 0 and 0 <= overlap < length.")
    if signal.size < length or not np.isfinite(signal).all():
        raise ValueError("Signal must be finite and contain at least one complete window.")
    starts = np.arange(0, signal.size - length + 1, length - overlap)
    windows = np.lib.stride_tricks.sliding_window_view(signal, length)[::length - overlap].copy()
    return windows, starts


def signal_statistics(signal: np.ndarray) -> tuple[float, float, float]:
    """Mean, population standard deviation and RMS for one signal/window."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Expected a nonempty finite signal.")
    return float(signal.mean()), float(signal.std(ddof=0)), float(np.sqrt(np.mean(signal ** 2)))


def extract_features(windows: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([signal_statistics(window) for window in windows], columns=["mean", "std", "rms"])


def compute_spectrum(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> tuple[np.ndarray, np.ndarray, float | None]:
    """One-sided amplitude spectrum; subtract DC and preserve a sine's amplitude."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError("Expected at least two finite samples.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be finite and positive.")
    centered = signal - signal.mean()
    frequency = np.fft.rfftfreq(len(signal), d=1 / sampling_rate)
    amplitude = np.abs(np.fft.rfft(centered)) / len(signal)
    if len(signal) % 2 == 0:
        amplitude[1:-1] *= 2  # Do not double DC or the Nyquist bin.
    else:
        amplitude[1:] *= 2
    peak = int(np.argmax(amplitude[1:]) + 1)
    threshold = np.finfo(float).eps * max(1., float(np.max(np.abs(signal))))
    dominant = float(frequency[peak]) if amplitude[peak] > threshold else None
    return frequency, amplitude, dominant


def time_window(signal: np.ndarray, sampling_rate: float, start: float = 2., end: float = 2.01) -> tuple[np.ndarray, np.ndarray]:
    """Select original samples in [start, end); never renumber or downsample them."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0 or not 0 <= start < end:
        raise ValueError("Expected a positive sampling rate and 0 <= start < end.")
    if end > len(signal) / sampling_rate:
        raise ValueError("The requested time window exceeds the recording.")
    times = np.arange(len(signal)) / sampling_rate
    mask = (times >= start) & (times < end)
    if not mask.any():
        raise ValueError("No samples fall in the requested time window.")
    return times[mask], signal[mask]


def run(data: Path, output: Path, key: str = "X201_DE_time", length: int = 128, overlap: int = 64, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> dict:
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be finite and positive.")
    payload = loadmat(data)
    if key not in payload:
        raise ValueError(f"Signal key {key!r} not found in {data.name}.")
    signal = np.asarray(payload[key], dtype=float).reshape(-1)
    windows, starts = segment_signal(signal, length, overlap)
    segments = pd.DataFrame(windows)
    features = extract_features(windows)
    frequency, spectrum, dominant = compute_spectrum(signal, sampling_rate)
    zoom_time, zoom_signal = time_window(signal, sampling_rate)
    selected = np.arange(13, 131, 13)  # Ten positive multiples of 13, zero-based rows.
    if selected[-1] >= len(segments):
        raise ValueError("At least 131 segments are required for the requested overlay.")
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "segment_features.csv", index=False, float_format="%.10g")
    summary = {
        "revision_owner": "raha-sajed", "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "data_source": SOURCE, "mat_reader_type": type(payload).__name__,
        "mat_fields": list(payload),
        "mat_variables": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in payload.items() if not name.startswith("__")},
        "signal_key": key, "n_samples": len(signal), "sampling_rate_hz": sampling_rate,
        "sampling_rate_status": "48 kHz as specified in assignment and CWRU 48k table" if sampling_rate == DEFAULT_SAMPLING_RATE else "Explicit CLI override",
        "duration_seconds": len(signal) / sampling_rate,
        "last_sample_seconds": (len(signal) - 1) / sampling_rate,
        "zoom_interval_seconds": [2., 2.01], "zoom_samples": len(zoom_signal),
        "window_length": length, "overlap": overlap, "hop_length": length - overlap,
        "n_windows": len(windows), "segments_shape": list(segments.shape),
        "overlay_rows_zero_based": selected.tolist(),
        "discarded_tail_samples": int(len(signal) - (starts[-1] + length)),
        "feature_input": "Original uncentered signal windows",
        "fft_preprocessing": "Subtract global mean for spectrum only; exclude DC when finding dominant frequency",
        "fft_amplitude": "Single-sided amplitude / N; double positive bins except Nyquist",
        "std_ddof": 0, "dominant_frequency_hz": dominant, "fft_bin_width_hz": sampling_rate / len(signal),
        "dominant_is_lowest_positive_bin": bool(dominant is not None and np.isclose(dominant, frequency[1])),
        "csv_columns": list(features.columns), "csv_index_written": False,
        "feature_summary": features.describe().to_dict(),
        "all_features_finite": bool(np.isfinite(features.to_numpy()).all()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 1, figsize=(11, 12.5), layout="constrained")
    times = np.arange(len(signal)) / sampling_rate
    axes[0].plot(times, signal, color="#1e3a8a", linewidth=.35)
    axes[0].set(title="Full drive-end recording", xlabel="Time (s)", ylabel="Amplitude (recorded units)")
    axes[1].plot(zoom_time, zoom_signal, color="#2563eb", linewidth=.9)
    axes[1].set(title="Original samples from 2.00 to 2.01 seconds", xlabel="Time (s)", ylabel="Amplitude (recorded units)")
    local_time = np.arange(length) / sampling_rate
    colors = plt.get_cmap("tab10")
    for i, row in enumerate(selected):
        axes[2].plot(local_time, segments.iloc[row].to_numpy(), color=colors(i), linewidth=.9, label=f"Row {row}")
    axes[2].set(title="Ten DataFrame rows at multiples of 13", xlabel="Time within segment (s)", ylabel="Amplitude (recorded units)")
    axes[2].legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.24))
    axes[3].plot(starts / sampling_rate, features.rms, color="#0d9488", linewidth=.7)
    axes[3].set(title="RMS for all complete windows", xlabel="Window start time (s)", ylabel="RMS (recorded units)")
    fig.savefig(output / "signal_analysis.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, limit, title in zip(axes, (sampling_rate / 2, 5.), ("Full frequency range", "Low-frequency detail (0–5 Hz)")):
        mask = frequency <= limit
        ax.plot(frequency[mask], spectrum[mask], color="#2563eb", linewidth=.8)
        if dominant is not None and dominant <= limit:
            ax.axvline(dominant, color="#b45309", linestyle="--", linewidth=.8, label=f"Peak bin: {dominant:.3f} Hz")
            ax.legend(frameon=False, fontsize=9)
        ax.set(title=title, xlabel="Frequency (Hz)", ylabel="Amplitude (recorded units)")
    fig.suptitle("Single-sided amplitude spectrum after mean removal")
    fig.savefig(output / "frequency_spectrum.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("n_samples", "sampling_rate_hz", "zoom_samples", "n_windows", "dominant_frequency_hz", "all_features_finite")}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "201.mat")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--key", default="X201_DE_time")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--sampling-rate", type=float, default=DEFAULT_SAMPLING_RATE)
    args = parser.parse_args()
    run(args.data, args.output, args.key, args.length, args.overlap, args.sampling_rate)


if __name__ == "__main__":
    main()
