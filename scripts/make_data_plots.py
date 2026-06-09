"""Produce the data-overview plots referenced from the README.

Reads ``example_data/data_BBH_highSNR.h5`` and writes:

* ``plots/strain_timeseries.png`` — raw L1 strain (signal + O3a noise)
* ``plots/psd.png`` — one-sided amplitude spectral density
* ``plots/whitened_qtransform.png`` — whitened, band-passed L1 strain plus
  its Q-transform around the merger
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

DATA = Path("example_data/data_BBH_highSNR.h5")
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

COLOR = {"L1": "#4ba6ff"}
DETECTORS = ("L1",)


def load() -> dict:
    arrays: dict = {}
    with h5py.File(DATA, "r") as f:
        f.visititems(
            lambda name, obj: arrays.update({name: obj[...]})
            if isinstance(obj, h5py.Dataset)
            else None
        )
        attrs = dict(f.attrs)
    arrays["__attrs__"] = attrs
    return arrays


def strain_plot(arrays: dict, fs: float) -> None:
    fig, axes = plt.subplots(len(DETECTORS), 1, figsize=(10, 3), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    n = arrays[f"{DETECTORS[0]}/strain"].shape[0]
    t = np.arange(n) / fs
    for ax, det in zip(axes, DETECTORS):
        ax.plot(t, arrays[f"{det}/strain"], color=COLOR[det], lw=0.6)
        ax.set_ylabel(f"{det} strain")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title("Raw strain: injected BBH signal + O3a detector noise")
    fig.tight_layout()
    fig.savefig(OUT / "strain_timeseries.png", dpi=140)
    plt.close(fig)


def psd_plot(arrays: dict, fs: float) -> None:
    n = arrays[f"{DETECTORS[0]}/strain"].shape[0]
    df = fs / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for det in DETECTORS:
        ax.loglog(freqs, np.sqrt(arrays[f"{det}/psd"]),
                  color=COLOR[det], label=f"{det} ASD")
    ax.set_xlim(10, fs / 2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"ASD $\sqrt{S_n(f)}\;[\mathrm{Hz}^{-1/2}]$")
    ax.set_title(f"One-sided amplitude spectral densities (Δf = {df:.3f} Hz)")
    ax.axvline(20.0, ls="--", color="gray", alpha=0.6, label="f_min = 20 Hz")
    ax.legend()
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "psd.png", dpi=140)
    plt.close(fig)


def whitened_qtransform(arrays: dict, fs: float, det: str = "L1") -> None:
    n = arrays[f"{det}/strain"].shape[0]
    ts = TimeSeries(arrays[f"{det}/strain"], t0=0, sample_rate=fs)
    asd = FrequencySeries(arrays[f"{det}/psd"], f0=0, df=fs / n) ** 0.5
    white = ts.whiten(asd=asd).bandpass(30, 400)

    tc = float(arrays["truth/tc"])
    out_lo, out_hi = max(0.0, tc - 1.0), min(n / fs - 0.01, tc + 0.5)
    qtrans = white.q_transform(
        frange=(20, 500), outseg=(out_lo, out_hi), whiten=False
    )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6))
    ax0.plot(white.times.value, white.value, color=COLOR[det], lw=0.7)
    ax0.set_xlim(max(0.0, tc - 1.5), min(n / fs, tc + 0.5))
    ax0.axvline(tc, ls="--", color="black", alpha=0.6, label=f"truth tc = {tc:.2f} s")
    ax0.set_ylabel(f"whitened {det} strain")
    ax0.set_xlabel("time [s]")
    ax0.legend()
    ax0.grid(alpha=0.3)

    pcm = ax1.pcolormesh(
        qtrans.times.value, qtrans.frequencies.value, qtrans.value.T,
        vmin=0, vmax=25, cmap="viridis", shading="auto",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("frequency [Hz]")
    ax1.set_xlabel("time [s]")
    ax1.axvline(tc, ls="--", color="white", alpha=0.7)
    fig.colorbar(pcm, ax=ax1, label="normalised energy")
    ax1.set_title(f"{det} Q-transform around merger")
    fig.tight_layout()
    fig.savefig(OUT / "whitened_qtransform.png", dpi=140)
    plt.close(fig)


def main() -> None:
    arrays = load()
    attrs = arrays["__attrs__"]
    fs = float(attrs.get("sample_rate", 4096.0))
    strain_plot(arrays, fs)
    psd_plot(arrays, fs)
    whitened_qtransform(arrays, fs, det="L1")
    print("Wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
