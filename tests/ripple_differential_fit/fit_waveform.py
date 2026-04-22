"""Differential fit of a ripple waveform to H1/L1 time-series strain.

Pipeline
--------
1. Read pre-generated LIGO signal+background *time series* for H1 and L1.
2. Window and rFFT each detector's strain to the frequency domain.
3. Generate a frequency-domain waveform (h+, hx) with ripple on the rFFT grid.
4. Project to each detector via its antenna-pattern factors (F+, Fx).
5. Fit chirp mass via JAX gradient descent on the matched-filter
   log-likelihood; track network SNR at every step.
6. Plot SNR vs chirp mass across the fitting trajectory.

Expected HDF5 layout for `--data`:

    /H1/strain                     float64 [N]        time-domain strain (signal + background)
    /H1/psd                        float64 [N//2+1]   one-sided PSD on the rFFT grid
    /L1/strain, /L1/psd            same as H1
    /antenna/H1/fplus,  /fcross    scalar float       antenna-pattern factors
    /antenna/L1/fplus,  /fcross    scalar float
    attrs:
        sample_rate (Hz)           scalar             time-series sample rate
        f_ref       (Hz)           scalar             waveform reference frequency

All four time-series must share the same length N. Each PSD must be on
the rFFT grid of that time series, i.e. `numpy.fft.rfftfreq(N, 1/fs)`.

Run:

    python fit_waveform.py --data data.h5 --out fit_history.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from scipy.signal.windows import tukey

jax.config.update("jax_enable_x64", True)

DETECTORS = ("H1", "L1")


def _rfft_to_strain(x_td: np.ndarray, fs: float, alpha: float = 0.1) -> np.ndarray:
    """Windowed one-sided FFT in strain units (Hz^-1)."""
    window = tukey(x_td.size, alpha=alpha)
    return np.fft.rfft(x_td * window) / fs


def load_data(path: Path, tukey_alpha: float = 0.1) -> dict:
    with h5py.File(path, "r") as f:
        fs = float(f.attrs["sample_rate"])
        f_ref = float(f.attrs["f_ref"])

        lengths = {det: f[f"{det}/strain"].shape[0] for det in DETECTORS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Detector time series lengths differ: {lengths}")
        n = lengths["H1"]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        detectors = {}
        for det in DETECTORS:
            td = np.asarray(f[f"{det}/strain"][()], dtype=np.float64)
            psd = np.asarray(f[f"{det}/psd"][()], dtype=np.float64)
            if psd.shape[0] != freqs.shape[0]:
                raise ValueError(
                    f"{det}: PSD length {psd.shape[0]} does not match "
                    f"rFFT grid length {freqs.shape[0]}"
                )
            detectors[det] = {
                "strain": jnp.asarray(_rfft_to_strain(td, fs, tukey_alpha),
                                      dtype=jnp.complex128),
                "psd": jnp.asarray(psd, dtype=jnp.float64),
                "fplus": float(f[f"antenna/{det}/fplus"][()]),
                "fcross": float(f[f"antenna/{det}/fcross"][()]),
            }

        truth = None
        if "truth" in f:
            truth = {k: np.asarray(v[()]) for k, v in f["truth"].items()}

    return {
        "freqs": jnp.asarray(freqs, dtype=jnp.float64),
        "sample_rate": fs,
        "duration": n / fs,
        "f_ref": f_ref,
        "truth": truth,
        **detectors,
    }


def _float(x) -> float:
    """Coerce a numpy scalar or 0/1-d array to a python float."""
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def init_from_truth(args: argparse.Namespace, truth: dict) -> None:
    """Override non-mass CLI init values with truth params from HDF5.

    Mutates `args` in place. Chirp mass is intentionally not overridden —
    the fit still searches it starting from `--mc-init`.
    """
    if truth is None:
        raise SystemExit("--init-from-truth requested but /truth not in HDF5")

    def pick(*keys):
        for k in keys:
            if k in truth:
                return _float(truth[k])
        return None

    mass_ratio = pick("mass_ratio")
    if mass_ratio is not None:
        q = mass_ratio
        args.eta = q / (1.0 + q) ** 2  # symmetric mass ratio from m2/m1

    for attr, keys in [
        ("chi1", ("chi1", "s1z")),
        ("chi2", ("chi2", "s2z")),
        ("dist", ("distance",)),
        ("tc", ("tc",)),
        ("phic", ("phic",)),
        ("inclination", ("inclination",)),
    ]:
        val = pick(*keys)
        if val is not None:
            setattr(args, attr, val)

    print("[init-from-truth] starting points overridden from /truth:")
    for attr in ("eta", "chi1", "chi2", "dist", "tc", "phic", "inclination"):
        print(f"    {attr:12s} = {getattr(args, attr):+.6g}")


def ripple_polarizations(theta, freqs, f_ref):
    """h+, hx in frequency domain from ripple's IMRPhenomD."""
    hp, hc = gen_IMRPhenomD_polar(freqs, theta, f_ref)
    return hp, hc


def project(hp, hc, fplus: float, fcross: float):
    return fplus * hp + fcross * hc


def inner_product(a, b, psd, df):
    return 4.0 * df * jnp.real(jnp.sum(jnp.conj(a) * b / psd))


def network_loglike_and_snr(theta, data):
    freqs = data["freqs"]
    df = freqs[1] - freqs[0]
    hp, hc = ripple_polarizations(theta, freqs, data["f_ref"])

    ll = jnp.array(0.0)
    snr_sq = jnp.array(0.0)
    for det in DETECTORS:
        d = data[det]
        h = project(hp, hc, d["fplus"], d["fcross"])
        dh = inner_product(d["strain"], h, d["psd"], df)
        hh = inner_product(h, h, d["psd"], df)
        ll = ll + dh - 0.5 * hh
        snr_sq = snr_sq + dh * dh / hh
    return ll, jnp.sqrt(snr_sq)


def make_fit_fns(data, fixed_rest):
    """Return jitted loss, grad and SNR functions that take only chirp mass."""
    fixed_rest = jnp.asarray(fixed_rest, dtype=jnp.float64)

    def theta_of(mc):
        return jnp.concatenate([jnp.array([mc]), fixed_rest])

    def loss(mc):
        ll, _ = network_loglike_and_snr(theta_of(mc), data)
        return -ll

    def snr(mc):
        _, s = network_loglike_and_snr(theta_of(mc), data)
        return s

    return jax.jit(jax.value_and_grad(loss)), jax.jit(snr)


def fit(data, init_theta, n_steps: int, lr: float):
    init_theta = np.asarray(init_theta, dtype=np.float64)
    mc = jnp.asarray(init_theta[0])
    loss_grad_fn, snr_fn = make_fit_fns(data, init_theta[1:])

    opt = optax.adam(lr)
    opt_state = opt.init(mc)

    history = {"mc": [], "snr": [], "loss": []}
    for step in range(n_steps):
        loss, grad = loss_grad_fn(mc)
        snr = snr_fn(mc)
        history["mc"].append(float(mc))
        history["snr"].append(float(snr))
        history["loss"].append(float(loss))
        updates, opt_state = opt.update(grad, opt_state)
        mc = optax.apply_updates(mc, updates)

    return float(mc), history


def plot_history(history, out_path: Path) -> None:
    mcs = np.asarray(history["mc"])
    snrs = np.asarray(history["snr"])
    iters = np.arange(len(mcs))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mcs, snrs, "-", color="gray", alpha=0.4, zorder=1)
    sc = ax.scatter(mcs, snrs, c=iters, cmap="viridis", s=18, zorder=2)
    ax.set_xlabel(r"Chirp mass $\mathcal{M}$ [$M_\odot$]")
    ax.set_ylabel("Network SNR")
    ax.set_title("Differential fit trajectory")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path,
                   help="HDF5 file with H1/L1 time-series strain, PSD, and antenna factors")
    p.add_argument("--out", default=Path("fit_history.png"), type=Path,
                   help="Output path for the SNR-vs-mass plot")
    p.add_argument("--tukey-alpha", type=float, default=0.1,
                   help="Tukey window alpha applied before rFFT")
    p.add_argument("--mc-init", type=float, default=1.4,
                   help="Initial chirp mass guess [Msun]")
    p.add_argument("--eta", type=float, default=0.24,
                   help="Symmetric mass ratio (fixed during fit)")
    p.add_argument("--chi1", type=float, default=0.0)
    p.add_argument("--chi2", type=float, default=0.0)
    p.add_argument("--dist", type=float, default=100.0, help="Distance [Mpc]")
    p.add_argument("--tc", type=float, default=0.0)
    p.add_argument("--phic", type=float, default=0.0)
    p.add_argument("--inclination", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--init-from-truth", action="store_true",
                   help="Initialise all non-mass parameters from the /truth "
                        "group in the HDF5 (simulation-only info; use only "
                        "for sanity checks). Chirp mass still starts at --mc-init.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data, tukey_alpha=args.tukey_alpha)

    if args.init_from_truth:
        init_from_truth(args, data["truth"])

    init_theta = [
        args.mc_init, args.eta, args.chi1, args.chi2,
        args.dist, args.tc, args.phic, args.inclination,
    ]
    mc_fit, history = fit(data, init_theta, n_steps=args.steps, lr=args.lr)

    print(f"Recovered chirp mass: {mc_fit:.4f} Msun")
    print(f"Final network SNR:    {history['snr'][-1]:.3f}")
    plot_history(history, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
