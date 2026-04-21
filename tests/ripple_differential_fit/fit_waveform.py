"""Differential fit of a ripple waveform to H1/L1 strain.

Pipeline
--------
1. Read pre-generated LIGO signal+background for H1 and L1 from HDF5.
2. Generate a frequency-domain waveform (h+, hx) with ripple.
3. Project to each detector via its antenna-pattern factors (F+, Fx).
4. Fit chirp mass via JAX gradient descent on the matched-filter
   log-likelihood; track network SNR at every step.
5. Plot SNR vs chirp mass across the fitting trajectory.

Expected HDF5 layout for `--data`:

    /freqs                         float64 [Nf]      one-sided frequency grid (Hz)
    /H1/strain                     complex128 [Nf]   signal + background, freq domain
    /H1/psd                        float64 [Nf]      one-sided PSD on /freqs
    /L1/strain, /L1/psd            same as H1
    /antenna/H1/fplus,  /fcross    scalar float      antenna pattern factors
    /antenna/L1/fplus,  /fcross    scalar float
    attrs: f_ref (Hz)

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

jax.config.update("jax_enable_x64", True)

DETECTORS = ("H1", "L1")


def load_data(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        freqs = jnp.asarray(f["freqs"][()], dtype=jnp.float64)
        detectors = {}
        for det in DETECTORS:
            detectors[det] = {
                "strain": jnp.asarray(f[f"{det}/strain"][()], dtype=jnp.complex128),
                "psd": jnp.asarray(f[f"{det}/psd"][()], dtype=jnp.float64),
                "fplus": float(f[f"antenna/{det}/fplus"][()]),
                "fcross": float(f[f"antenna/{det}/fcross"][()]),
            }
        f_ref = float(f.attrs["f_ref"])
    return {"freqs": freqs, "f_ref": f_ref, **detectors}


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
                   help="HDF5 file with H1/L1 strain, PSD, and antenna factors")
    p.add_argument("--out", default=Path("fit_history.png"), type=Path,
                   help="Output path for the SNR-vs-mass plot")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)

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
