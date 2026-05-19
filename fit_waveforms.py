"""Network chirp-mass fit via differentiable matched filtering.

Mirrors the final section of ``FirstTests.ipynb``: builds per-detector
matched-filter inputs from a single HDF5 event (the format produced by
``generate_fit_input.py``) and gradient-descends on the time-maximised
network SNR squared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc


def load_event(path: str) -> dict:
    arrays: dict = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda name, obj: arrays.update({name: obj[...]})
            if isinstance(obj, h5py.Dataset)
            else None
        )
    return arrays


def tukey_window(n: int, alpha: float = 0.01) -> jnp.ndarray:
    n_taper = int(round(alpha * n / 2))
    if n_taper == 0:
        return jnp.ones(n)
    ramp = 0.5 * (1 - jnp.cos(jnp.pi * (jnp.arange(n_taper) + 0.5) / n_taper))
    return jnp.concatenate([ramp, jnp.ones(n - 2 * n_taper), ramp[::-1]])


def build_detectors(
    arrays: dict,
    detectors: tuple[str, ...],
    fs: float,
    n: int,
    band: jnp.ndarray,
    tukey_alpha: float = 0.01,
) -> dict:
    window_td = tukey_window(n, tukey_alpha)
    out = {}
    for det in detectors:
        strain = jnp.asarray(arrays[f"{det}/strain"], dtype=jnp.float64)
        psd = jnp.asarray(arrays[f"{det}/psd"], dtype=jnp.float64)
        out[det] = {
            "d_full": jnp.fft.rfft(strain * window_td) / fs,
            "psd_safe": jnp.where(band, psd, jnp.inf),
            "fplus": float(arrays[f"antenna/{det}/fplus"]),
            "fcross": float(arrays[f"antenna/{det}/fcross"]),
        }
    return out


def theta_from_truth(arrays: dict) -> jnp.ndarray:
    """Ripple BBH order: [Mc, eta, chi1, chi2, D, tc, phic, iota]."""
    q = float(arrays["truth/mass_ratio"])
    return jnp.array(
        [
            float(arrays["truth/chirp_mass"]),
            q / (1 + q) ** 2,
            float(arrays["truth/chi1"]),
            float(arrays["truth/chi2"]),
            float(arrays["truth/distance"]),
            float(arrays["truth/tc"]),
            float(arrays["truth/phic"]),
            float(arrays["truth/inclination"]),
        ]
    )


def make_loss(
    theta_fixed: jnp.ndarray,
    detectors: dict,
    freqs: jnp.ndarray,
    band: jnp.ndarray,
    df: float,
    fs: float,
    n: int,
    f_ref: float,
):
    """Return ``-(ρ²_H1 + ρ²_L1 + ...)`` as a JIT-compiled value-and-grad fn."""

    def _rho_sq_det(theta: jnp.ndarray, det: dict) -> jnp.ndarray:
        hp, hc = gen_IMRPhenomD_hphc(freqs[band].astype(jnp.float64), theta, f_ref)
        h_b = (det["fplus"] * hp + det["fcross"] * hc).astype(jnp.complex128)
        h = jnp.zeros_like(freqs, dtype=jnp.complex128).at[band].set(h_b)
        hh = 4.0 * df * jnp.real(jnp.sum(jnp.abs(h) ** 2 / det["psd_safe"]))
        integrand = jnp.conj(det["d_full"]) * h / det["psd_safe"]
        pad = jnp.zeros(n, dtype=jnp.complex128).at[: integrand.shape[0]].set(integrand)
        z = 4.0 * fs * jnp.fft.ifft(pad)
        return jnp.max(jnp.abs(z) ** 2) / hh

    @jax.jit
    @jax.value_and_grad
    def loss(mc: jnp.ndarray) -> jnp.ndarray:
        theta = theta_fixed.at[0].set(mc)
        return -sum(_rho_sq_det(theta, det) for det in detectors.values())

    return loss


def fit_network_mc(
    arrays: dict,
    *,
    detectors: tuple[str, ...] = ("H1", "L1"),
    mc_init: float | None = None,
    mc_offset: float = 10.0,
    steps: int = 500,
    lr_init: float = 0.1,
    lr_final: float = 1e-3,
    tukey_alpha: float = 0.01,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> dict:
    fs = float(arrays.get("__sample_rate__", 4096.0))
    n = arrays[f"{detectors[0]}/strain"].shape[0]
    df = fs / n
    freqs = jnp.fft.rfftfreq(n, d=1.0 / fs)
    f_hi = fs / 2 if f_max is None else f_max
    band = (freqs >= f_min) & (freqs <= f_hi)
    f_ref = f_min

    theta_truth = theta_from_truth(arrays)
    mc_truth = float(theta_truth[0])
    if mc_init is None:
        mc_init = mc_truth + mc_offset

    det_data = build_detectors(arrays, detectors, fs, n, band, tukey_alpha)
    loss_fn = make_loss(theta_truth, det_data, freqs, band, df, fs, n, f_ref)

    mc = jnp.float64(mc_init)
    schedule = optax.cosine_decay_schedule(
        init_value=lr_init, decay_steps=steps, alpha=lr_final / lr_init
    )
    opt = optax.adam(schedule)
    state = opt.init(mc)

    hist_mc, hist_rho = [], []
    for _ in range(steps):
        value, grad = loss_fn(mc)
        hist_mc.append(float(mc))
        hist_rho.append(float(jnp.sqrt(-value)))
        if not np.isfinite(grad):
            print(f"non-finite gradient at Mc = {float(mc):.4f}; stopping")
            break
        updates, state = opt.update(grad, state)
        mc = optax.apply_updates(mc, updates)

    return {
        "mc_history": np.asarray(hist_mc),
        "rho_history": np.asarray(hist_rho),
        "mc_truth": mc_truth,
        "rho_truth_net": float(arrays["truth/snr"]),
        "detectors": detectors,
    }


def plot_history(result: dict, out_path: str) -> None:
    mc_hist = result["mc_history"]
    rho_hist = result["rho_history"]
    mc_truth = result["mc_truth"]
    rho_truth = result["rho_truth_net"]
    iters = np.arange(len(mc_hist))

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    ax0.plot(iters, mc_hist, color="C0")
    ax0.axhline(mc_truth, ls="--", color="crimson", label=f"truth Mc = {mc_truth:.3f}")
    ax0.set_xlabel("iteration")
    ax0.set_ylabel(r"$\mathcal{M}$ [$M_\odot$]")
    ax0.legend()

    sc = ax1.scatter(mc_hist, rho_hist, c=iters, cmap="viridis", s=20)
    ax1.plot(mc_hist, rho_hist, color="gray", alpha=0.3, lw=1)
    ax1.axvline(mc_truth, ls="--", color="crimson", alpha=0.7,
                label=f"truth Mc = {mc_truth:.3f}")
    ax1.axhline(rho_truth, ls="--", color="navy", alpha=0.7,
                label=fr"truth $\rho_\mathrm{{net}}$ = {rho_truth:.2f}")
    ax1.set_xlabel(r"$\mathcal{M}$ [$M_\odot$]")
    ax1.set_ylabel(r"$\rho_\mathrm{net}$")
    ax1.legend()
    fig.colorbar(sc, ax=ax1, label="iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="example_data/data_BBH_highSNR.h5",
                        help="HDF5 event with H1/L1 strain, PSD, antenna, truth.")
    parser.add_argument("--out", default="fit_history.png",
                        help="Where to write the SNR-vs-Mc trajectory plot.")
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"],
                        help="Detectors to include in the network sum.")
    parser.add_argument("--mc-init", type=float, default=None,
                        help="Starting chirp mass [Msun]. Defaults to truth + --mc-offset.")
    parser.add_argument("--mc-offset", type=float, default=10.0,
                        help="If --mc-init unset, start this far above truth Mc.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1, dest="lr_init")
    parser.add_argument("--lr-final", type=float, default=1e-3)
    parser.add_argument("--tukey-alpha", type=float, default=0.01)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=None,
                        help="Defaults to Nyquist.")
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    args = parser.parse_args()

    arrays = load_event(args.data)
    arrays["__sample_rate__"] = args.sample_rate

    result = fit_network_mc(
        arrays,
        detectors=tuple(args.detectors),
        mc_init=args.mc_init,
        mc_offset=args.mc_offset,
        steps=args.steps,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        tukey_alpha=args.tukey_alpha,
        f_min=args.f_min,
        f_max=args.f_max,
    )

    print(f"Initial Mc : {result['mc_history'][0]:.4f} Msun"
          f"   →  ρ_net = {result['rho_history'][0]:.3f}")
    print(f"Final   Mc : {result['mc_history'][-1]:.4f} Msun"
          f"   →  ρ_net = {result['rho_history'][-1]:.3f}")
    print(f"Truth   Mc : {result['mc_truth']:.4f} Msun"
          f"           (truth ρ_net = {result['rho_truth_net']:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_history(result, args.out)
    print(f"Wrote trajectory plot → {args.out}")


if __name__ == "__main__":
    main()
