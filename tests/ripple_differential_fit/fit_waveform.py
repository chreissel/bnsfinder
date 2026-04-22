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
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.TaylorF2 import gen_TaylorF2_hphc
from scipy.signal.windows import tukey

jax.config.update("jax_enable_x64", True)

DETECTORS = ("H1", "L1")


def _rfft_to_strain(x_td: np.ndarray, fs: float, alpha: float = 0.01) -> np.ndarray:
    """Windowed one-sided FFT in strain units (Hz^-1).

    Pipeline-standard practice is a very mild Tukey (alpha ~ 0.01): just
    enough to kill the FFT wrap-around discontinuity without biting
    into the in-band signal.
    """
    window = tukey(x_td.size, alpha=alpha)
    return np.fft.rfft(x_td * window) / fs


def _soft_onset_taper(freqs: np.ndarray, f_min: float, f_max: float,
                      f_width: float) -> np.ndarray:
    """Raised-cosine ramp from 0 at f_min to 1 at f_min + f_width; hard 0
    above f_max. Applied symmetrically to both data and template, this is
    the frequency-domain equivalent of the time-domain onset taper that
    pipelines (``pycbc.waveform.taper_timeseries``) use to suppress Gibbs
    ringing from the hard f_min cutoff. Returns ``T(f)`` (not T^2).
    """
    if f_width <= 0.0:
        return ((freqs >= f_min) & (freqs <= f_max)).astype(np.float64)
    x = np.clip((freqs - f_min) / f_width, 0.0, 1.0)
    ramp = 0.5 * (1.0 - np.cos(np.pi * x))
    return ramp * (freqs <= f_max).astype(np.float64)


def _inverse_spectrum_truncation(psd: np.ndarray, fs: float,
                                 max_filter_len: float,
                                 band_mask: np.ndarray) -> np.ndarray:
    """Truncate the 1/sqrt(S_n) whitening kernel to finite time duration.

    Without this, the whitening filter's impulse response has a long tail
    that wraps around the segment via the FFT's implicit periodicity and
    biases the matched filter. Standard procedure (``pycbc.psd.
    inverse_spectrum_truncation``): iFFT the inverse ASD, keep a central
    window of length ``max_filter_len`` seconds, FFT back, square to get
    a truncated PSD. ``band_mask`` zeroes the inverse ASD outside
    [f_min, f_max] before the iFFT.
    """
    if max_filter_len <= 0.0:
        return psd
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_asd = np.where(band_mask, 1.0 / np.sqrt(psd), 0.0)
    kernel = np.fft.irfft(inv_asd)
    n = kernel.size
    half_len = int(round(max_filter_len * fs / 2.0))
    half_len = min(half_len, n // 2 - 1)
    # Roll so the kernel's peak at lag 0 sits at index n//2, window, unroll.
    centered = np.roll(kernel, n // 2)
    w = np.zeros(n)
    lo, hi = n // 2 - half_len, n // 2 + half_len
    w[lo:hi] = tukey(hi - lo, alpha=0.25)
    centered *= w
    truncated = np.roll(centered, -(n // 2))
    inv_asd_new = np.fft.rfft(truncated)
    return 1.0 / np.maximum(np.abs(inv_asd_new) ** 2, 1e-300)


def load_data(path: Path, tukey_alpha: float = 0.01,
              waveform: str = "imrphenomd",
              f_taper_width: float = 1.0,
              psd_max_filter_length: float = 4.0) -> dict:
    with h5py.File(path, "r") as f:
        fs = float(f.attrs["sample_rate"])
        f_ref = float(f.attrs["f_ref"])
        f_min = float(f.attrs.get("f_min", 20.0))
        f_max = float(f.attrs.get("f_max", fs / 2.0))

        lengths = {det: f[f"{det}/strain"].shape[0] for det in DETECTORS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Detector time series lengths differ: {lengths}")
        n = lengths["H1"]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        band_mask = (freqs >= f_min) & (freqs <= f_max)
        # Soft onset taper, applied symmetrically to both data and template
        # via the inner-product integrand weight T(f)^2.
        taper = _soft_onset_taper(freqs, f_min, f_max, f_taper_width)
        weight = taper ** 2
        # Mask used for gradient-safety in ripple evaluation (hard bool).
        mask = band_mask

        detectors = {}
        for det in DETECTORS:
            td = np.asarray(f[f"{det}/strain"][()], dtype=np.float64)
            psd = np.asarray(f[f"{det}/psd"][()], dtype=np.float64)
            if psd.shape[0] != freqs.shape[0]:
                raise ValueError(
                    f"{det}: PSD length {psd.shape[0]} does not match "
                    f"rFFT grid length {freqs.shape[0]}"
                )
            # Pipeline-standard inverse spectrum truncation.
            psd_tr = _inverse_spectrum_truncation(
                psd, fs, psd_max_filter_length, band_mask,
            )
            # Replace out-of-band PSD values with a finite sentinel so
            # division is safe even when the weight drops the contribution.
            psd_safe = np.where(band_mask, psd_tr, 1.0)
            detectors[det] = {
                "strain": jnp.asarray(_rfft_to_strain(td, fs, tukey_alpha),
                                      dtype=jnp.complex128),
                "psd": jnp.asarray(psd_safe, dtype=jnp.float64),
                "fplus": float(f[f"antenna/{det}/fplus"][()]),
                "fcross": float(f[f"antenna/{det}/fcross"][()]),
            }

        truth = None
        if "truth" in f:
            truth = {k: np.asarray(v[()]) for k, v in f["truth"].items()}

    return {
        "freqs": jnp.asarray(freqs, dtype=jnp.float64),
        "mask": jnp.asarray(mask, dtype=bool),
        "weight": jnp.asarray(weight, dtype=jnp.float64),
        "sample_rate": fs,
        "duration": n / fs,
        "f_ref": f_ref,
        "f_min": f_min,
        "f_max": f_max,
        "truth": truth,
        "waveform": waveform,
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


def ripple_polarizations(theta, freqs, f_ref, waveform: str):
    """h+, hx in frequency domain from ripple.

    theta layout is the 8-element BBH vector:
        [Mc, eta, chi1, chi2, D, tc, phic, inclination]

    For ``waveform='taylorf2'`` the ml4gw-compatible point-particle
    TaylorF2 is used (lambda1 = lambda2 = 0, no tides), which matches
    the waveform GWDatasetGeneration's BNS path produces.
    """
    if waveform == "imrphenomd":
        return gen_IMRPhenomD_hphc(freqs, theta, f_ref)
    if waveform == "taylorf2":
        theta_tf2 = jnp.array([
            theta[0], theta[1], theta[2], theta[3],
            0.0, 0.0,  # lambda1, lambda2 -> point-particle
            theta[4], theta[5], theta[6], theta[7],
        ])
        return gen_TaylorF2_hphc(
            freqs, theta_tf2, f_ref, use_lambda_tildes=False,
        )
    raise ValueError(f"Unknown waveform: {waveform!r}")


def project(hp, hc, fplus: float, fcross: float):
    return fplus * hp + fcross * hc


def inner_product(a, b, psd, df, weight):
    """Real matched-filter inner product 4 df Re Σ T^2 · conj(a) b / S_n.

    ``weight = T(f)^2`` applies the symmetric soft onset taper; treating
    it as an integrand factor is equivalent to ``⟨T·a | T·b⟩`` so data
    and template are filtered identically.
    """
    integrand = weight * jnp.conj(a) * b / psd
    return 4.0 * df * jnp.real(jnp.sum(integrand))


def complex_inner_product(a, b, psd, df, weight):
    """Complex matched-filter inner product 4 df Σ T^2 · conj(a) b / S_n.

    Taking |·| of this phase-maximises over the coalescence phase.
    """
    integrand = weight * jnp.conj(a) * b / psd
    return 4.0 * df * jnp.sum(integrand)


def network_stats(theta, data):
    """Compute (log_likelihood, rho_sq_net, rho_net) for one parameter set."""
    freqs = data["freqs"]
    mask = data["mask"]
    weight = data["weight"]
    f_min = data["f_min"]
    df = freqs[1] - freqs[0]
    # Ripple's phase/amplitude have 1/f terms, so evaluating it at f=0 or
    # outside its support produces NaN/inf values *and* NaN gradients. The
    # outer mask zeroes those contributions in the forward pass, but
    # autodiff still runs chain rule through the unsafe bins
    # (NaN * 0 = NaN). Replace every out-of-band frequency with a safe
    # in-band value (f_min) so ripple only ever sees finite inputs.
    safe_freqs = jnp.where(mask, freqs, f_min)
    hp, hc = ripple_polarizations(
        theta, safe_freqs, data["f_ref"], data["waveform"],
    )
    hp = jnp.where(mask, hp, 0.0 + 0.0j)
    hc = jnp.where(mask, hc, 0.0 + 0.0j)

    ll = jnp.array(0.0)
    rho_sq = jnp.array(0.0)
    for det in DETECTORS:
        d = data[det]
        h = project(hp, hc, d["fplus"], d["fcross"])
        dh_real = inner_product(d["strain"], h, d["psd"], df, weight)
        dh_complex = complex_inner_product(d["strain"], h, d["psd"], df, weight)
        hh = inner_product(h, h, d["psd"], df, weight)
        ll = ll + dh_real - 0.5 * hh
        # Phase-maximised per-detector ρ² = |⟨d|h⟩|² / ⟨h|h⟩
        rho_sq = rho_sq + (dh_complex * jnp.conj(dh_complex)).real / hh
    return ll, rho_sq, jnp.sqrt(rho_sq)


def make_fit_fns(data, fixed_rest, loss_kind: str = "snr"):
    """Return jitted loss, grad and SNR functions that take only chirp mass.

    loss_kind:
        "snr"  — minimise -ρ²_net (maximise phase-maximised network SNR²).
        "logl" — minimise -log L (Gaussian matched-filter log-likelihood).
    """
    fixed_rest = jnp.asarray(fixed_rest, dtype=jnp.float64)

    def theta_of(mc):
        return jnp.concatenate([jnp.array([mc]), fixed_rest])

    if loss_kind == "snr":
        def loss(mc):
            _, rho_sq, _ = network_stats(theta_of(mc), data)
            return -rho_sq
    elif loss_kind == "logl":
        def loss(mc):
            ll, _, _ = network_stats(theta_of(mc), data)
            return -ll
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind!r}")

    def snr(mc):
        _, _, s = network_stats(theta_of(mc), data)
        return s

    return jax.jit(jax.value_and_grad(loss)), jax.jit(snr)


def _build_schedule(lr: float, lr_final: float, n_steps: int, kind: str):
    """Step-size schedule for Adam. Bigger for exploration, smaller for refine."""
    if kind == "const" or lr_final is None or lr_final == lr:
        return optax.constant_schedule(lr)
    if kind == "cosine":
        return optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=max(n_steps - 1, 1),
            alpha=lr_final / lr,
        )
    if kind == "exponential":
        if lr <= 0 or lr_final <= 0:
            raise ValueError("exponential schedule requires positive lr / lr_final")
        rate = (lr_final / lr) ** (1.0 / max(n_steps - 1, 1))
        return optax.exponential_decay(
            init_value=lr, transition_steps=1, decay_rate=rate,
        )
    raise ValueError(f"Unknown lr schedule: {kind!r}")


def fit(data, init_theta, n_steps: int, lr: float, loss_kind: str = "snr",
        lr_final: float | None = None, lr_schedule: str = "cosine"):
    init_theta = np.asarray(init_theta, dtype=np.float64)
    mc = jnp.asarray(init_theta[0])
    loss_grad_fn, snr_fn = make_fit_fns(data, init_theta[1:], loss_kind)

    schedule = _build_schedule(lr, lr_final, n_steps, lr_schedule)
    opt = optax.adam(schedule)
    opt_state = opt.init(mc)

    history = {"mc": [], "snr": [], "loss": [], "lr": []}
    for step in range(n_steps):
        loss, grad = loss_grad_fn(mc)
        snr = snr_fn(mc)
        history["mc"].append(float(mc))
        history["snr"].append(float(snr))
        history["loss"].append(float(loss))
        history["lr"].append(float(schedule(step)))
        if not np.isfinite(grad):
            print(f"[step {step}] non-finite gradient (mc={float(mc):.4f}, "
                  f"loss={float(loss)}, snr={float(snr)}); stopping")
            break
        updates, opt_state = opt.update(grad, opt_state)
        mc = optax.apply_updates(mc, updates)

    return float(mc), history


def plot_history(history, out_path: Path, truth: dict | None = None) -> None:
    mcs = np.asarray(history["mc"])
    snrs = np.asarray(history["snr"])
    lrs = np.asarray(history.get("lr", []))
    iters = np.arange(len(mcs))

    has_lr = lrs.size == iters.size and lrs.size > 0
    if has_lr:
        fig, (ax, ax_lr) = plt.subplots(
            1, 2, figsize=(11, 5),
            gridspec_kw={"width_ratios": [2.5, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(mcs, snrs, "-", color="gray", alpha=0.4, zorder=1)
    sc = ax.scatter(mcs, snrs, c=iters, cmap="viridis", s=18, zorder=2)
    if truth is not None:
        if "chirp_mass" in truth:
            ax.axvline(_float(truth["chirp_mass"]), ls="--", color="crimson",
                       lw=1, alpha=0.7, label="truth $\\mathcal{M}$")
        if "snr" in truth:
            ax.axhline(_float(truth["snr"]), ls="--", color="navy",
                       lw=1, alpha=0.7, label="truth network SNR")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=9)
    ax.set_xlabel(r"Chirp mass $\mathcal{M}$ [$M_\odot$]")
    ax.set_ylabel("Network SNR")
    ax.set_title("Differential fit trajectory")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iteration")

    if has_lr:
        ax_lr.plot(iters, lrs, color="C2")
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("Iteration")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_title("Step-size schedule")
        ax_lr.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path,
                   help="HDF5 file with H1/L1 time-series strain, PSD, and antenna factors")
    p.add_argument("--out", default=Path("fit_history.png"), type=Path,
                   help="Output path for the SNR-vs-mass plot")
    p.add_argument("--tukey-alpha", type=float, default=0.01,
                   help="Tukey window alpha applied to the time series before "
                        "rFFT. Production pipelines use ~0.01 so the taper "
                        "only kills the FFT-wrap discontinuity, not in-band "
                        "signal. Set 0 to disable (risky).")
    p.add_argument("--f-taper-width", type=float, default=1.0,
                   help="Width [Hz] of the raised-cosine onset taper applied "
                        "symmetrically to data and template at f_min. "
                        "Frequency-domain equivalent of PyCBC's "
                        "taper_timeseries; suppresses Gibbs ringing from the "
                        "hard f_min cutoff. Set 0 for a hard cut.")
    p.add_argument("--psd-max-filter-length", type=float, default=4.0,
                   help="Duration [s] to truncate the 1/sqrt(S_n) whitening "
                        "kernel to before matched filtering (inverse spectrum "
                        "truncation). Prevents the whitening filter's impulse "
                        "response from wrapping around the segment and biasing "
                        "the SNR. Set 0 to disable.")
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
    p.add_argument("--lr", type=float, default=0.1,
                   help="Initial Adam learning rate (explore phase).")
    p.add_argument("--lr-final", type=float, default=1e-3,
                   help="Final Adam learning rate at the last step "
                        "(refine phase). Ignored if --lr-schedule const.")
    p.add_argument("--lr-schedule", choices=("cosine", "exponential", "const"),
                   default="cosine",
                   help="Learning-rate schedule from --lr down to --lr-final.")
    p.add_argument("--init-from-truth", action="store_true",
                   help="Initialise all non-mass parameters from the /truth "
                        "group in the HDF5 (simulation-only info; use only "
                        "for sanity checks). Chirp mass still starts at --mc-init.")
    p.add_argument("--waveform", choices=("imrphenomd", "taylorf2"),
                   default="imrphenomd",
                   help="Template waveform family. Use 'taylorf2' for BNS "
                        "injections generated by GWDatasetGeneration's BNS "
                        "config (point-particle, no tides).")
    p.add_argument("--loss", choices=("snr", "logl"), default="snr",
                   help="Optimisation target. 'snr' maximises the "
                        "phase-maximised network SNR squared (standard CBC "
                        "search statistic, distance-independent); 'logl' "
                        "maximises the Gaussian matched-filter log-likelihood.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(
        args.data,
        tukey_alpha=args.tukey_alpha,
        waveform=args.waveform,
        f_taper_width=args.f_taper_width,
        psd_max_filter_length=args.psd_max_filter_length,
    )
    print(f"Waveform template:    {args.waveform}")
    print(f"Tukey alpha:          {args.tukey_alpha}")
    print(f"Onset taper width:    {args.f_taper_width:.2f} Hz")
    print(f"PSD truncation:       "
          f"{args.psd_max_filter_length:.2f} s"
          f"{'  (disabled)' if args.psd_max_filter_length <= 0 else ''}")

    truth = data["truth"]
    if truth is not None:
        if "chirp_mass" in truth:
            print(f"Truth chirp mass:     {_float(truth['chirp_mass']):.4f} Msun")
        if "snr" in truth:
            print(f"Truth network SNR:    {_float(truth['snr']):.3f}")

    if args.init_from_truth:
        init_from_truth(args, truth)

    init_theta = [
        args.mc_init, args.eta, args.chi1, args.chi2,
        args.dist, args.tc, args.phic, args.inclination,
    ]
    mc_fit, history = fit(data, init_theta, n_steps=args.steps, lr=args.lr,
                          loss_kind=args.loss,
                          lr_final=args.lr_final,
                          lr_schedule=args.lr_schedule)

    print(f"Recovered chirp mass: {mc_fit:.4f} Msun")
    print(f"Final network SNR:    {history['snr'][-1]:.3f}")
    plot_history(history, args.out, truth=truth)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
