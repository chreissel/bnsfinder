"""Network chirp-mass fit via differentiable matched filtering.

Mirrors the chirp-mass fit in ``FirstTests.ipynb``: builds per-detector
matched-filter inputs from a single HDF5 event (the format produced by
``generate_fit_input.py``) and gradient-descends on the time-maximised
network SNR squared. PyTorch autograd differentiates straight through an
``ml4gw.waveforms.IMRPhenomD`` template -- the same waveform model
``GWDatasetGeneration`` uses to *make* the injections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from ml4gw.waveforms import IMRPhenomD

torch.set_default_dtype(torch.float64)


# --- Two shims make ml4gw's IMRPhenomD differentiable. It is written for
# forward evaluation, so a couple of ops have no usable autograd path. ---

# shim 1: torch.heaviside has no gradient rule. ml4gw only uses it to build
# piecewise-constant PN-region masks, whose derivative is 0 almost
# everywhere, so a detached version is the correct gradient.
_orig_heaviside = torch.heaviside
torch.heaviside = lambda inp, val: _orig_heaviside(inp.detach(), val.detach())


# shim 2: phenom_d_mrd_amp does `exp(...) *= ...` in place, which corrupts
# that exp's backward pass. Re-bind an out-of-place equivalent.
def _phenom_d_mrd_amp(self, Mf, eta, eta2, chi1, chi2, xi, fRD, fDM):
    g1 = self.gamma1_fun(eta, eta2, xi)
    g2 = self.gamma2_fun(eta, eta2, xi)
    g3 = self.gamma3_fun(eta, eta2, xi)
    fDMg3 = fDM * g3
    pow2 = (torch.ones_like(Mf).mT * fDMg3 * fDMg3).mT
    fminfRD = Mf - (torch.ones_like(Mf).mT * fRD).mT
    etl = torch.exp(fminfRD.mT * g2 / fDMg3).mT
    etl = etl * (fminfRD ** 2 + pow2)            # upstream uses `*=` here
    amp = (1 / etl.mT * g1 * g3 * fDM).mT
    Damp = (fminfRD.mT * -2 * fDM * g1 * g3) / (
        fminfRD * fminfRD + pow2
    ).mT - (g2 * g1)
    return amp, Damp.mT / etl


IMRPhenomD.phenom_d_mrd_amp = _phenom_d_mrd_amp


def load_event(path: str) -> dict:
    arrays: dict = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda name, obj: arrays.update({name: obj[...]})
            if isinstance(obj, h5py.Dataset)
            else None
        )
    return arrays


def tukey_window(n: int, alpha: float = 0.01) -> torch.Tensor:
    n_taper = int(round(alpha * n / 2))
    if n_taper == 0:
        return torch.ones(n, dtype=torch.float64)
    ramp = 0.5 * (1 - torch.cos(torch.pi * (torch.arange(n_taper) + 0.5) / n_taper))
    return torch.cat([ramp, torch.ones(n - 2 * n_taper), ramp.flip(0)])


def build_detectors(
    arrays: dict,
    detectors: tuple[str, ...],
    fs: float,
    n: int,
    band: torch.Tensor,
    tukey_alpha: float = 0.01,
) -> dict:
    """Per-detector matched-filter inputs as torch tensors."""
    window_td = tukey_window(n, tukey_alpha)
    out = {}
    for det in detectors:
        strain = torch.tensor(arrays[f"{det}/strain"], dtype=torch.float64)
        psd = torch.tensor(arrays[f"{det}/psd"], dtype=torch.float64)
        out[det] = {
            "d_full": torch.fft.rfft(strain * window_td) / fs,
            "psd_safe": torch.where(band, psd, torch.inf),
            "fplus": float(arrays[f"antenna/{det}/fplus"]),
            "fcross": float(arrays[f"antenna/{det}/fcross"]),
        }
    return out


def fixed_params_from_truth(arrays: dict) -> dict:
    """Non-mass IMRPhenomD parameters, held fixed at their truth values.

    ml4gw's IMRPhenomD takes ``mass_ratio`` directly (no eta) and has no
    ``tc`` argument -- the matched filter's max_t absorbs the arrival time.
    """
    def _const(v: float) -> torch.Tensor:
        return torch.tensor([float(v)], dtype=torch.float64)

    return dict(
        mass_ratio=_const(arrays["truth/mass_ratio"]),
        chi1=_const(arrays["truth/chi1"]),
        chi2=_const(arrays["truth/chi2"]),
        distance=_const(arrays["truth/distance"]),
        phic=_const(arrays["truth/phic"]),
        inclination=_const(arrays["truth/inclination"]),
    )


def make_loss(
    phenom: IMRPhenomD,
    fixed_params: dict,
    detectors: dict,
    freqs: torch.Tensor,
    band: torch.Tensor,
    band_idx: torch.Tensor,
    df: float,
    fs: float,
    n: int,
    f_ref: float,
):
    """Return ``-(ρ²_H1 + ρ²_L1 + ...)`` as a differentiable function of Mc.

    Each ρ² is time-maximised with a single inverse FFT, so there is no need
    to also fit tc: as Mc moves the merger drifts inside the segment and
    ``max_t`` tracks it automatically.
    """

    def loss(mc: torch.Tensor) -> torch.Tensor:
        hc, hp = phenom(freqs[band], chirp_mass=mc, f_ref=f_ref, **fixed_params)
        rho_sq = mc.new_zeros(())
        for det in detectors.values():
            h_band = det["fplus"] * hp[0] + det["fcross"] * hc[0]
            h = torch.zeros(n // 2 + 1, dtype=torch.complex128).index_put(
                (band_idx,), h_band
            )
            hh = 4.0 * df * torch.sum(torch.abs(h) ** 2 / det["psd_safe"]).real
            integrand = torch.conj(det["d_full"]) * h / det["psd_safe"]
            pad = torch.zeros(n, dtype=torch.complex128).index_put(
                (torch.arange(integrand.shape[0]),), integrand
            )
            z = 4.0 * fs * torch.fft.ifft(pad)
            rho_sq = rho_sq + torch.max(torch.abs(z) ** 2) / hh
        return -rho_sq

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
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs)
    f_hi = fs / 2 if f_max is None else f_max
    band = (freqs >= f_min) & (freqs <= f_hi)
    band_idx = band.nonzero().squeeze()
    f_ref = f_min

    mc_truth = float(arrays["truth/chirp_mass"])
    if mc_init is None:
        mc_init = mc_truth + mc_offset

    det_data = build_detectors(arrays, detectors, fs, n, band, tukey_alpha)
    phenom = IMRPhenomD()
    fixed_params = fixed_params_from_truth(arrays)
    loss_fn = make_loss(
        phenom, fixed_params, det_data, freqs, band, band_idx, df, fs, n, f_ref
    )

    mc = torch.tensor([mc_init], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mc], lr=lr_init)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr_final
    )

    hist_mc, hist_rho = [], []
    for _ in range(steps):
        opt.zero_grad()
        value = loss_fn(mc)
        value.backward()
        hist_mc.append(mc.item())
        hist_rho.append(float(torch.sqrt(-value.detach())))
        if not torch.isfinite(mc.grad).all():
            print(f"non-finite gradient at Mc = {mc.item():.4f}; stopping")
            break
        opt.step()
        schedule.step()

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
