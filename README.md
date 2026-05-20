# ripple_differential_fit

Differential fitting of a [ripple](https://github.com/tedwards2412/ripple)
gravitational-wave waveform to H1/L1 strain data. The chirp mass is
recovered by gradient-descending the time-maximised matched-filter SNR
with JAX + optax; the trajectory in (Mc, ρ) is recorded along the way.

Upstream dataset generator
[`GWDatasetGeneration`](https://github.com/chreissel/GWDatasetGeneration)
is vendored as a git submodule under `GWDatasetGeneration/` and is used
to produce the fit input.

## Repository layout

```
FirstTests.ipynb        step-by-step walkthrough of all the building blocks
fit_waveforms.py        standalone network chirp-mass fit (no notebook)
generate_fit_input.py   build a single-event HDF5 from GWDatasetGeneration
example_data/           pre-baked BBH events (high / low network SNR)
scripts/make_data_plots.py   regenerate the README data-overview plots
plots/                  PNGs embedded below
```

## Install

```
git submodule update --init
pip install -r requirements.txt
```

`ripplegw` pulls in JAX. Install a matching `jax[cuda]` wheel first if
you want GPU acceleration. `ml4gw` pulls in PyTorch and is only needed
for the data-generation wrapper (`generate_fit_input.py`), not for the
fit itself.

## Input data

Both the notebook and `fit_waveforms.py` read a single HDF5 event that
contains both detectors' time-domain strain (signal + background), their
PSDs on the matching rFFT grid, and the antenna-pattern factors for the
source sky location and time:

```
/H1/strain                       float   [N]         time-domain strain at H1
/H1/psd                          float   [N//2 + 1]  one-sided PSD on rfftfreq(N, 1/fs)
/L1/strain, /L1/psd              same as H1
/antenna/{H1,L1}/{fplus,fcross}  scalar              F+, Fx for that IFO
/truth/*                         scalar              chirp_mass, mass_ratio, chi1/2, distance,
                                                     tc, phic, inclination, snr, snr_H1, snr_L1
attrs: sample_rate, f_min, f_max, f_ref, duration
```

Two pre-generated BBH events live in `example_data/`
(`data_BBH_highSNR.h5`, `data_BBH_lowSNR.h5`). Make your own with
`generate_fit_input.py`:

```
python generate_fit_input.py \
    --config GWDatasetGeneration/configs/config_BBH.yaml \
    --data   /path/to/background_data \
    --out    data.h5
```

The generator mirrors upstream `injections.py`:

- **SNR reweighting is applied by default.** Each waveform's amplitude
  is rescaled so the network SNR matches a draw from
  `config.snr_reweighting` (e.g. `PowerLaw[12,100,-3]` in the BBH
  config). Pass `--no-reweight` to use the raw distance-implied SNR
  instead.
- **Whitening is intentionally skipped.** The matched-filter inner
  product `4 Δf Re Σ d*·h / S_n` already "whitens" both sides via the
  `1/S_n` factor, so the fit needs raw coloured strain + PSD rather
  than a pre-whitened time series.

## What the data looks like

Raw 8-second, 4096 Hz strain at both detectors — a high-SNR BBH chirp
buried in real O3a noise:

![Strain time series](plots/strain_timeseries.png)

The PSDs are the H1 and L1 noise floors over the same segment. The
band `[f_min, f_max]` used in the inner product (default `[20, fs/2]`)
keeps the `f = 0` bin (where ripple's IMRPhenomD is NaN and the PSD is
~0) out of the sum:

![PSDs](plots/psd.png)

Whitening the H1 strain by `1/√S_n` and band-passing 30–400 Hz makes
the chirp visible by eye; the Q-transform shows the characteristic
frequency sweep ramping up to merger at `truth/tc`:

![Whitened + Q-transform](plots/whitened_qtransform.png)

## What the notebook does

`FirstTests.ipynb` walks through every component, in order:

1. **Load** the HDF5 event and unpack arrays / truth parameters into a
   ripple parameter vector `[Mc, η, χ1, χ2, D, tc, φc, ι]`.
2. **Visualise** the raw and whitened strain plus a Q-transform around
   the merger (the plots above).
3. **Baseline matched filter at truth parameters.** Builds `d̃(f)` by
   FFT'ing a Tukey-windowed strain segment, generates the template
   `h̃(f) = F+·h+ + F×·h×` on the rFFT grid, and computes
   `√⟨h|h⟩` to confirm it matches the stored per-IFO truth SNR.
4. **Time-maximised matched filter.** Zero-pads the one-sided
   integrand `conj(d̃)·h̃ / S_n` back to length N, inverse-FFTs it to
   get `z(t)`, and recovers the coalescence-time shift from
   `argmax |z(t)|`.
5. **Fit chirp mass with JAX** (single IFO) — `differential_matched_filter`
   regenerates the template at each Mc and gradient-descends on
   `-max_t |z(t)|² / ⟨h|h⟩`. The `max_t` step absorbs the merger-time
   drift induced by changes in Mc, so tc need not be fit.
6. **Fit arrival time alone.** `differential_matched_filter_tc`
   phase-shifts the truth template by `exp(-2πi f Δtc)` (FT shift
   theorem — no template regeneration) and gradient-descends on
   `-|⟨d|h_{Δtc}⟩|² / ⟨h|h⟩`.
7. **Joint Mc + Δtc fit.** `differential_matched_filter_joint`
   combines both: phase-maximised SNR² (no `max_t`) so Δtc has a real
   gradient signal instead of being absorbed by the time-max.
8. **Network fit.** `differential_matched_filter_network` sums the
   time-maximised ρ² across H1 and L1, each with its own
   `(d̃, S_n, F+, F×)`. This is exactly the loss exported by
   `fit_waveforms.py`.
9. **Network fit with PyTorch + ml4gw.** `network_loss_torch` rebuilds
   the exact same network fit on PyTorch autograd, using
   `ml4gw.waveforms.IMRPhenomD` (the model `GWDatasetGeneration` uses to
   make the injections) instead of `ripple`. Two shims make the ml4gw
   waveform differentiable — a detached `torch.heaviside` and an
   out-of-place `phenom_d_mrd_amp` — and the recovered chirp mass /
   network SNR match the JAX fit, since matched-filter ρ is normalised.

## `fit_waveforms.py` — the network fit, scripted

The standalone script reproduces step 8 from the notebook without the
preceding pedagogy. Internally it:

- loads the HDF5 event (`load_event`),
- builds per-detector `(d̃, 1/S_n, F+, F×)` with a Tukey-windowed FFT
  (`build_detectors`, `tukey_window`),
- builds a ripple parameter vector from `/truth` (`theta_from_truth`),
- constructs a `jax.jit`-ed `jax.value_and_grad` loss
  `-(ρ²_H1 + ρ²_L1)` where each per-detector ρ² is time-maximised via
  one inverse FFT (`make_loss`),
- runs Adam with a cosine-decayed learning rate (`fit_network_mc`),
- and writes a two-panel trajectory plot (`plot_history`).

Usage:

```
python fit_waveforms.py \
    --data example_data/data_BBH_highSNR.h5 \
    --out  plots/fit_history.png \
    --mc-offset 10.0 --steps 500 --lr 0.1 --lr-final 1e-3
```

Defaults reproduce the notebook's network fit: start `mc_truth + 10`
Msun off, 500 Adam steps with `lr 0.1 → 1e-3` cosine decay. Only the
chirp mass is updated; all other ripple parameters are held at their
truth values. To fit additional parameters, extend `make_loss` in
`fit_waveforms.py` to return gradients over the full vector.

Running on `example_data/data_BBH_highSNR.h5` recovers truth
(Mc ≈ 32.40 Msun) from a +10 Msun offset and reaches the truth network
SNR within the last few iterations:

![Network chirp-mass fit](plots/fit_history.png)

Stdout reports the initial / final / truth (Mc, ρ_net) triple.

## Regenerating the README plots

```
python scripts/make_data_plots.py                       # data overview
python fit_waveforms.py --out plots/fit_history.png     # fit trajectory
```

Plots write to `plots/` and are committed; everything else under
`*.png` stays gitignored.

## Notes

- Uses `ripple.waveforms.IMRPhenomD.gen_IMRPhenomD_hphc`. For BNS with
  tidal effects swap in `IMRPhenomD_NRTidalv2` and extend the parameter
  vector with `lambda1, lambda2` (and consider TaylorF2 instead).
- Matched-filter inner product is `4 Δf Re Σ d*·h / S_n`, assuming a
  one-sided PSD on a uniform frequency grid.
- The network sum is **incoherent** — each detector's peak is chosen
  independently by `max_t |z_j(t)|²`, so H1 and L1 are not constrained
  to share a common geocentric arrival time. This matches
  coincident-search conventions, not coherent ones.
- Antenna factors are loaded as constants from the HDF5; extending the
  fit to sky location would make them functions of `(ra, dec, ψ, t_gps)`.
