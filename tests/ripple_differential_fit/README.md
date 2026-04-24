# ripple_differential_fit

Differential fitting of a ripple gravitational-wave waveform to H1/L1
strain data. Chirp mass is recovered via gradient descent on the matched-
filter log-likelihood; network SNR is tracked along the trajectory.

Upstream dataset generator [`GWDatasetGeneration`](https://github.com/chreissel/GWDatasetGeneration)
is vendored as a git submodule under `GWDatasetGeneration/` and is used to
produce the fit input.

## Install

```
git submodule update --init
pip install -r requirements.txt
```

`ripplegw` pulls in JAX. Install a matching `jax[cuda]` wheel first if you
want GPU acceleration. `ml4gw` pulls in PyTorch and is only needed for the
data-generation wrapper, not for the fit itself.

## Generate fit input from GWDatasetGeneration

Upstream writes batched, *whitened* time series (`sig_{i}.h5`, `bkg_{i}.h5`).
The differential fit needs a single event with raw strain, PSD, and antenna
factors. Use `generate_fit_input.py` — it reuses the waveform + PSD path
from the submodule but skips whitening and adds the missing fields:

```
python generate_fit_input.py \
    --config GWDatasetGeneration/configs/config_BBH.yaml \
    --data   /path/to/background_data \
    --out    data.h5
```

Download background data once with the submodule's `load_data.py`. Truth
parameters sampled during generation are preserved under `/truth` in the
output HDF5 for comparison against the fit. Alongside `/truth/snr`
(network), the generator also writes per-IFO truth SNRs
(`/truth/snr_H1`, `/truth/snr_L1`) via `ml4gw.gw.compute_ifo_snr`, so a
single-detector fit can be compared against the matching truth value.

The generator mirrors upstream `injections.py` in two places:

- **SNR reweighting is applied by default.** Each waveform's amplitude is
  rescaled so the network SNR matches a draw from the
  `config.snr_reweighting` distribution (e.g. `PowerLaw[12,100,-3]` in
  the BBH config). Pass `--no-reweight` to use the raw SNR implied by
  the sampled distance instead.
- **Whitening is intentionally skipped.** The matched-filter inner
  product `4 Δf Re Σ d*·h / S_n` already "whitens" both sides via the
  `1/S_n` factor, so the fit needs raw coloured strain + PSD rather than
  a pre-whitened time series.

## Input data

`fit_waveform.py` reads a single HDF5 file (e.g. produced by
`generate_fit_input.py` above) that contains both detectors' *time-domain*
strain (signal + background), their PSDs on the matching rFFT grid, and
the antenna-pattern factors for the source sky location and time.

```
/H1/strain                     float64   [N]         time-domain strain at H1
/H1/psd                        float64   [N//2 + 1]  one-sided PSD on rfftfreq(N, 1/fs)
/L1/strain, /L1/psd            same as H1
/antenna/H1/fplus, /fcross     scalar    F+, Fx for H1
/antenna/L1/fplus, /fcross     scalar    F+, Fx for L1
attrs:
  sample_rate                  scalar    time-series sample rate [Hz]
  f_ref                        scalar    waveform reference frequency [Hz]
  f_min                        scalar    low-frequency cutoff [Hz] (default 20)
  f_max                        scalar    high-frequency cutoff [Hz] (default Nyquist)
```

The inner product is restricted to the `[f_min, f_max]` band. This keeps
the `f = 0` bin (where the IMRPhenomD waveform is NaN and the PSD is ~0)
out of the sum, and matches the band used during data generation.

H1 and L1 time series must share length `N`. The loader applies a Tukey
window (alpha configurable via `--tukey-alpha`) and takes the one-sided
FFT, scaled by `1/fs` to yield strain in `Hz^-1`.

## Run

```
python fit_waveform.py --data data.h5 --out fit_history.png \
    --mc-init 1.4 --eta 0.24 --steps 300 --lr 0.01
```

Non-mass parameters (`--eta`, `--chi1`, `--chi2`, `--dist`, `--tc`,
`--phic`, `--inclination`) are held fixed at their initial values; only
chirp mass is fit. To fit additional parameters, expand
`make_fit_fns` in `fit_waveform.py` to return gradients w.r.t. the full
parameter vector.

### Learning-rate schedule (explore then refine)

Adam's step size follows a schedule selected by `--lr-schedule`:

- `cosine` (default) — smooth decay from `--lr` down to `--lr-final` over
  `--steps` iterations. Early steps are large enough to traverse flat
  regions of the loss; late steps shrink so the fit can settle into the
  peak without overshooting.
- `exponential` — multiplicative decay from `--lr` to `--lr-final`.
- `const` — no decay; use the old fixed-rate behaviour (ignores
  `--lr-final`).

Defaults are `--lr 0.1 --lr-final 1e-3`; pick a larger `--lr` if the
trajectory gets stuck in a local feature. The effective LR at each step
is recorded in `history["lr"]` and drawn alongside the SNR-vs-mass plot.

### Matched-filter data conditioning

`fit_waveform.py` applies three pipeline-standard preprocessing steps
before the inner product. Defaults mirror what PyCBC/GstLAL use:

- **Mild Tukey window on the segment (`--tukey-alpha 0.01`).** Kills the
  FFT wrap-around discontinuity without eating in-band signal.
- **Symmetric soft onset taper at `f_min` (`--f-taper-width 1.0 Hz`).**
  A raised-cosine ramp `T(f)` rising from 0 at `f_min` to 1 at
  `f_min + f_taper_width`, applied to both data and template as an
  integrand factor `T(f)^2` in the inner product. Equivalent to
  `pycbc.waveform.taper_timeseries`; suppresses Gibbs ringing from the
  hard low-frequency cutoff. Set `--f-taper-width 0` for a hard cut.
- **Inverse spectrum truncation (`--psd-max-filter-length 4.0 s`).**
  The `1/sqrt(S_n)` whitening filter is iFFT'd to the time domain,
  windowed down to the chosen duration, and FFT'd back. Prevents the
  whitening kernel's long tail from wrapping around the segment via
  circular FFT and biasing the SNR. Standard trick from
  `pycbc.psd.inverse_spectrum_truncation`. Set 0 to disable.

On top of that, the template itself is evaluated on an internally-
longer grid (`--template-eval-factor`, default 4×) and then
iFFT'd → truncated to the segment → windowed → FFT'd back, the same
way `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator` avoids
circular aliasing. Without this step a ~100 s BNS chirp sampled on the
segment's `df = 1/T` grid has its early inspiral wrapping around the
segment; `⟨h|h⟩` then counts the wrapped energy while `⟨d|h⟩` does not,
which drops the recovered SNR by ~3× even at the correct chirp mass.
Bump the factor if the in-band chirp duration exceeds
`(factor - 1) × T`.

### Waveform template

`--waveform` selects which ripple model is used as the template:

- `imrphenomd` (default) — `gen_IMRPhenomD_hphc`. Matches
  `GWDatasetGeneration`'s BBH path (`ml4gw.waveforms.IMRPhenomD`).
- `taylorf2` — `gen_TaylorF2_hphc` with `use_lambda_tildes=False` and
  `lambda1 = lambda2 = 0`. Matches `GWDatasetGeneration`'s BNS path
  (`ml4gw.waveforms.TaylorF2`, point-particle, 3.5 PN phase). Using
  IMRPhenomD against a TaylorF2 BNS injection produces a large phase
  mismatch that keeps the optimiser stuck — switch to `taylorf2` for
  BNS data.

The fit parameter vector is the same `[Mc, eta, chi1, chi2, D, tc,
phic, inclination]` for both choices; the TaylorF2 path internally
inserts `lambda1 = lambda2 = 0` before calling ripple.

### Optimisation target

Two losses are available via `--loss`:

- `--loss snr` (default) — minimise `-ρ²_net` where, per detector,
  `ρ²_j = max_{|t| ≤ Δt} |z_j(t)|² / ⟨h|h⟩_j` with
  `z_j(t) = 4 ∫ conj(d̃_j) h̃_j e^{2πift} / S_n df`. `z_j(t)` is computed
  via one inverse FFT per detector, so the statistic is **time- and
  phase-maximised** just like `pycbc.filter.matched_filter`. The time
  maximisation matters: the generator applies per-detector light-travel
  delays (H1↔L1 up to ~10 ms) via `compute_observed_strain`, and a
  fixed-`tc` template would be dephased at ~200 Hz enough to kill the
  Mc gradient.

  The max is taken over a narrow circular window `|t| ≤ Δt`
  (`--max-time-shift`, default 50 ms) centred on the template's `tc`,
  not the full 64 s segment. Searching every sample lets `|z(t)|²`
  saturate its Cauchy–Schwarz ceiling `⟨d|d⟩` on a noise-aligned
  template — for a 2-IFO LIGO segment at 64 s duration that ceiling is
  ~500, so the optimiser would report ρ ≈ 500 regardless of the true
  signal SNR (~14). Restricting to ±50 ms covers every physical
  light-travel delay with margin while keeping the noise ceiling near
  the true-signal SNR.
- `--loss logl` — minimise `-(⟨d|h⟩ - ½⟨h|h⟩)`, the Gaussian
  matched-filter log-likelihood at fixed `tc`. Sensitive to distance
  and phase; only use when those are initialised near truth.

`--detector` picks which IFO(s) enter the sum. `network` (default)
sums `ρ²_H1 + ρ²_L1`; `H1` or `L1` restricts the statistic to a single
detector for diagnostics. Note the network sum is **incoherent** —
each detector's peak is chosen independently by `max_t |z_j(t)|²`, so
H1 and L1 are not constrained to share a common geocentric arrival
time (this matches coincident-search conventions, not coherent ones).
The plot's horizontal truth line follows the `--detector` choice
(`/truth/snr` for `network`, `/truth/snr_H1` or `/truth/snr_L1`
otherwise).

### Initialising non-mass parameters from truth

For sanity checks you can seed every non-mass parameter from the `/truth`
group written by `generate_fit_input.py`:

```
python fit_waveform.py --data data.h5 --init-from-truth \
    --mc-init 10.0 --steps 300
```

With the flag set, `--eta` is derived from `truth/mass_ratio`
(`eta = q / (1+q)^2`), and `chi1/chi2/dist/tc/phic/inclination` are
overridden from the matching truth datasets. Chirp mass is **not**
overridden — it still starts at `--mc-init` so there is something to fit.
This is simulation-only information; do not use it when benchmarking
against real strain.

## Output

- `fit_history.png` — scatter of network SNR vs chirp mass, colored by
  iteration index, showing the path the optimizer traced.
- Stdout: recovered chirp mass and final network SNR.

## Notes

- Uses `ripple.waveforms.IMRPhenomD.gen_IMRPhenomD_hphc`. For BNS with
  tidal effects swap in `IMRPhenomD_NRTidalv2` and extend `theta` with
  `lambda1, lambda2`.
- Matched-filter inner product uses `4 df Re Σ d*·h / S_n`, assuming a
  one-sided PSD and a uniform frequency grid.
- Antenna factors are treated as constants loaded from the data file;
  extending the fit to sky location would make them functions of
  `(ra, dec, psi, t_gps)`.
