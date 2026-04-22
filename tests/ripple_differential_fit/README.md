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
output HDF5 for comparison against the fit.

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
```

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

- Uses `ripple.waveforms.IMRPhenomD.gen_IMRPhenomD_polar`. For BNS with
  tidal effects swap in `IMRPhenomD_NRTidalv2` and extend `theta` with
  `lambda1, lambda2`.
- Matched-filter inner product uses `4 df Re Σ d*·h / S_n`, assuming a
  one-sided PSD and a uniform frequency grid.
- Antenna factors are treated as constants loaded from the data file;
  extending the fit to sky location would make them functions of
  `(ra, dec, psi, t_gps)`.
