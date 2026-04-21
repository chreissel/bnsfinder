# ripple_differential_fit

Differential fitting of a ripple gravitational-wave waveform to H1/L1
strain data. Chirp mass is recovered via gradient descent on the matched-
filter log-likelihood; network SNR is tracked along the trajectory.

## Install

```
pip install -r requirements.txt
```

`ripplegw` pulls in JAX. Install a matching `jax[cuda]` wheel first if you
want GPU acceleration.

## Input data

The script reads a single HDF5 file produced elsewhere (e.g. by an injection
pipeline) that contains both detectors' strain (signal + background) in the
frequency domain, their PSDs, and the antenna-pattern factors for the source
sky location and time.

```
/freqs                         float64   [Nf]    one-sided frequency grid [Hz]
/H1/strain                     complex   [Nf]    d(f) at H1
/H1/psd                        float64   [Nf]    one-sided PSD on /freqs
/L1/strain, /L1/psd            same as H1
/antenna/H1/fplus, /fcross     scalar    F+, Fx for H1
/antenna/L1/fplus, /fcross     scalar    F+, Fx for L1
attrs: f_ref                   scalar    reference frequency [Hz]
```

The frequency grid must be uniform. All arrays share the same length.

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
