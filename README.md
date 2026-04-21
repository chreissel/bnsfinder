# bnsfinder — test repository

Collection of small, self-contained test subprojects. Each subdirectory under
`tests/` is an independent experiment with its own README and requirements.

## Tests

| Directory                            | Purpose                                                      |
| ------------------------------------ | ------------------------------------------------------------ |
| `tests/ripple_differential_fit/`     | Differential fitting of ripple waveforms to H1/L1 strain.    |

More tests will be added as new subdirectories.

## Submodules

Clone with `--recurse-submodules`, or after a plain clone run:

```
git submodule update --init
```

Current submodules:

- `tests/ripple_differential_fit/GWDatasetGeneration` — dataset generator
  used to produce fit input for the ripple differential-fit test.
