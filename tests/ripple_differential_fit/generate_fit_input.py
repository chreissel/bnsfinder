"""Generate fit-ready HDF5 input from GWDatasetGeneration.

Upstream `GWDatasetGeneration/main.py` writes *whitened* time series plus
sampled parameters per batch in `sig_{i}.h5` / `bkg_{i}.h5`. The
differential fit in `fit_waveform.py` instead needs, for a single event:

    - Raw (un-whitened) detector strain time series for H1 and L1
    - One-sided PSD per detector on the rFFT grid of that time series
    - Antenna-pattern factors (F+, Fx) that were applied to the injection

This script reuses the waveform generation path from `GWDatasetGeneration`
(`waveforms.generate_signals`) together with `ml4gw` directly for PSD
estimation and antenna-response extraction, and writes a single-event
HDF5 in the fit-expected layout. Truth parameters are preserved under
a `/truth` group for downstream comparison.

Run:

    python generate_fit_input.py \
        --config GWDatasetGeneration/configs/config_BBH.yaml \
        --data  /path/to/background_data \
        --out   data.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GW_DIR = HERE / "GWDatasetGeneration"
if not GW_DIR.is_dir():
    raise SystemExit(
        f"Submodule not found at {GW_DIR}. "
        "Run `git submodule update --init` first."
    )
sys.path.insert(0, str(GW_DIR))

from utils import load_config  # noqa: E402  (from submodule)
from waveforms import generate_signals  # noqa: E402  (from submodule)

from ml4gw.dataloading import Hdf5TimeSeriesDataset  # noqa: E402
from ml4gw.gw import compute_antenna_responses, get_ifo_geometry  # noqa: E402
from ml4gw.transforms import SpectralDensity  # noqa: E402


def _interp_psd(psd: np.ndarray, fs: float, n_target: int) -> np.ndarray:
    """Resample a one-sided PSD onto the length-n_target rFFT grid."""
    if psd.shape[-1] == n_target:
        return psd
    f_orig = np.linspace(0.0, fs / 2.0, psd.shape[-1])
    f_target = np.fft.rfftfreq(2 * (n_target - 1), d=1.0 / fs)
    return np.stack([np.interp(f_target, f_orig, row) for row in psd])


def generate(config_path: Path, data_dir: Path, out_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(str(config_path))

    # Force single-event generation regardless of the config's batch_size.
    config.general.batch_size = 1

    ifos = list(config.general.ifos)
    fs = int(config.general.sample_rate)
    f_min = float(config.general.f_min)
    f_ref = float(config.general.f_ref)
    kernel_length = float(config.general.waveform_duration)

    fduration = float(config.whiten.fduration)
    fftlength = float(config.whiten.fftlength)
    psd_length = float(config.whiten.psd_length)
    overlap = config.whiten.overlap
    average = config.whiten.average

    psd_size = int(psd_length * fs)
    kernel_size = int(kernel_length * fs)
    window_size = int((psd_length + fduration + kernel_length) * fs)

    # --- Background kernel + PSD (same path as injections.py, no whitening) ---
    fnames = list(Path(data_dir).iterdir())
    if not fnames:
        raise SystemExit(f"No background files in {data_dir}")
    loader = Hdf5TimeSeriesDataset(
        fnames=fnames,
        channels=ifos,
        kernel_size=window_size,
        batch_size=1,
        batches_per_epoch=1,
        coincident=False,
    )
    background = next(iter(loader)).to(device)  # [1, n_ifos, window_size]

    sd = SpectralDensity(
        sample_rate=fs, fftlength=fftlength, overlap=overlap, average=average
    ).to(device)
    psd = sd(background[..., :psd_size].double())  # [1, n_ifos, n_psd_freqs]
    kernel = background[..., psd_size:]            # [1, n_ifos, fduration+kernel]

    # --- Signal generation ---
    waveforms, params = generate_signals(config, device=device, save=False)
    # waveforms: [1, n_ifos, num_samples] projected (observed) strain.

    # Match injections.py padding: drop fduration/2 from each edge of kernel.
    pad = int(fduration / 2 * fs)
    injected = kernel.detach().clone()
    injected[:, :, pad:-pad] += waveforms[..., -kernel_size:]
    strain = injected[:, :, pad:pad + kernel_size]  # [1, n_ifos, kernel_size]

    # --- Antenna pattern factors (F+, Fx) for the sampled sky location ---
    tensors, _ = get_ifo_geometry(*ifos)
    theta = torch.pi / 2.0 - params["dec"]  # ml4gw zenith angle
    antenna = compute_antenna_responses(
        theta, params["psi"], params["phi"],
        tensors.to(device), modes=["plus", "cross"],
    )  # [1, 2, n_ifos]
    fplus = antenna[0, 0, :].detach().cpu().numpy()
    fcross = antenna[0, 1, :].detach().cpu().numpy()

    # --- PSD onto rFFT grid of the kernel ---
    n_freqs = kernel_size // 2 + 1
    psd_np = _interp_psd(psd[0].detach().cpu().numpy(), fs, n_freqs)

    strain_np = strain[0].detach().cpu().numpy()

    # --- Write fit-ready HDF5 ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["sample_rate"] = float(fs)
        h5.attrs["f_ref"] = f_ref
        h5.attrs["f_min"] = f_min
        h5.attrs["duration"] = kernel_length
        for i, det in enumerate(ifos):
            h5.create_dataset(f"{det}/strain", data=strain_np[i])
            h5.create_dataset(f"{det}/psd", data=psd_np[i])
            h5.create_dataset(f"antenna/{det}/fplus", data=float(fplus[i]))
            h5.create_dataset(f"antenna/{det}/fcross", data=float(fcross[i]))
        truth = h5.create_group("truth")
        for k, v in params.items():
            if hasattr(v, "detach"):
                arr = v.detach().cpu().numpy()
                truth.create_dataset(k, data=arr[0] if arr.ndim else arr)
            else:
                truth.create_dataset(k, data=np.asarray(v))

    print(f"Wrote {out_path}")
    if "chirp_mass" in params:
        print(f"  truth chirp_mass = {float(params['chirp_mass'][0]):.4f} Msun")
    if "snr" in params:
        print(f"  truth network SNR = {float(params['snr'][0]):.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="GWDatasetGeneration config yaml")
    p.add_argument("--data", required=True, type=Path,
                   help="Directory with background HDF5 files")
    p.add_argument("--out", default=Path("data.h5"), type=Path,
                   help="Output fit-ready HDF5 file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.config, args.data, args.out)


if __name__ == "__main__":
    main()
