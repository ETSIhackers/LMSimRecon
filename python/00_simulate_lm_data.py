from __future__ import annotations
import sys

sys.path.append("../PETSIRD/python")

import prd
import parallelproj
import parallelproj_utils
import utils
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
from array_api_compat import to_device
from prd_io import write_prd_from_numpy_arrays
from pathlib import Path

# ----------------------------------------------------------------
# -- Choose you favorite array backend and device here -----------
# ----------------------------------------------------------------

import numpy.array_api as xp

dev: str = "cpu"

# ----------------------------------------------------------------
# ----------------------------------------------------------------

output_dir: str = "../data/sim_LM_acq_1"
output_sens_img_file: str = "sensitivity_image.npz"
output_prd_file: str = "simulated_lm.prd"
expected_num_trues: float = 1e6

np.random.seed(42)

# create the output directory
Path(output_dir).mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --- setup the scanner / LOR geometry ---------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# setup a line of response descriptor that describes the LOR start / endpoints of
# a "narrow" clinical PET scanner with 9 rings
lor_descriptor = parallelproj_utils.DemoPETScannerLORDescriptor(
    xp, dev, num_rings=4, radial_trim=141
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --- setup a simple 3D test image -------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# image properties
voxel_size = (2.66, 2.66, 2.66)
num_trans = 128
num_ax = 2 * lor_descriptor.scanner.num_modules

# setup a box like test image
img_shape = (num_trans, num_trans, num_ax)
n0, n1, n2 = img_shape

# setup the image for the 1st time frame
img_f1 = xp.asarray(
    np.tile(np.load("../data/SL.npy")[..., None], (1, 1, num_ax)),
    device=dev,
    dtype=xp.float32,
)
img_f1[:, :, :2] = 0
img_f1[:, :, -2:] = 0

img_f1 /= xp.max(img_f1)

# setup the image for the 2nd time frame
img_f2 = xp.sqrt(img_f1)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --- setup a non-TOF projector and project ----------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

res_model = parallelproj.GaussianFilterOperator(
    img_shape, sigma=4.5 / (2.355 * xp.asarray(voxel_size))
)
projector = parallelproj_utils.RegularPolygonPETProjector(
    lor_descriptor, img_shape, voxel_size, resolution_model=res_model
)
projector.tof = True  # set this to True to get a time of flight projector

# repeat number of TOF bin times here
num_tof_bins = projector.tof_parameters.num_tofbins

# forward project the image
noise_free_sinogram_f1 = projector(img_f1)
noise_free_sinogram_f2 = projector(img_f2)

# rescale the forward projection and image such that we get the expected number of trues
scale = expected_num_trues / float(xp.sum(noise_free_sinogram_f1))
noise_free_sinogram_f1 *= scale
noise_free_sinogram_f2 *= scale
img_f1 *= scale
img_f2 *= scale

# calculate the sensitivity image
sens_img = projector.adjoint(
    xp.ones(noise_free_sinogram_f1.shape, device=dev, dtype=xp.float32)
)

# add poisson noise to the noise free sinogram
noisy_sinogram_f1 = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_sinogram_f1, "cpu"))), device=dev
)
noisy_sinogram_f2 = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_sinogram_f2, "cpu"))), device=dev
)

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

event_det_id_1_f1, event_det_id_2_f1, event_tof_bin_f1 = utils.noisy_tof_sinogram_to_lm(
    noisy_sinogram_f1, lor_descriptor, xp, dev
)
event_det_id_1_f2, event_det_id_2_f2, event_tof_bin_f2 = utils.noisy_tof_sinogram_to_lm(
    noisy_sinogram_f2, lor_descriptor, xp, dev
)
print(f"number of simulated events f1: {event_det_id_1_f1.shape[0]}")
print(f"number of simulated events f2: {event_det_id_1_f2.shape[0]}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ---- convert LM detector ID arrays into PRD here ---------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# get a lookup table that contains the world coordinates of all scanner detectors
# this is a 2D array of shape (num_detectors, 3)
scanner_lut = lor_descriptor.scanner.all_lor_endpoints

# generate a list of detector coordinates of our scanner
detector_list = []
for i in range(scanner_lut.shape[0]):
    detector_list.append(
        prd.Detector(
            id=int(i),
            x=float(scanner_lut[i, 0]),
            y=float(scanner_lut[i, 1]),
            z=float(scanner_lut[i, 2]),
        )
    )

# setup the edges of all TOF bins
tof_bin_edges = (
    xp.arange(num_tof_bins + 1, dtype=xp.float32) - ((num_tof_bins + 1) / 2 - 0.5)
) * projector.tof_parameters.tofbin_width

# setup the scanner information containing detector and TOF information
# WARNING: DEFINITION OF TOF RESOLUTION (sigma vs FWHM) not clear yet
scanner_information = prd.ScannerInformation(
    model_name="DummyPET",
    detectors=detector_list,
    tof_bin_edges=np.asarray(to_device(tof_bin_edges, "cpu")),
    tof_resolution=projector.tof_parameters.sigma_tof,
)

# write the data to PETSIRD
write_prd_from_numpy_arrays(
    [event_det_id_1_f1, event_det_id_1_f2],
    [event_det_id_2_f1, event_det_id_2_f2],
    scanner_information,
    tof_idx_array_blocks=[event_tof_bin_f1, event_tof_bin_f2],
    output_file=str(Path(output_dir) / output_prd_file),
)
print(f"wrote PETSIRD LM file to {str(Path(output_dir) / output_prd_file)}")

# HACK: write the sensitivity image to file
# this is currently needed since it is not agreed on how to store
# all valid detector pair combinations + attn / sens values in the PRD file
np.savez(
    Path(output_dir) / output_sens_img_file,
    sens_img=np.asarray(to_device(sens_img, "cpu")),
    voxel_size=np.asarray(voxel_size),
    img_origin=np.asarray(to_device(projector.img_origin, "cpu")),
)
print(f"wrote sensitivity image to {str(Path(output_dir) / output_sens_img_file)}")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

fig_dir = Path("../figs")
fig_dir.mkdir(exist_ok=True)

vmax = 1.2 * xp.max(img_f1)
fig, ax = plt.subplots(
    2, img_shape[2], figsize=(img_shape[2] * 2, 2 * 2), sharex=True, sharey=True
)
for i in range(img_shape[2]):
    ax[0, i].imshow(
        np.asarray(to_device(img_f1[:, :, i], "cpu")), vmin=0, vmax=vmax, cmap="Greys"
    )
    ax[0, i].set_title(f"sl {i+1}", fontsize="small")

    ax[1, i].imshow(
        np.asarray(to_device(img_f2[:, :, i], "cpu")), vmin=0, vmax=vmax, cmap="Greys"
    )

ax[0, 0].set_ylabel("ground truth frame 1", fontsize="small")
ax[1, 0].set_ylabel("ground truth frame 2", fontsize="small")

fig.tight_layout()
fig.savefig(fig_dir / "simulated_phantom.png")
