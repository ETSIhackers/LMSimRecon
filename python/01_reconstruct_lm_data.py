from __future__ import annotations

import parallelproj
from array_api_compat import to_device
import matplotlib.pyplot as plt
from prd_io import read_prd_to_numpy_arrays
from pathlib import Path

import array_api_compat.numpy as np

# ----------------------------------------------------------------
# -- Choose you favorite array backend and device here -i---------
# ----------------------------------------------------------------

import numpy.array_api as xp

dev = "cpu"

# ----------------------------------------------------------------
# ----------------------------------------------------------------

lm_data_dir: str = "../data/sim_LM_acq_1"
sens_img_file: str = "sensitivity_image.npz"
prd_file: str = "simulated_lm.prd"

num_iter: int = 2
num_subsets: int = 20

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# read the LM file header and all event attributes
header, event_attributes = read_prd_to_numpy_arrays(
    str(Path(lm_data_dir) / prd_file),
    xp,
    dev,
    read_tof=None,
    read_energy=False,
    time_block_ids=range(1, 2),
)

# read the detector coordinates into a 2D lookup table
scanner_lut = xp.asarray(
    [[det.x, det.y, det.z] for det in header.scanner.detectors],
    dtype=xp.float32,
    device=dev,
)

xstart = xp.take(scanner_lut, event_attributes[:, 0], axis=0)
xend = xp.take(scanner_lut, event_attributes[:, 1], axis=0)

# check if we have TOF data and generate the corresponding TOF parameters we need for the
# TOF joseph projector
if event_attributes.shape[1] == 3:
    tof = True
    event_tof_bin = event_attributes[:, 2]
    num_tof_bins = header.scanner.tof_bin_edges.shape[0] - 1
    tofbin_width = header.scanner.tof_bin_edges[1] - header.scanner.tof_bin_edges[0]
    sigma_tof = xp.asarray([header.scanner.tof_resolution], dtype=xp.float32)
    tofcenter_offset = xp.asarray([0], dtype=xp.float32)
    nsigmas = 3.0
    print(f"read {event_attributes.shape[0]} TOF events")
else:
    tof = False
    print(f"read {event_attributes.shape[0]} non-TOF events")

# HACK: write the sensitivity image to file
# this is currently needed since it is not agreed on how to store
# all valid detector pair combinations + attn / sens values in the PRD file
sens_img_data = np.load(Path(lm_data_dir) / sens_img_file)
sens_img = xp.asarray(sens_img_data["sens_img"], device=dev)
img_shape = sens_img.shape
voxel_size = xp.asarray(sens_img_data["voxel_size"], device=dev)
img_origin = xp.asarray(sens_img_data["img_origin"], device=dev)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ---- LM recon using the event detector IDs and the scanner LUT -------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

res_model = parallelproj.GaussianFilterOperator(
    img_shape, sigma=4.5 / (2.355 * xp.asarray(voxel_size))
)

recon = xp.ones(img_shape, dtype=xp.float32, device=dev)

for it in range(num_iter):
    for isub in range(num_subsets):
        print(f"it {(it+1):03} / ss {(isub+1):03}", end="\r")
        xs_sub = xstart[isub::num_subsets, :]
        xe_sub = xend[isub::num_subsets, :]

        recon_sm = res_model(recon)

        if tof:
            event_tof_bin_sub = event_tof_bin[isub::num_subsets] - num_tof_bins // 2
            exp = parallelproj.joseph3d_fwd_tof_lm(
                xs_sub,
                xe_sub,
                recon,
                img_origin,
                voxel_size,
                tofbin_width,
                sigma_tof,
                tofcenter_offset,
                nsigmas,
                event_tof_bin_sub,
            )

            ratio_back = parallelproj.joseph3d_back_tof_lm(
                xs_sub,
                xe_sub,
                img_shape,
                img_origin,
                voxel_size,
                1 / exp,
                tofbin_width,
                sigma_tof,
                tofcenter_offset,
                nsigmas,
                event_tof_bin_sub,
            )
        else:
            exp = parallelproj.joseph3d_fwd(
                xs_sub, xe_sub, recon_sm, img_origin, voxel_size
            )
            ratio_back = parallelproj.joseph3d_back(
                xs_sub, xe_sub, img_shape, img_origin, voxel_size, 1 / exp
            )

        ratio_back_sm = res_model.adjoint(ratio_back)

        recon *= ratio_back_sm / (sens_img / num_subsets)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

fig_dir = Path("../figs")
fig_dir.mkdir(exist_ok=True)

vmax = 0.055
fig, ax = plt.subplots(1, recon.shape[2], figsize=(recon.shape[2] * 2, 2))
for i in range(recon.shape[2]):
    ax[i].imshow(
        np.asarray(to_device(recon[:, :, i], "cpu")), vmin=0, vmax=vmax, cmap="Greys"
    )
    ax[i].set_title(f"LM recon sl {i+1}", fontsize="small")

fig.tight_layout()
fig.savefig(fig_dir / "lm_reconstruction.png")
