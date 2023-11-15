from __future__ import annotations

import parallelproj
from array_api_compat import to_device
import matplotlib.pyplot as plt
from io_yardl import read_prd_to_numpy_arrays
from pathlib import Path

import array_api_compat.numpy as np
import numpy.array_api as xp

dev = "cpu"
lm_data_dir: str = "../data/sim_LM_acq_1"
sens_img_file: str = "sensitivity_image.npy"
prd_file: str = "simulated_lm.prd"

num_iter: int = 2
num_subsets: int = 20

# hard coded input parameters
voxel_size = (2.66, 2.66, 2.66)
img_shape = (128, 128, 8)
img_origin = [-168.91, -168.91, -9.31]

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# TODO: used array structures with named columns

event_attributes, scanner_lut = read_prd_to_numpy_arrays(
    str(Path(lm_data_dir) / prd_file), xp, dev, read_tof=False, read_energy=False
)

xstart = xp.take(scanner_lut, event_attributes[:, 0], axis=0)
xend = xp.take(scanner_lut, event_attributes[:, 1], axis=0)

# HACK: write the sensitivity image to file
# this is currently needed since it is not agreed on how to store
# all valid detector pair combinations + attn / sens values in the PRD file
sens_img = xp.asarray(np.load(Path(lm_data_dir) / sens_img_file), device=dev)

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
        xp.asarray(to_device(recon[:, :, i], "cpu")), vmin=0, vmax=vmax, cmap="Greys"
    )
    ax[i].set_title(f"LM recon sl {i+1}", fontsize="small")

fig.tight_layout()
fig.savefig(fig_dir / "lm_reconstruction.png")
