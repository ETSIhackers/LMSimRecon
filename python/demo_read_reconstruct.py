from __future__ import annotations

import parallelproj
import array_api_compat.numpy as np
from array_api_compat import to_device
import matplotlib.pyplot as plt
from io_yardl import read_yardl


dev = "cpu"

# image properties
voxel_size = (2.66, 2.66, 2.66)
img_shape = (128, 128, 8)
img_origin = [-168.91, -168.91, -9.31]

num_iter = 2
num_subsets = 20

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

event_det_id_1, event_det_id_2, scanner_lut = read_yardl("write_test.prd")

xstart = scanner_lut[event_det_id_1, :]
xend = scanner_lut[event_det_id_2, :]

# HACK: load the sensitivity image
sens_img = np.load("sensitivity_image.npy")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ---- LM recon using the event detector IDs and the scanner LUT -------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

res_model = parallelproj.GaussianFilterOperator(
    img_shape, sigma=4.5 / (2.355 * np.asarray(voxel_size))
)

recon = np.ones(img_shape, dtype=np.float32, device=dev)

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

vmax = 0.055
fig, ax = plt.subplots(1, recon.shape[2], figsize=(recon.shape[2] * 2, 2))
for i in range(recon.shape[2]):
    ax[i].imshow(
        np.asarray(to_device(recon[:, :, i], "cpu")), vmin=0, vmax=vmax, cmap="Greys"
    )
    ax[i].set_title(f"ground truth sl {i+1}", fontsize="small")

fig.tight_layout()
fig.savefig("lm_reconstruction.png")
