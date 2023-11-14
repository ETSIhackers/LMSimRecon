# TODO: - absolute scale of recon (maybe with sinogram recon)
#      - additive MLEM

from __future__ import annotations

import parallelproj
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
from io_yardl import read_yardl


dev = "cpu"

# image properties
voxel_size = (2.66, 2.66, 2.66)
img_shape = (128, 128, 4)
img_origin = [-168.91, -168.91, -3.9900002]

event_det_id_1, event_det_id_2, scanner_lut = read_yardl("write_test.yardl")

# hack until we have the reader / writer implemented

xstart = scanner_lut[event_det_id_1, :]
xend = scanner_lut[event_det_id_2, :]

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ---- LM recon using the event detector IDs and the scanner LUT -------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

recon = np.ones(img_shape, dtype=np.float32, device=dev)
img = np.tile(np.load("../data/SL.npy")[..., None], (1, 1, 4))
num_iter = 2
num_subsets = 20
for it in range(num_iter):
    for isub in range(num_subsets):
        print(f"it {(it+1):03} / ss {(isub+1):03}", end="\r")
        xs_sub = xstart[isub::num_subsets, :]
        xe_sub = xend[isub::num_subsets, :]
        exp = parallelproj.joseph3d_fwd(xs_sub, xe_sub, recon, img_origin, voxel_size)
        tmp = parallelproj.joseph3d_back(
            xs_sub, xe_sub, img_shape, img_origin, voxel_size, 1 / exp
        )
        recon *= tmp

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
fig.colorbar(ax[0].imshow(img[:, :, 2]))
fig.colorbar(ax[1].imshow(recon[:, :, 2]))
# difference
fig.colorbar(ax[2].imshow(recon[:, :, 2] - img[:, :, 2]))
fig.savefig("read_reconstruct_test.png")
