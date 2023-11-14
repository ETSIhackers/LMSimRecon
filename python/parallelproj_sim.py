from __future__ import annotations

import utils
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
from array_api_compat import to_device

# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu

dev = "cpu"
expected_num_trues = 1e6
np.random.seed(1)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --- setup the scanner / LOR geometry ---------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# setup a line of response descriptor that describes the LOR start / endpoints of
# a "narrow" clinical PET scanner with 9 rings
lor_descriptor = utils.DemoPETScannerLORDescriptor(
    np, dev, num_rings=2, radial_trim=141
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --- setup a simple 3D test image -------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# image properties
voxel_size = (2.66, 2.66, 2.66)
num_trans = 140
num_ax = 2 * lor_descriptor.scanner.num_modules

# setup a box like test image
img_shape = (num_trans, num_trans, num_ax)
n0, n1, n2 = img_shape

# setup an image containing a box
img = np.zeros(img_shape, dtype=np.float32, device=dev)
img[(n0 // 4) : (3 * n0 // 4), (n1 // 4) : (3 * n1 // 4), :] = 1

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# --- setup a non-TOF projector and project ----------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

projector = utils.RegularPolygonPETProjector(lor_descriptor, img_shape, voxel_size)
projector.tof = False  # set this to True to get a time of flight projector

# forward project the image
noise_free_sinogram = projector(img)

# rescale the forward projection and image such that we get the expected number of trues
scale = expected_num_trues / np.sum(noise_free_sinogram)
noise_free_sinogram *= scale
img *= scale

# calculate the sensitivity image
sens_img = projector.adjoint(np.ones(noise_free_sinogram.shape, device=dev, dtype=np.float32))

## get the start and end points of all sinogram bins
#xstart, xend = lor_descriptor.get_lor_coordinates()

# get the two dimensional indices of all sinogram bins
start_mods, end_mods, start_inds, end_inds = lor_descriptor.get_lor_indices()

# convert the 2D indices to a 1D index
det_start_index = lor_descriptor.scanner.num_modules * start_mods + start_inds
det_end_index = lor_descriptor.scanner.num_modules * end_mods + end_inds

# add poisson noise to the noise free sinogram
noisy_sinogram = np.random.poisson(noise_free_sinogram)

# ravel the noisy sinogram and the detector start and end "index" sinograms 
noisy_sinogram = noisy_sinogram.ravel()
det_start_index = det_start_index.ravel()
det_end_index = det_end_index.ravel()

# generate listmode data from the noisy sinogram
inds = np.repeat(np.arange(noisy_sinogram.shape[0]), noisy_sinogram)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# show the scanner geometry and one view in one sinogram plane
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
lor_descriptor.scanner.show_lor_endpoints(ax)
lor_descriptor.show_views(
    ax,
    views=np.asarray([lor_descriptor.num_views // 4], device=dev),
    planes=np.asarray([lor_descriptor.scanner.num_modules // 2], device=dev),
)
fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
ax2[0].imshow(np.asarray(to_device(img[:, :, 3], "cpu")))
if projector.tof:
    ax2[1].imshow(np.asarray(to_device(noise_free_sinogram[:, :, 0, 15], "cpu")))
else:
    ax2[1].imshow(np.asarray(to_device(noise_free_sinogram[:, :, 0], "cpu")))
ax2[2].imshow(np.asarray(to_device(sens_img[:, :, 3], "cpu")))
fig2.tight_layout()
fig2.show()
