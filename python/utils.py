import array_api_compat.numpy as np
import parallelproj_utils
from numpy.array_api._array_object import Array
from types import ModuleType


def noisy_tof_sinogram_to_lm(
    noisy_sinogram: Array,
    lor_descriptor: parallelproj_utils.PETLORDescriptor,
    xp: ModuleType,
    dev: str,
) -> tuple[Array, Array, Array]:
    """convert a noisy sinogram to listmode data

    Parameters
    ----------
    noisy_sinogram : Array
        sinogram containing Poisson noise (integer values)
    lor_descriptor : parallelproj_utils.PETLORDescriptor
        description of the LOR geometry
    xp : ModuleType
        array module
    dev : str
        device

    Returns
    -------
    tuple[Array, Array, Array]
        event_det_id_1, event_det_id_2, event_tof_bin
    """
    if noisy_sinogram.ndim != 4:
        raise ValueError(
            f"noisy_sinogram must be a 4D array, the last axis must be the TOF axis, but has shape {noisy_sinogram.shape}"
        )

    num_tof_bins = noisy_sinogram.shape[3]

    # ravel the noisy sinogram and the detector start and end "index" sinograms
    noisy_sinogram = xp.reshape(noisy_sinogram, (noisy_sinogram.size,))

    # get the two dimensional indices of all sinogram bins
    start_mods, end_mods, start_inds, end_inds = lor_descriptor.get_lor_indices()

    # generate two sinograms that contain the linearized detector start and end indices
    sino_det_start_index = (
        lor_descriptor.scanner.num_lor_endpoints_per_module[0] * start_mods + start_inds
    )
    sino_det_end_index = (
        lor_descriptor.scanner.num_lor_endpoints_per_module[0] * end_mods + end_inds
    )

    sino_det_start_index = xp.reshape(
        xp.stack(num_tof_bins * [sino_det_start_index], axis=-1),
        sino_det_start_index.size * num_tof_bins,
    )

    sino_det_end_index = xp.reshape(
        xp.stack(num_tof_bins * [sino_det_end_index], axis=-1),
        sino_det_end_index.size * num_tof_bins,
    )

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # --- convert the index sinograms in to listmode data ------------------------
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------

    # generate listmode data from the noisy sinogram
    event_sino_inds = np.repeat(np.arange(noisy_sinogram.shape[0]), noisy_sinogram)
    # shuffle the event sinogram indices
    np.random.shuffle(event_sino_inds)
    # convert event sino indices to xp array
    event_sino_inds = xp.asarray(event_sino_inds, device=dev)

    event_det_id_1 = xp.take(sino_det_start_index, event_sino_inds)
    event_det_id_2 = xp.take(sino_det_end_index, event_sino_inds)
    event_tof_bin = event_sino_inds % num_tof_bins

    return event_det_id_1, event_det_id_2, event_tof_bin
