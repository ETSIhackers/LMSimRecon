from __future__ import annotations
import sys

sys.path.append("../PETSIRD/python")
import prd

import numpy.array_api as np
from types import ModuleType


def write_prd_from_numpy_arrays(
    detector_1_id_array: np.array[int],
    detector_2_id_array: np.array[int],
    scanner_lut: np.array[float, float],
    tof_idx_array: np.array[int] | None = None,
    energy_1_idx_array: np.array[int] | None = None,
    energy_2_idx_array: np.array[int] | None = None,
    output_file: str | None = None,
    num_events: int | None = None,
) -> None:
    """Write a PRD file from numpy arrays. Currently all into one time block

    Parameters
    ----------
    detector_1_id_array : np.array[int]
        array containing the detector 1 id for each event
    detector_2_id_array : np.array[int]
        array containing the detector 2 id for each event
    scanner_lut : np.array[float, float]
        a 2D float array of size (num_det, 3) containing the world
        coordinates of each detector (in mm)
    tof_idx_array : np.array[int] | None, optional
        array containing the tof bin index of each event
    energy_1_idx_array : np.array[int] | None, optional
        array containing the energy 1 index of each event
    energy_2_idx_array : np.array[int] | None, optional
        array containing the energy 2 index of each event
    output_file : str | None, optional
        output file, if None write to stdout
    num_events : int | None, optional
        number of events to write, if None write all events
    """

    if num_events is None:
        num_events = detector_1_id_array.shape[0]

    detector_list = [
        prd.Detector(
            id=int(i), x=float(coords[0]), y=float(coords[1]), z=float(coords[2])
        )
        for i, coords in enumerate(scanner_lut)
    ]

    scanner_information = prd.ScannerInformation(detectors=detector_list)

    events = []
    for i in range(num_events):
        det_id_1 = int(detector_1_id_array[i])
        det_id_2 = int(detector_2_id_array[i])

        if tof_idx_array is not None:
            tof_idx = int(tof_idx_array[i])
        else:
            tof_idx = 0

        if energy_1_idx_array is not None:
            energy_1_idx = int(energy_1_idx_array[i])
        else:
            energy_1_idx = 0

        if energy_2_idx_array is not None:
            energy_2_idx = int(energy_2_idx_array[i])
        else:
            energy_2_idx = 0

        events.append(
            prd.CoincidenceEvent(
                detector_1_id=det_id_1,
                detector_2_id=det_id_2,
                tof_idx=tof_idx,
                energy_1_idx=energy_1_idx,
                energy_2_idx=energy_2_idx,
            )
        )

    time_block = prd.TimeBlock(id=0, prompt_events=events)

    if output_file is None:
        with prd.BinaryPrdExperimentWriter(sys.stdout.buffer) as writer:
            writer.write_header(prd.Header(scanner=scanner_information))
            writer.write_time_blocks((time_block,))
    else:
        if output_file.endswith(".ndjson"):
            with prd.NDJsonPrdExperimentWriter(output_file) as writer:
                writer.write_header(prd.Header(scanner=scanner_information))
                writer.write_time_blocks((time_block,))
        else:
            with prd.BinaryPrdExperimentWriter(output_file) as writer:
                writer.write_header(prd.Header(scanner=scanner_information))
                writer.write_time_blocks((time_block,))


def read_prd_to_numpy_arrays(
    prd_file: str,
    xp: ModuleType,
    dev: str,
    read_tof: bool | None = None,
    read_energy: bool | None = None,
) -> tuple[np.array[int], np.array[int], np.array[float]]:
    """Reads a yardl list mode file and returns two detector arrays and a scanner lookup table.

    Args:
        prd_file (str): File to read from.

    Returns:
        tuple[np.array[int], np.array[int], np.array[float]]: Two detector arrays and a scanner lookup table.
    """

    with prd.BinaryPrdExperimentReader(prd_file) as reader:
        # Read header and build lookup table
        header = reader.read_header()

        # bool that decides whether the scanner has TOF and whether it is
        # meaningful to read TOF
        if read_tof is None:
            read_tof: bool = len(header.scanner.tof_bin_edges) <= 1

        # bool that decides whether the scanner has energy and whether it is
        # meaningful to read energy
        if read_energy is None:
            read_energy: bool = len(header.scanner.energy_bin_edges) <= 1

        # read the detector coordinate look up table
        scanner_lut = xp.asarray(
            [[det.x, det.y, det.z] for det in header.scanner.detectors],
            dtype=xp.float32,
            device=dev,
        )

        # loop over all time blocks and read all meaningful event attributes
        for time_block in reader.read_time_blocks():
            if read_tof and read_energy:
                event_attribute_list = [
                    [
                        e.detector_1_id,
                        e.detector_2_id,
                        e.tof_idx,
                        e.energy_1_idx,
                        e.energy_2_idx,
                    ]
                    for e in time_block.prompt_events
                ]
            elif read_tof and (not read_energy):
                event_attribute_list = [
                    [
                        e.detector_1_id,
                        e.detector_2_id,
                        e.tof_idx,
                    ]
                    for e in time_block.prompt_events
                ]
            elif (not read_tof) and read_energy:
                event_attribute_list = [
                    [
                        e.detector_1_id,
                        e.detector_2_id,
                        e.energy_1_idx,
                        e.energy_2_idx,
                    ]
                    for e in time_block.prompt_events
                ]
            else:
                event_attribute_list = [
                    [
                        e.detector_1_id,
                        e.detector_2_id,
                    ]
                    for e in time_block.prompt_events
                ]

    return xp.asarray(event_attribute_list, device=dev), scanner_lut
