from __future__ import annotations
import sys

sys.path.append("../PETSIRD/python")
import prd
import numpy as np


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


def read_yardl(prd_file: str) -> tuple[np.array[int], np.array[int], np.array[float]]:
    """Reads a yardl list mode file and returns two detector arrays and a scanner lookup table.

    Args:
        prd_file (str): File to read from.

    Returns:
        tuple[np.array[int], np.array[int], np.array[float]]: Two detector arrays and a scanner lookup table.
    """
    with prd.BinaryPrdExperimentReader(prd_file) as reader:
        # Read header and build lookup table
        header = reader.read_header()
        scanner_information = header.scanner
        detectors = scanner_information.detectors
        scanner_lut = np.zeros((len(detectors), 3))
        for i, detector in enumerate(detectors):
            scanner_lut[i, 0] = detector.x
            scanner_lut[i, 1] = detector.y
            scanner_lut[i, 2] = detector.z
        # Read events
        detector_hits = []
        for time_blocks in reader.read_time_blocks():
            for event in time_blocks.prompt_events:
                prompt = [event.detector_1_id, event.detector_2_id]
                detector_hits.append(prompt)
        det_1, det_2 = np.asarray(detector_hits).T
    return det_1, det_2, scanner_lut


#if __name__ == "__main__":
#    det_1 = np.asarray([0, 1, 2, 3, 4])
#    det_2 = np.asarray([5, 6, 7, 8, 9])
#    scanner_lut = np.random.rand(10, 3)
#    write_yardl(det_1, det_2, scanner_lut, output_file="test.yardl")
#    det_1_read, det_2_read, scanner_lut_read = read_yardl("test.yardl")
#    print(scanner_lut == scanner_lut_read)
#    print(np.isclose(scanner_lut, scanner_lut_read).all())
#    print(scanner_lut.dtype)
#    print(scanner_lut_read.dtype)
