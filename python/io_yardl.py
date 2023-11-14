from __future__ import annotations
import sys

sys.path.append("../PETSIRD/python")
import prd
import numpy as np


def write_yardl(
    det_1_array: np.array[int],
    det_2_array: np.array[int],
    scanner_lut: np.array[float],
    output_file: str | None = None,
) -> None:
    """Writes a yardl list mode file from two detector arrays and a scanner lookup table.

    Args:
        output_file (str): File to write to.
        det_1_array (np.array[int]): Indices corresponding to detector 1.
        det_2_array (np.array[int]): Indices corresponding to detector 2.
        scanner_lut (np.array[float]): Lookup table for detector world coordinates.
    """
    num_counts = len(det_1_array)
    # create list of detectors
    detectors = []
    for i in range(scanner_lut.shape[0]):
        detectors.append(
            prd.Detector(
                id=i, x=scanner_lut[i, 0], y=scanner_lut[i, 1], z=scanner_lut[i, 2]
            )
        )
    scanner_information = prd.ScannerInformation(detectors=detectors)
    events = []
    for i in range(num_counts):
        events.append(
            prd.CoincidenceEvent(
                detector_1_id=int(det_1_array[i]), detector_2_id=int(det_2_array[i])
            )
        )
    time_block = prd.TimeBlock(id=0, prompt_events=events)
    if output_file is None:
        with prd.BinaryPrdExperimentWriter(sys.stdout.buffer) as writer:
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


if __name__ == "__main__":
    det_1 = np.asarray([0, 1, 2, 3, 4])
    det_2 = np.asarray([5, 6, 7, 8, 9])
    scanner_lut = np.random.rand(10, 3)
    write_yardl(det_1, det_2, scanner_lut, output_file="test.yardl")
    det_1_read, det_2_read, scanner_lut_read = read_yardl("test.yardl")
    print(scanner_lut == scanner_lut_read)
    print(np.isclose(scanner_lut, scanner_lut_read).all())
    print(scanner_lut.dtype)
    print(scanner_lut_read.dtype)
