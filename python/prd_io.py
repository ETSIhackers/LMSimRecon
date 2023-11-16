from __future__ import annotations
import sys

sys.path.append("../PETSIRD/python")
import prd

from numpy.array_api._array_object import Array
from types import ModuleType


def write_prd_from_numpy_arrays(
    detector_1_id_array: Array,
    detector_2_id_array: Array,
    scanner_information: prd.ScannerInformation,
    tof_idx_array: Array | None = None,
    energy_1_idx_array: Array | None = None,
    energy_2_idx_array: Array | None = None,
    output_file: str | None = None,
) -> None:
    """Write a PRD file from numpy arrays. Currently all into one time block

    Parameters
    ----------
    detector_1_id_array : Array
        array containing the detector 1 id for each event
    detector_2_id_array : Array
        array containing the detector 2 id for each event
    scanner_information : prd.ScannerInformation
        description of the scanner according to PETSIRD
        (e.g. including all detector coordinates)
    tof_idx_array : Array | None, optional
        array containing the tof bin index of each event
    energy_1_idx_array : Array | None, optional
        array containing the energy 1 index of each event
    energy_2_idx_array : Array | None, optional
        array containing the energy 2 index of each event
    output_file : str | None, optional
        output file, if None write to stdout
    """

    num_events: int = detector_1_id_array.size

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
) -> tuple[prd.types.Header, Array]:
    """Read all time blocks of a PETSIRD listmode file

    Parameters
    ----------
    prd_file : str
        the PETSIRD listmode file
    xp : ModuleType
        the array backend module
    dev : str
        device used for the returned arrays
    read_tof : bool | None, optional
        read the TOF bin information of every event
        default None means that is is auto determined
        based on the scanner information (length of tof bin edges)
    read_energy : bool | None, optional
        read the energy information of every event
        default None means that is is auto determined
        based on the scanner information (length of energy bin edges)

    Returns
    -------
    tuple[prd.types.Header, Array]
        PRD listmode file header, 2D array containing all event attributes
    """
    with prd.BinaryPrdExperimentReader(prd_file) as reader:
        # Read header and build lookup table
        header = reader.read_header()

        # bool that decides whether the scanner has TOF and whether it is
        # meaningful to read TOF
        if read_tof is None:
            r_tof: bool = len(header.scanner.tof_bin_edges) > 1
        else:
            r_tof = read_tof

        # bool that decides whether the scanner has energy and whether it is
        # meaningful to read energy
        if read_energy is None:
            r_energy: bool = len(header.scanner.energy_bin_edges) > 1
        else:
            r_energy = read_energy

        # loop over all time blocks and read all meaningful event attributes
        for time_block in reader.read_time_blocks():
            if r_tof and r_energy:
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
            elif r_tof and (not r_energy):
                event_attribute_list = [
                    [
                        e.detector_1_id,
                        e.detector_2_id,
                        e.tof_idx,
                    ]
                    for e in time_block.prompt_events
                ]
            elif (not r_tof) and r_energy:
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

    return header, xp.asarray(event_attribute_list, device=dev)
