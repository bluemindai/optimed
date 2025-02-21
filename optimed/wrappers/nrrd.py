from optimed.wrappers.operations import exists, isfile
from typing import Tuple
import numpy as np
import nrrd
import slicerio
import random


# fmt: off
def load_nrrd(
    file: str,
    data_only: bool = False,
    header_only: bool = False
) -> Tuple[np.ndarray, dict]: # fmt: on
    """
    Load a NRRD image from the specified file.

    Parameters:
        file (str): Path to the NRRD file.
        data_only (bool): If True, returns only the data.
        header_only (bool): If True, returns only the header.

    Returns:
        tuple: (data, header) or one of them depending on the parameters.
    """
    assert exists(file), f"File does not exist: {file}"
    assert isfile(file), f"{file} is not a file"
    assert file.lower().endswith(".nrrd"), "Invalid file extension, expected '.nrrd'"

    try:
        data, header = nrrd.read(file)
    except Exception as e:
        raise ValueError(f"Error reading NRRD file: {str(e)}")

    if data_only:
        return data
    elif header_only:
        return header
    return data, header


def save_nrrd(
    ndarray: np.ndarray,
    header: dict = None,
    compression_level: int = 9,
    save_to: str = "image.nrrd",
) -> None:
    """
    Save a NRRD image.

    Parameters:
        ndarray (np.ndarray): Data to be saved.
        header (dict, optional): Custom header to be merged with the default.
        compression_level (int): Compression level (from 1 to 9).
        save_to (str): Path to save the file.

    Returns:
        None
    """
    assert isinstance(ndarray, np.ndarray), "ndarray must be a numpy.ndarray object."
    assert (
        isinstance(compression_level, int) and 1 <= compression_level <= 9
    ), "Compression level must be an integer between 1 and 9."
    assert save_to.lower().endswith(".nrrd"), "Invalid file extension, expected '.nrrd'"

    default_header = {
        "encoding": "gzip",
        "endian": "little",
        "dimension": 3,
        "type": str(ndarray.dtype),
        "space": "left-posterior-superior",
        "space directions": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
        "space origin": ["0", "0", "0"],
        "kinds": ["domain", "domain", "domain"],
        "space units": ["mm", "mm", "mm"],
        "space dimension": 3,
        "sizes": ndarray.shape,
    }

    final_header = default_header.copy()
    if header is not None:
        final_header.update(header)

    try:
        nrrd.write(
            file=save_to,
            data=ndarray,
            header=final_header,
            compression_level=compression_level,
        )
    except Exception as e:
        raise ValueError(f"Error saving NRRD file: {str(e)}")


def save_multilabel_nrrd_seg(
    ndarray: np.ndarray,
    segment_metadata: list = None,
    save_to: str = "segmentation.seg.nrrd",
) -> None:
    """
    Save a multilabel segmentation in NRRD format.

    Parameters:
        ndarray (np.ndarray): Data to be saved.
        segment_metadata (list): Metadata for each class.
        save_to (str): Path to save the file.

    Returns:
        None
    """
    assert isinstance(ndarray, np.ndarray), "ndarray must be a numpy.ndarray object."
    assert save_to.endswith(".seg.nrrd"), "Invalid file extension, expected '.seg.nrrd'"
    if segment_metadata:
        for seg in segment_metadata:
            if not isinstance(seg, dict):
                raise ValueError("Each segment metadata must be a dictionary.")
            if "labelValue" not in seg or "name" not in seg:
                raise ValueError(
                    "Each segment metadata dictionary must have 'labelValue' and 'name' keys."
                )
            if not isinstance(seg["labelValue"], int):
                raise ValueError(
                    "Each segment metadata 'labelValue' must be an integer."
                )
            if "color" in seg:
                if not isinstance(seg["color"], list):
                    raise ValueError(
                        "Segment metadata 'color' must be a list of 3 floats."
                    )
                if len(seg["color"]) != 3:
                    raise ValueError(
                        "Segment metadata 'color' must be a list of 3 floats."
                    )
                # Adapt color if values appear to be in the 0-255 range
                if any(c > 1 for c in seg["color"]):
                    seg["color"] = [round(c / 255.0, 2) for c in seg["color"]]

    def _random_color():
        """Generate a random RGB color with values between 0 and 1."""
        return [round(random.random(), 2) for _ in range(3)]

    for seg in segment_metadata:
        if "color" not in seg:
            seg["color"] = _random_color()

    # Assign metadata to the segmentation dictionary
    ndarray["segments"] = segment_metadata

    slicerio.write_segmentation(save_to, ndarray)
