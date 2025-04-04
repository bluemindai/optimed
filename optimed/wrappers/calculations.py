from typing import Tuple
import numpy as np
import warnings

from optimed import _cupy_available

from scipy.ndimage import label as _scipy_label_cpu
from scipy.ndimage import binary_dilation as _scipy_binary_dilation_cpu
from scipy.ndimage import binary_closing as _scipy_binary_closing_cpu
from scipy.ndimage import binary_erosion as _scipy_binary_erosion_cpu
from scipy.ndimage import binary_opening as _scipy_binary_opening_cpu
from scipy.ndimage import binary_fill_holes as _scipy_binary_fill_holes_cpu
from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt_cpu
from scipy.ndimage import median_filter as _scipy_median_filter_cpu
from scipy.ndimage import minimum as _scipy_minimum_cpu
from scipy.ndimage import sum as _scipy_sum_cpu
from scipy.ndimage import center_of_mass as _scipy_center_of_mass_cpu

# Try to import cupy and its GPU-enabled scipy.ndimage functions.
try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as _scipy_label_gpu
    from cupyx.scipy.ndimage import binary_dilation as _scipy_binary_dilation_gpu
    from cupyx.scipy.ndimage import binary_closing as _scipy_binary_closing_gpu
    from cupyx.scipy.ndimage import binary_erosion as _scipy_binary_erosion_gpu
    from cupyx.scipy.ndimage import binary_opening as _scipy_binary_opening_gpu
    from cupyx.scipy.ndimage import binary_fill_holes as _scipy_binary_fill_holes_gpu
    from cupyx.scipy.ndimage import (
        distance_transform_edt as _scipy_distance_transform_edt_gpu,
    )
    from cupyx.scipy.ndimage import median_filter as _scipy_median_filter_gpu
    from cupyx.scipy.ndimage import minimum as _scipy_minimum_gpu
    from cupyx.scipy.ndimage import sum as _scipy_sum_gpu
    from cupyx.scipy.ndimage import center_of_mass as _scipy_center_of_mass_gpu
except ImportError:
    # Fallback to CPU versions
    _cupy_available = None  # noqa


def _ensure_numpy(array):
    """
    Helper function that converts a Cupy array to a NumPy array.
    If the input is already a NumPy array, it is returned unchanged.
    """
    if hasattr(array, "get"):
        return array.get()
    return array


def scipy_label(
    input: np.ndarray, structure: np.ndarray = None, use_gpu: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Labels connected components in the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        structure (np.ndarray): The structuring element.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        Tuple[np.ndarray, int]: (labeled array, number of labels)
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_label_cpu(input, structure)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        components, num_labels = _scipy_label_gpu(input, structure)
        return _ensure_numpy(components), num_labels


def scipy_binary_dilation(
    input: np.ndarray,
    structure: np.ndarray = None,
    iterations: int = 1,
    brute_force: bool = False,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Applies binary dilation to the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        structure (np.ndarray): The structuring element.
        iterations (int): The number of iterations.
        brute_force (bool): If True, uses a brute-force method.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The dilated array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_dilation_cpu(
            input, structure, iterations, brute_force=brute_force
        )
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_dilation_gpu(
            input, structure, iterations, brute_force=brute_force
        )
        return _ensure_numpy(result)


def scipy_binary_closing(
    input: np.ndarray,
    structure: np.ndarray = None,
    iterations: int = 1,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Applies binary closing to the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        structure (np.ndarray): The structuring element.
        iterations (int): The number of iterations.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The closed array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_closing_cpu(input, structure, iterations)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_closing_gpu(input, structure, iterations)
        return _ensure_numpy(result)


def scipy_binary_erosion(
    input: np.ndarray,
    structure: np.ndarray = None,
    iterations: int = 1,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Applies binary erosion to the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        structure (np.ndarray): The structuring element.
        iterations (int): The number of iterations.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The eroded array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_erosion_cpu(input, structure, iterations)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_erosion_gpu(input, structure, iterations)
        return _ensure_numpy(result)


def scipy_distance_transform_edt(
    input: np.ndarray, sampling: Tuple[float, float] = (1, 1), use_gpu: bool = True
) -> np.ndarray:
    """
    Computes the Euclidean distance transform of the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        sampling (Tuple[float, float]): The pixel spacing in the x and y directions.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The distance-transformed array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        return _scipy_distance_transform_edt_cpu(input, sampling)
    else:
        input = cp.asarray(input)
        result = _scipy_distance_transform_edt_gpu(input, sampling)
        return _ensure_numpy(result)


def scipy_binary_opening(
    input: np.ndarray,
    structure: np.ndarray = None,
    iterations: int = 1,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Applies binary opening to the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        structure (np.ndarray): The structuring element.
        iterations (int): The number of iterations.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The opened array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_opening_cpu(input, structure, iterations)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_opening_gpu(input, structure, iterations)
        return _ensure_numpy(result)


def scipy_binary_fill_holes(
    input: np.ndarray, structure: np.ndarray = None, use_gpu: bool = True
) -> np.ndarray:
    """
    Fills holes in the binary input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        structure (np.ndarray): The structuring element.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The filled array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_fill_holes_cpu(input, structure)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_fill_holes_gpu(input, structure)
        return _ensure_numpy(result)


def scipy_median_filter(
    input: np.ndarray,
    size: Tuple[int, int] = (3, 3),
    footprint: np.ndarray = None,
    mode: str = "reflect",
    cval: float = 0.0,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Applies a median filter to the input array.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        size (Tuple[int, int]): The size of the filter.
        footprint (np.ndarray): The structuring element.
        mode (str): The mode for handling borders.
        cval (float): The value to use for constant mode.
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The filtered array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        footprint = np.asarray(footprint) if footprint is not None else None
        return _scipy_median_filter_cpu(input, size, footprint, None, mode, cval)
    else:
        input = cp.asarray(input)
        footprint = cp.asarray(footprint) if footprint is not None else None
        result = _scipy_median_filter_gpu(input, size, footprint, None, mode, cval)
        return _ensure_numpy(result)


def scipy_minimum(
    input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True
) -> np.ndarray:
    """
    Finds the minimum value of 'input' within the regions defined by 'labels' for a given index.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        labels (np.ndarray): The labels array.
        index (int): The index of the label to compute the minimum for.
        use_gpu (bool): If True, uses GPU acceleration

    Returns:
        np.ndarray: The minimum value (or array) for the given label.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        labels = np.asarray(labels)
        return _scipy_minimum_cpu(input, labels, index)
    else:
        input = cp.asarray(input)
        labels = cp.asarray(labels)
        result = _scipy_minimum_gpu(input, labels, index)
        return _ensure_numpy(result)


def scipy_sum(
    input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True
) -> np.ndarray:
    """
    Computes the sum of the input values within the regions defined by 'labels' for a given index.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        labels (np.ndarray): The labels array.
        index (int): The index of the label to compute the sum for.
        use_gpu (bool): If True, uses GPU acceleration

    Returns:
        np.ndarray: The computed sum.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        labels = np.asarray(labels)
        return _scipy_sum_cpu(input, labels, index)
    else:
        input = cp.asarray(input)
        labels = cp.asarray(labels)
        result = _scipy_sum_gpu(input, labels, index)
        return _ensure_numpy(result)


def filter_mask(
    mask: np.ndarray,
    labels: list,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Retains only the pixels in 'mask' whose values are in 'lbls',
    replacing all other values with 0.
    Uses GPU acceleration if available and requested.

    Parameters:
        mask (np.ndarray): The mask image.
        labels (list): List of labels
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The filtered mask (always a NumPy array).
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        xp_local = np
    else:
        xp_local = cp

    mask = xp_local.asarray(mask)
    max_val = int(mask.max())
    lut = xp_local.zeros(max_val + 1, dtype=mask.dtype)
    for lbl in labels:
        if lbl <= max_val:
            lut[lbl] = lbl
    filtered_mask = lut[mask]

    return _ensure_numpy(filtered_mask)


def scipy_center_of_mass(
    input: np.ndarray,
    labels: np.ndarray = None,
    index: int = None,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Computes the center of mass of the input array, optionally for a single label or multiple labels.
    Uses GPU acceleration if available and requested.

    Parameters:
        input (np.ndarray): The input array.
        labels (np.ndarray): The labels array (optional).
        index (int): The index or array of indices to compute the center of mass for (optional).
        use_gpu (bool): If True, uses GPU acceleration.

    Returns:
        np.ndarray: The computed center of mass (one or more tuples).
    """
    if not use_gpu or not _cupy_available:
        warnings.warn(
            "GPU is not available or not requested. Using CPU for calculations."
        )
        input = np.asarray(input)
        labels = np.asarray(labels) if labels is not None else None
        return _scipy_center_of_mass_cpu(input, labels, index)
    else:
        input = cp.asarray(input)
        labels = cp.asarray(labels) if labels is not None else None
        result = _scipy_center_of_mass_gpu(input, labels, index)
        return _ensure_numpy(result)
