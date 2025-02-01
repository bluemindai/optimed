from typing import Tuple
import numpy as np

# Try to import cupy and its GPU-enabled scipy.ndimage functions.
try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as scipy_label_gpu
    from cupyx.scipy.ndimage import binary_dilation as scipy_binary_dilation_gpu
    from cupyx.scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt_gpu
    from cupyx.scipy.ndimage import minimum as scipy_minimum_gpu
    from cupyx.scipy.ndimage import sum as scipy_sum_gpu
    xp = cp
except ImportError as exc:
    # Fallback to CPU versions
    import traceback as tb
    from scipy.ndimage import label as scipy_label_cpu
    from scipy.ndimage import binary_dilation as scipy_binary_dilation_cpu
    from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt_cpu
    from scipy.ndimage import minimum as scipy_minimum_cpu
    from scipy.ndimage import sum as scipy_sum_cpu
    xp = np

import warnings
import importlib

# Check if cupy is available
cupy_available = importlib.util.find_spec("cupy") is not None


def _ensure_numpy(array):
    """
    Helper function that converts a Cupy array to a NumPy array.
    If the input is already a NumPy array, it is returned unchanged.
    """
    if hasattr(array, "get"):
        return array.get()
    return array


def scipy_label(input: np.ndarray, structure: np.ndarray = None, use_gpu: bool = True) -> Tuple[np.ndarray, int]:
    """
    Labels connected components in the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        Tuple[np.ndarray, int]: (labeled array, number of labels)
    """
    if not use_gpu or not cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        return scipy_label_cpu(input, structure)
    else:
        components, num_labels = scipy_label_gpu(input, structure)
        return _ensure_numpy(components), num_labels


def scipy_binary_dilation(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, use_gpu: bool = True) -> np.ndarray:
    """
    Applies binary dilation to the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The dilated array.
    """
    if not use_gpu or not cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        return scipy_binary_dilation_cpu(input, structure, iterations)
    else:
        result = scipy_binary_dilation_gpu(input, structure, iterations)
        return _ensure_numpy(result)


def scipy_distance_transform_edt(input: np.ndarray, sampling: Tuple[float, float] = (1, 1), use_gpu: bool = True) -> np.ndarray:
    """
    Computes the Euclidean distance transform of the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The distance-transformed array.
    """
    if not use_gpu or not cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        return scipy_distance_transform_edt_cpu(input, sampling)
    else:
        result = scipy_distance_transform_edt_gpu(input, sampling)
        return _ensure_numpy(result)


def scipy_minimum(input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True) -> np.ndarray:
    """
    Finds the minimum value of 'input' within the regions defined by 'labels' for a given index.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The minimum value (or array) for the given label.
    """
    if not use_gpu or not cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        return scipy_minimum_cpu(input, labels, index)
    else:
        result = scipy_minimum_gpu(input, labels, index)
        return _ensure_numpy(result)


def scipy_sum(input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True) -> np.ndarray:
    """
    Computes the sum of the input values within the regions defined by 'labels' for a given index.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The computed sum.
    """
    if not use_gpu or not cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        return scipy_sum_cpu(input, labels, index)
    else:
        result = scipy_sum_gpu(input, labels, index)
        return _ensure_numpy(result)


def filter_mask(mask: np.ndarray, lbls: list, use_gpu: bool = True, verbose: bool = True) -> np.ndarray:
    """
    Retains only the pixels in 'mask' whose values are in 'lbls',
    replacing all other values with 0.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The filtered mask (always a NumPy array).
    """
    # Set the array module (cupy or numpy) based on the use_gpu flag.
    if not use_gpu or not cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        xp = np
    else:
        xp = cp

    # Ensure 'mask' is an array in the proper module.
    mask = xp.asarray(mask)

    # Compute the lookup table (LUT).
    max_val = int(mask.max())
    lut = xp.zeros(max_val + 1, dtype=mask.dtype)
    for l in lbls:
        if l <= max_val:
            lut[l] = l
    # Apply the LUT to the mask.
    filtered_mask = lut[mask]

    # Ensure the result is returned as a NumPy array.
    return _ensure_numpy(filtered_mask)
