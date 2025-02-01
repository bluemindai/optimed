from typing import Tuple
import numpy as np
import warnings
import importlib

# Check if cupy is available
_cupy_available = importlib.util.find_spec("cupy") is not None

from scipy.ndimage import label as _scipy_label_cpu
from scipy.ndimage import binary_dilation as _scipy_binary_dilation_cpu
from scipy.ndimage import binary_closing as _scipy_binary_closing_cpu
from scipy.ndimage import binary_erosion as _scipy_binary_erosion_cpu
from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt_cpu
from scipy.ndimage import minimum as _scipy_minimum_cpu
from scipy.ndimage import sum as _scipy_sum_cpu

# Try to import cupy and its GPU-enabled scipy.ndimage functions.
try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as _scipy_label_gpu
    from cupyx.scipy.ndimage import binary_dilation as _scipy_binary_dilation_gpu
    from cupyx.scipy.ndimage import binary_closing as _scipy_binary_closing_gpu
    from cupyx.scipy.ndimage import binary_erosion as _scipy_binary_erosion_gpu
    from cupyx.scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt_gpu
    from cupyx.scipy.ndimage import minimum as _scipy_minimum_gpu
    from cupyx.scipy.ndimage import sum as _scipy_sum_gpu
except ImportError as exc:
    # Fallback to CPU versions
    _cupy_available = None

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
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
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
        use_gpu: bool = True) -> np.ndarray:
    """
    Applies binary dilation to the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The dilated array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_dilation_cpu(input, structure, iterations, brute_force=brute_force)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_dilation_gpu(input, structure, iterations, brute_force=brute_force)
        return _ensure_numpy(result)
    

def scipy_binary_closing(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, use_gpu: bool = True) -> np.ndarray:
    """
    Applies binary closing to the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The closed array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_closing_cpu(input, structure, iterations)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_closing_gpu(input, structure, iterations)
        return _ensure_numpy(result)
    

def scipy_binary_erosion(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, use_gpu: bool = True) -> np.ndarray:
    """
    Applies binary erosion to the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The eroded array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        input = np.asarray(input)
        structure = np.asarray(structure) if structure is not None else None
        return _scipy_binary_erosion_cpu(input, structure, iterations)
    else:
        input = cp.asarray(input)
        structure = cp.asarray(structure) if structure is not None else None
        result = _scipy_binary_erosion_gpu(input, structure, iterations)
        return _ensure_numpy(result)


def scipy_distance_transform_edt(input: np.ndarray, sampling: Tuple[float, float] = (1, 1), use_gpu: bool = True) -> np.ndarray:
    """
    Computes the Euclidean distance transform of the input array.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The distance-transformed array.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        input = np.asarray(input)
        return _scipy_distance_transform_edt_cpu(input, sampling)
    else:
        input = cp.asarray(input)
        result = _scipy_distance_transform_edt_gpu(input, sampling)
        return _ensure_numpy(result)


def scipy_minimum(input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True) -> np.ndarray:
    """
    Finds the minimum value of 'input' within the regions defined by 'labels' for a given index.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The minimum value (or array) for the given label.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        input = np.asarray(input)
        labels = np.asarray(labels)
        return _scipy_minimum_cpu(input, labels, index)
    else:
        input = cp.asarray(input)
        labels = cp.asarray(labels)
        result = _scipy_minimum_gpu(input, labels, index)
        return _ensure_numpy(result)


def scipy_sum(input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True) -> np.ndarray:
    """
    Computes the sum of the input values within the regions defined by 'labels' for a given index.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The computed sum.
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        input = np.asarray(input)
        labels = np.asarray(labels)
        return _scipy_sum_cpu(input, labels, index)
    else:
        input = cp.asarray(input)
        labels = cp.asarray(labels)
        result = _scipy_sum_gpu(input, labels, index)
        return _ensure_numpy(result)


def filter_mask(mask: np.ndarray, lbls: list, use_gpu: bool = True, verbose: bool = True) -> np.ndarray:
    """
    Retains only the pixels in 'mask' whose values are in 'lbls',
    replacing all other values with 0.
    Uses GPU acceleration if available and requested.

    Returns:
        np.ndarray: The filtered mask (always a NumPy array).
    """
    if not use_gpu or not _cupy_available:
        warnings.warn("GPU is not available or not requested. Using CPU for calculations.")
        xp_local = np
    else:
        xp_local = cp

    mask = xp_local.asarray(mask)
    max_val = int(mask.max())
    lut = xp_local.zeros(max_val + 1, dtype=mask.dtype)
    for l in lbls:
        if l <= max_val:
            lut[l] = l
    filtered_mask = lut[mask]

    return _ensure_numpy(filtered_mask)
