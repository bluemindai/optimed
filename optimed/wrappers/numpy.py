from typing import Tuple
import numpy as np


def mean(arr: np.ndarray, axis: int = None, exclude_nan: bool = False) -> float:
    """
    Calculate the mean of an array with optional exclusion of NaN values.

    Parameters:
        arr (np.ndarray): The input array.
        axis (int, optional): Axis along which the mean is computed. Default is None (mean of the entire array).
        exclude_nan (bool, optional): If True, NaN values will be ignored in the calculation. Default is False.

    Returns:
        float: The mean value of the array.
    """
    if exclude_nan:
        return np.nanmean(arr, axis=axis)
    else:
        return np.mean(arr, axis=axis)


def sum(arr: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the sum of an array with an option to retain dimensions.

    Parameters:
        arr (np.ndarray): The input array.
        axis (int, optional): Axis along which the sum is computed. Default is None (sum of the entire array).
        keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        np.ndarray: The sum of the array along the specified axis.
    """
    return np.sum(arr, axis=axis, keepdims=keepdims)


def median(arr: np.ndarray, axis: int = None, exclude_nan: bool = False) -> float:
    """
    Calculate the median of an array with optional exclusion of NaN values.

    Parameters:
        arr (np.ndarray): The input array.
        axis (int, optional): Axis along which the median is computed. Default is None (median of the entire array).
        exclude_nan (bool, optional): If True, NaN values will be ignored in the calculation. Default is False.

    Returns:
        float: The median value of the array.
    """
    if exclude_nan:
        return np.nanmedian(arr, axis=axis)
    else:
        return np.median(arr, axis=axis)


def std(arr: np.ndarray, axis: int = None, ddof: int = 0, exclude_nan: bool = False) -> float:
    """
    Calculate the standard deviation of an array with optional exclusion of NaN values.

    Parameters:
        arr (np.ndarray): The input array.
        axis (int, optional): Axis along which the standard deviation is computed. Default is None (standard deviation of the entire array).
        ddof (int, optional): Delta Degrees of Freedom. The divisor used in the calculation is N - ddof, where N represents the number of elements.
        exclude_nan (bool, optional): If True, NaN values will be ignored in the calculation. Default is False.

    Returns:
        float: The standard deviation of the array.
    """
    if exclude_nan:
        return np.nanstd(arr, axis=axis, ddof=ddof)
    else:
        return np.std(arr, axis=axis, ddof=ddof)


def reshape(arr: np.ndarray, new_shape: Tuple[int], flatten_first: bool = False) -> np.ndarray:
    """
    Reshape an array with an option to flatten the array first.

    Parameters:
        arr (np.ndarray): The input array.
        new_shape (tuple): The desired shape of the output array.
        flatten_first (bool, optional): If True, the array will be flattened before reshaping. Default is False.

    Returns:
        np.ndarray: The reshaped array.
    """
    if np.prod(new_shape) != arr.size:
        raise ValueError(f"Cannot reshape array of size {arr.size} into shape {new_shape}")
    if flatten_first:
        arr = arr.flatten()
    return np.reshape(arr, new_shape)


def concatenate(arrays: list, axis: int = 0) -> np.ndarray:
    """
    Concatenate a sequence of arrays along the specified axis.

    Parameters:
        arrays (list): A list of arrays to concatenate.
        axis (int, optional): The axis along which the arrays will be joined. Default is 0.

    Returns:
        np.ndarray: The concatenated array.
    """
    try:
        return np.concatenate(arrays, axis=axis)
    except ValueError as e:
        raise ValueError(f"Concatenation failed: {e}")


def linspace(start: float, stop: float, num: int = 50, endpoint: bool = True) -> np.ndarray:
    """
    Generate evenly spaced values over a specified interval.

    Parameters:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int, optional): The number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, the stop value is included. Default is True.

    Returns:
        np.ndarray: A NumPy array of evenly spaced values.
    """
    if num < 1:
        raise ValueError("Number of samples must be at least 1.")
    return np.linspace(start, stop, num=num, endpoint=endpoint)


def argmax(arr: np.ndarray, axis: int = None) -> int:
    """
    Returns the indices of the maximum values along an axis.

    Parameters:
        arr (np.ndarray): The input array.
        axis (int, optional): Axis along which to find the maximum. Default is None (flattened array).

    Returns:
        int: The indices of the maximum values.
    """
    try:
        return np.argmax(arr, axis=axis)
    except ValueError as e:
        raise ValueError(f"Argmax operation failed: {e}")


def clip(arr: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """
    Clip (limit) the values in an array to a specified minimum and maximum.

    Parameters:
        arr (np.ndarray): The input array.
        min_value (float): The minimum value to clip to.
        max_value (float): The maximum value to clip to.

    Returns:
        np.ndarray: The clipped array.
    """
    try:
        return np.clip(arr, min_value, max_value)
    except ValueError as e:
        raise ValueError(f"Clipping operation failed: {e}")


def unique(arr: np.ndarray, return_counts: bool = False) -> np.ndarray:
    """
    Find the unique elements of an array with an option to return counts.

    Parameters:
        arr (np.ndarray): The input array.
        return_counts (bool, optional): If True, also return the number of times each unique item appears in the array. Default is False.

    Returns:
        np.ndarray: The sorted unique elements of the array.
        np.ndarray (optional): The counts of the unique elements.
    """
    if return_counts:
        return np.unique(arr, return_counts=True)
    else:
        return np.unique(arr)


def zeros_like(arr: np.ndarray, dtype=None) -> np.ndarray:
    """
    Create an array of zeros with the same shape and type as the given array.

    Parameters:
        arr (np.ndarray): The input array whose shape and type will be used.
        dtype (data-type, optional): Overrides the data type of the result. Default is None (use the dtype of `arr`).

    Returns:
        np.ndarray: An array of zeros with the same shape and dtype as the input array.
    """
    if dtype is None:
        dtype = arr.dtype
    return np.zeros(arr.shape, dtype=dtype)


def ones_like(arr: np.ndarray, dtype=None) -> np.ndarray:
    """
    Create an array of ones with the same shape and type as the given array.

    Parameters:
        arr (np.ndarray): The input array whose shape and type will be used.
        dtype (data-type, optional): Overrides the data type of the result. Default is None (use the dtype of `arr`).

    Returns:
        np.ndarray: An array of ones with the same shape and dtype as the input array.
    """
    if dtype is None:
        dtype = arr.dtype
    return np.ones(arr.shape, dtype=dtype)


def full_like(arr: np.ndarray, fill_value, dtype=None) -> np.ndarray:
    """
    Create an array filled with a specified value with the same shape and type as the given array.

    Parameters:
        arr (np.ndarray): The input array whose shape and type will be used.
        fill_value: The value to fill the array with.
        dtype (data-type, optional): Overrides the data type of the result. Default is None (use the dtype of `arr`).

    Returns:
        np.ndarray: An array filled with `fill_value` with the same shape and dtype as the input array.
    """
    if dtype is None:
        dtype = arr.dtype
    return np.full(arr.shape, fill_value, dtype=dtype)
