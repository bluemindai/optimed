# NumPy Utilities

This module provides a set of utility functions designed to simplify and streamline operations on NumPy arrays. It includes functions for statistical calculations, array reshaping, concatenation, and more. These functions are created with the goal of enhancing code readability and conciseness, while offering flexibility in handling different array operations such as axis-specific manipulations, excluding NaN values, and maintaining array dimensions.

## Functions

### 1. `mean(arr: np.ndarray, axis: int = None, exclude_nan: bool = False) -> float`
Calculates the mean of an array, with the option to exclude NaN values.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `axis` (int, optional): Axis along which the mean is computed. Default is `None` (mean of the entire array).
  - `exclude_nan` (bool, optional): If `True`, NaN values will be ignored.
- **Returns**: Mean value of the array.

### 2. `sum(arr: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray`
Computes the sum of an array, with an option to retain reduced dimensions.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `axis` (int, optional): Axis along which the sum is computed.
  - `keepdims` (bool, optional): If `True`, retains the reduced dimensions.
- **Returns**: Sum of the array along the specified axis.

### 3. `median(arr: np.ndarray, axis: int = None, exclude_nan: bool = False) -> float`
Calculates the median of an array, with the option to exclude NaN values.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `axis` (int, optional): Axis along which the median is computed.
  - `exclude_nan` (bool, optional): If `True`, NaN values will be ignored.
- **Returns**: Median value of the array.

### 4. `std(arr: np.ndarray, axis: int = None, ddof: int = 0, exclude_nan: bool = False) -> float`
Calculates the standard deviation of an array, with the option to exclude NaN values.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `axis` (int, optional): Axis along which the standard deviation is computed.
  - `ddof` (int, optional): Delta Degrees of Freedom, where the divisor is N - ddof.
  - `exclude_nan` (bool, optional): If `True`, NaN values will be ignored.
- **Returns**: Standard deviation of the array.

### 5. `reshape(arr: np.ndarray, new_shape: Tuple[int], flatten_first: bool = False) -> np.ndarray`
Reshapes an array to a new shape, with an option to flatten the array first.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `new_shape` (tuple): Desired shape of the output array.
  - `flatten_first` (bool, optional): If `True`, flattens the array before reshaping.
- **Returns**: Reshaped array.

### 6. `concatenate(arrays: list, axis: int = 0) -> np.ndarray`
Concatenates a list of arrays along a specified axis.

- **Parameters**:
  - `arrays` (list): List of arrays to concatenate.
  - `axis` (int, optional): Axis along which to join the arrays.
- **Returns**: Concatenated array.

### 7. `linspace(start: float, stop: float, num: int = 50, endpoint: bool = True) -> np.ndarray`
Generates evenly spaced values over a specified interval.

- **Parameters**:
  - `start` (float): Starting value of the sequence.
  - `stop` (float): End value of the sequence.
  - `num` (int, optional): Number of samples to generate.
  - `endpoint` (bool, optional): If `True`, includes the stop value.
- **Returns**: Array of evenly spaced values.

### 8. `argmax(arr: np.ndarray, axis: int = None) -> int`
Returns the indices of the maximum values along an axis.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `axis` (int, optional): Axis along which to find the maximum.
- **Returns**: Indices of the maximum values.

### 9. `clip(arr: np.ndarray, min_value: float, max_value: float) -> np.ndarray`
Clips (limits) the values in an array between a specified minimum and maximum.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `min_value` (float): Minimum value.
  - `max_value` (float): Maximum value.
- **Returns**: Clipped array.

### 10. `unique(arr: np.ndarray, return_counts: bool = False) -> np.ndarray`
Finds unique elements in an array with an option to return their counts.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `return_counts` (bool, optional): If `True`, returns the counts of unique elements.
- **Returns**: Unique elements, optionally with their counts.

### 11. `zeros_like(arr: np.ndarray, dtype=None) -> np.ndarray`
Creates an array of zeros with the same shape and type as the input array.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `dtype` (optional): Overrides the data type of the result.
- **Returns**: Array of zeros.

### 12. `ones_like(arr: np.ndarray, dtype=None) -> np.ndarray`
Creates an array of ones with the same shape and type as the input array.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `dtype` (optional): Overrides the data type of the result.
- **Returns**: Array of ones.

### 13. `full_like(arr: np.ndarray, fill_value, dtype=None) -> np.ndarray`
Creates an array filled with a specified value, with the same shape and type as the input array.

- **Parameters**:
  - `arr` (np.ndarray): Input array.
  - `fill_value`: Value to fill the array with.
  - `dtype` (optional): Overrides the data type of the result.
- **Returns**: Array filled with `fill_value`.

## Usage Examples

```python
import numpy as np
from optimed.wrappers.numpy import *

# Mean calculation ignoring NaN values
arr = np.array([1, 2, np.nan, 4])
print(mean(arr, exclude_nan=True))  # Output: 2.333...

# Reshaping an array
arr = np.array([1, 2, 3, 4])
new_arr = reshape(arr, (2, 2))
print(new_arr)  # Output: [[1, 2], [3, 4]]

# Summing across a specific axis
arr = np.array([[1, 2], [3, 4]])
print(sum(arr, axis=0))  # Output: [4, 6]
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please feel free to submit pull requests or raise issues.