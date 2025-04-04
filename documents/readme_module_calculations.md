# Calculations Utilities

## Overview
This module provides GPU-accelerated and CPU fallback implementations of common image processing calculations using SciPy and CuPy. Functions for connected component labeling, morphological operations, distance transform, and custom filtering are available to support robust image analysis.

## Functions

### 1. `scipy_label(input: np.ndarray, structure: np.ndarray = None, use_gpu: bool = True) -> Tuple[np.ndarray, int]`
Labels connected components in an input array using GPU acceleration when available.
- Additional Info: Connected component labeling identifies and labels contiguous regions within an image.
- **Returns**: A tuple with the labeled array and the number of components.

---

### 2. `scipy_binary_dilation(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, brute_force: bool = False, use_gpu: bool = True) -> np.ndarray`
Applies binary dilation to the input array, using the GPU if available.
- Additional Info: Binary dilation is an image processing operation that expands object boundaries, making objects larger.
- **Returns**: The dilated array.

---

### 3. `scipy_binary_closing(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, use_gpu: bool = True) -> np.ndarray`
Performs binary closing to fill gaps in binary images.
- Additional Info: Binary closing is an image processing technique that fills small holes within objects by dilating then eroding the image.
- **Returns**: The closed array.

---

### 4. `scipy_binary_erosion(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, use_gpu: bool = True) -> np.ndarray`
Applies binary erosion to shrink objects in a binary image.
- Additional Info: Binary erosion is an operation that reduces the object boundaries, effectively removing small-scale noise.
- **Returns**: The eroded array.

---

### 5. `scipy_distance_transform_edt(input: np.ndarray, sampling: Tuple[float, float] = (1, 1), use_gpu: bool = True) -> np.ndarray`
Calculates the Euclidean distance transform.
- Additional Info: The distance transform computes the distance of each pixel to the nearest background pixel, which is useful for shape analysis.
- **Returns**: The distance-transformed array.

---

### 6. `scipy_minimum(input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True) -> np.ndarray`
Finds the minimum value within regions defined by a label array.
- Additional Info: Minimum filtering extracts the smallest value in a specified region, assisting in segmentation tasks.
- **Returns**: The minimum value.

---

### 7. `scipy_sum(input: np.ndarray, labels: np.ndarray, index: int, use_gpu: bool = True) -> np.ndarray`
Computes the sum of values over regions specified by labels.
- Additional Info: This function aggregates pixel values over a labeled region, providing a measure of total intensity.
- **Returns**: The sum.

---

### 8. `filter_mask(mask: np.ndarray, labels: list, use_gpu: bool = True) -> np.ndarray`
Filters the input mask so that only specified label values remain.
- Additional Info: Filter mask operation retains only the designated labels, removing unwanted regions.
- **Returns**: The filtered mask.

---

### 9. `scipy_binary_opening(input: np.ndarray, structure: np.ndarray = None, iterations: int = 1, use_gpu: bool = True) -> np.ndarray`
Applies binary opening to the input array.
- Additional Info: Binary opening removes small objects and noise by first eroding and then dilating the image.
- **Returns**: The opened array.

---

### 10. `scipy_binary_fill_holes(input: np.ndarray, structure: np.ndarray = None, use_gpu: bool = True) -> np.ndarray`
Fills holes in the binary input array.
- Additional Info: This function fills interior holes within objects, improving segmentation results.
- **Returns**: The filled array.

---

### 11. `scipy_median_filter(input: np.ndarray, size: Tuple[int, int] = (3, 3), footprint: np.ndarray = None, mode: str = "reflect", cval: float = 0.0, use_gpu: bool = True) -> np.ndarray`
Applies a median filter to reduce noise while preserving edges.
- Additional Info: Median filtering is useful for removing impulse noise without blurring sharp edges.
- **Returns**: The filtered array.

## Usage Example

```python
import numpy as np
from optimed.wrappers.calculations import scipy_label, scipy_distance_transform_edt

# Create a sample binary image
binary_image = np.random.randint(0, 2, size=(100, 100))

# Label connected components (using GPU if available)
labeled, num_components = scipy_label(binary_image, use_gpu=True)
print("Number of components:", num_components)

# Compute the distance transform
dist_transform = scipy_distance_transform_edt(binary_image, sampling=(1, 1), use_gpu=True)
print("Distance transform shape:", dist_transform.shape)
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please feel free to submit pull requests or raise issues.
