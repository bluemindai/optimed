# Postprocess Utilities

## Overview
This module provides a set of functions for post-processing segmentation results. It includes operations for boundary checking, component analysis, segment removal based on size, distance, and connectivity, as well as hole filling. The functions support execution on both CPU and GPU for efficient processing of large datasets.

## Functions

### 1. `if_touches_border_3d(mask: np.ndarray, use_gpu: bool = True) -> bool`
Checks if a 3D mask touches any image boundaries.
- **Returns**: True if the mask touches at least one boundary.

---

### 2. `if_touches_border_2d(mask: np.ndarray, use_gpu: bool = True) -> bool`
Checks if a 2D mask touches the image boundaries.
- **Returns**: True if the mask touches a boundary.

---

### 3. `find_component_by_centroid(components: np.ndarray, target_centroid: list, num_components: int, atol: int = 3, use_gpu: bool = True) -> Union[int, None]`
Finds a component by its centroid with a given tolerance.
- **Returns**: The component label or None if no match is found.

---

### 4. `find_component_by_point(components: np.ndarray, target_point: list, use_gpu: bool = True) -> Union[int, None]`
Finds a component containing the specified point.
- **Returns**: The component label or None.

---

### 5. `delete_small_segments(binary_mask: np.ndarray, interval: list = [10, np.inf], use_gpu: bool = True, verbose: bool = True) -> np.ndarray`
Removes segments whose sizes fall outside the specified interval.
- **Returns**: The filtered binary mask.

---

### 6. `delete_segments_disconnected_from_point(binary_mask: np.ndarray, target_point: list, size_threshold: int = 10, use_gpu: bool = True, verbose: bool = True) -> np.ndarray`
Removes segments not connected to the specified point. If the component containing the specified point is smaller than size_threshold, it is removed.
- **Returns**: The updated binary mask, retaining only the component containing the target_point (if it is large enough).

---

### 7. `delete_segments_disconnected_from_parent(binary_mask: np.ndarray, parent_mask: np.ndarray, size_threshold: int = 10, use_gpu: bool = True, verbose: bool = True) -> np.ndarray`
Removes segments that either do not intersect with the parent mask or are smaller than size_threshold.
- **Returns**: The updated binary mask.

---

### 8. `delete_segments_distant_from_point(binary_mask: np.ndarray, target_point: list, distance_threshold: int = 5, size_threshold: int = 10, keep_large: bool = True, use_gpu: bool = True, verbose: bool = True) -> np.ndarray`
Removes segments that are too far from the specified point, except for those that are large enough.
- **Returns**: The updated binary mask.

---

### 9. `delete_nearby_segments_with_buffer(binary_mask: np.ndarray, parent_mask: np.ndarray, distance_threshold: int = 5, size_threshold: int = 10, use_gpu: bool = True, verbose: bool = True) -> np.ndarray`
Removes segments that are too close to the main component (specified by parent_mask) considering a buffer.
- **Returns**: The cleaned binary mask.

---

### 10. `delete_touching_border_segments(binary_mask: np.ndarray, size_threshold: int = 100, use_gpu: bool = True, verbose: bool = True) -> np.ndarray`
Removes segments that touch the image boundary and are smaller than a specified size threshold.
- **Returns**: The updated binary mask.

---

### 11. `fill_holes_2d(binary_mask: np.ndarray, max_hole_size: int = 100, use_gpu: bool = True) -> np.ndarray`
Fills small holes in a 2D binary mask.
- **Returns**: The 2D mask with filled holes.

---

### 12. `fill_holes_3d(binary_mask: np.ndarray, max_hole_size: int = 100, use_gpu: bool = True) -> np.ndarray`
Fills small holes in a 3D binary mask.
- **Returns**: The 3D mask with filled holes.

## Usage Example

```python
import numpy as np
from optimed.postprocess import (
    delete_small_segments,
    delete_segments_distant_from_point,
    fill_holes_3d
)

# Generate a random 3D binary mask
mask = np.random.randint(0, 2, size=(100, 100, 100))

# Remove small segments
filtered_mask = delete_small_segments(mask, interval=[10, 500], use_gpu=False, verbose=True)

# Remove segments distant from a specified point
target = [50, 50]  # example for 2D (adapt for 3D as needed)
updated_mask = delete_segments_distant_from_point(
    filtered_mask, target, distance_threshold=5, size_threshold=10, use_gpu=False, verbose=True
)

# Fill holes in the 3D mask
final_mask = fill_holes_3d(updated_mask, max_hole_size=50, use_gpu=False)
```
