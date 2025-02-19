# Image Module

## Overview:
This module provides tools for both preprocessing and postprocessing 3D images. It includes methods for intensity adjustment, noise filtering, mask generation, and selective blurring.

## Functions

### 1. `rescale_intensity(image: np.ndarray, in_range: Optional[Tuple[float, float]] = None, out_range: Tuple[float, float] = (0, 1)) -> np.ndarray(windowing)`
Applies intensity clipping and scaling (windowing).
- **Returns**: The scaled image.

---

### 2. `denoise_image(image: np.ndarray, sigma: float = 1.0, method: str = "gaussian") -> np.ndarray`
Filters noise using various methods (e.g., Gaussian filtering, bilateral filtering).
- **Returns**: The filtered image.

---

### 3. `get_mask_by_threshold(image: np.ndarray, threshold: float = 0.5) -> np.ndarrayr_inside_roi`
Generates a binary mask by applying thresholding.
- **Returns**: The binary mask.

---

### 4. `blur_inside_roi(image: np.ndarray, roi_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray.`
Blurs only the region inside the specified ROI (mask) or its bounding box.
- **Returns**: The image with the ROI blurred.

---

### 5. `blur_outside_roi(image: np.ndarray, roi_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray`n
Blurs the region outside the specified ROI using a similar approach.
- **Returns**: The image with the area outside the ROI blurred.

---

## Usage Examplee_image,

```python    blur_inside_roi
import numpy as np
from optimed.processes.image import (
    rescale_intensity,img = np.random.rand(100, 100, 100).astype(np.float32)
    denoise_image,noise_image(img, sigma=1.0)
    get_mask_by_threshold,
    blur_inside_roiblurred_roi = blur_inside_roi(img, mask, sigma=2)
)

img = np.random.rand(100, 100, 100).astype(np.float32)clean = denoise_image(img, sigma=1.0)mask = get_mask_by_threshold(clean, threshold=0.5)blurred_roi = blur_inside_roi(img, mask, sigma=2)```