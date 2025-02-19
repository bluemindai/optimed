# Resample Utilities

## Overview
This module provides functions for resampling medical images with support for both CPU and GPU execution.

## Functions

### 1. `change_spacing(image: Nifti1Image, new_spacing: Union[float, Tuple[float, float, float]], order: int = 1, nr_cpus: int = 4, use_gpu: bool = False) -> Nifti1Image`
Resamples a NIfTI image to a new voxel spacing or target shape.
- **Returns**: The resampled NIfTI image.

### 2. `change_spacing_of_affine(affine: np.ndarray, zoom: float) -> np.ndarray`
Adjusts the affine matrix to account for the specified zoom factor.
- **Returns**: The modified affine matrix.

### 3. `resample_img(img: np.ndarray, new_shape: Tuple[int, int, int], order: int = 1, nr_cpus: int = 4) -> np.ndarray`
Resamples a NumPy image using SciPy with parallel processing support.
- **Returns**: The resampled image.

### 4. `resample_img_cucim(img: np.ndarray, new_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray`
Resamples an image using cuCIM for accelerated GPU processing.
- **Returns**: The resampled image.

## Usage Example

```python
import nibabel as nib
from optimed.processes.resample import change_spacing

# Load an example NIfTI image
img = nib.load("input_image.nii.gz")

# Resample the image to 1.25 mm spacing using GPU if available
resampled_img = change_spacing(img, new_spacing=1.25, order=1, nr_cpus=4, use_gpu=True)

# Save the resampled image
nib.save(resampled_img, "resampled_image.nii.gz")
