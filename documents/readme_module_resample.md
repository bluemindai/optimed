# Resample Utilities

## Overview
This module provides functions for resampling medical images with support for both CPU and GPU execution.

## Functions

### 1. `change_spacing(img_in: nib.Nifti1Image, new_spacing: Union[float, list[float], np.ndarray] = 1.25, target_shape: Optional[tuple] = None, order: int = 0, nr_cpus: int = 1, dtype: Optional[np.dtype] = None, remove_negative: bool = False, force_affine: Optional[np.ndarray] = None, use_gpu: bool = True) -> nib.Nifti1Image`
Resamples a NIfTI image to a new voxel spacing or target shape.
- **Returns**: The resampled NIfTI image.

### 2. `change_spacing_of_affine(affine: np.ndarray, zoom: float = 0.5) -> np.ndarray`
Adjusts the affine matrix to account for the specified zoom factor.
- **Returns**: The modified affine matrix.

### 3. `resample_img(img: np.ndarray, zoom: float = 0.5, order: int = 0, nr_cpus: int = -1) -> np.ndarray`
Resamples a 2D, 3D, or 4D NumPy image using SciPy with parallel processing support.
- **Returns**: The resampled image.

### 4. `resample_img_cucim(img: np.ndarray, zoom: float = 0.5, order: int = 0) -> np.ndarray`
Resamples a 2D, 3D, or 4D image using cuCIM for accelerated GPU processing.
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
