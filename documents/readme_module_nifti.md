# NIfTI Utilities

## Overview
This module provides a set of convenient functions for working with NIfTI images using either the `nibabel` or `SimpleITK` libraries. These functions simplify the process of loading, saving, converting, and manipulating NIfTI images. They also allow easy conversion between image orientations and handling metadata. The main goal of the library is to make the code simpler, more concise, and versatile, making it easier and faster to work with images.


## Functions

### 1. `load_nifti(file: str, canonical: bool=False, engine: str='nibabel') -> object`
Loads a NIfTI image from a specified file using either `nibabel` or `SimpleITK`.

- **Parameters**:
  - `file`: Path to the NIfTI file (`.nii` or `.nii.gz`).
  - `canonical`: If `True`, returns the image in the closest canonical orientation (applicable to `nibabel` only).
  - `engine`: Specifies whether to use `'nibabel'` or `'sitk'` for loading.

- **Returns**: A `nibabel.Nifti1Image` or `SimpleITK.Image` object depending on the engine.

---

### 2. `save_nifti(ndarray: np.ndarray, ref_image: object, dtype: np.dtype, save_to: str, engine: str='nibabel') -> None`
Saves a NIfTI image to a specified file using metadata from a reference image.

- **Parameters**:
  - `ndarray`: The numpy array representing the image data to be saved.
  - `ref_image`: The reference image (either `nibabel.Nifti1Image` or `SimpleITK.Image`) to extract metadata.
  - `dtype`: Data type to convert the image to.
  - `save_to`: Destination path for saving the NIfTI file.
  - `engine`: Specifies whether to use `'nibabel'` or `'sitk'` for saving.

---

### 3. `as_closest_canonical(img: object, engine: str = 'nibabel') -> object`
Converts a NIfTI image to its closest canonical orientation (for both `nibabel` and `SimpleITK`).

- **Parameters**:
  - `img`: The NIfTI image to be converted.
  - `engine`: Specifies whether the image is from `'nibabel'` or `'sitk'`.

- **Returns**: A new image transformed to the closest canonical orientation.

---

### 4. `undo_canonical(img_can: object, img_orig: object, engine: str = 'nibabel') -> object`
Reverts a NIfTI image from its canonical orientation back to its original orientation.

- **Parameters**:
  - `img_can`: The canonical image.
  - `img_orig`: The original image before canonical transformation.
  - `engine`: Specifies whether the images are from `'nibabel'` or `'sitk'`.

- **Returns**: The image reverted to its original orientation.

---

### 5. `maybe_convert_nifti_image_in_dtype(img: object, to_dtype: str, engine: str='nibabel') -> object`
Converts a NIfTI image to a specified data type if necessary.

- **Parameters**:
  - `img`: The NIfTI image to be converted.
  - `to_dtype`: The target data type for conversion (e.g., `'int32'`, `'float32'`).
  - `engine`: Specifies whether the image is from `'nibabel'` or `'sitk'`.

- **Returns**: The converted image or the original if no conversion was needed.

---

### 6. `get_image_orientation(img: object, engine: str = 'nibabel') -> Tuple[str, str, str]`
Retrieves the orientation of a NIfTI image in terms of axes (e.g., `'R'`, `'A'`, `'S'` for Right-Anterior-Superior).

- **Parameters**:
  - `img`: The NIfTI image to retrieve orientation from.
  - `engine`: Specifies whether the image is from `'nibabel'` or `'sitk'`.

- **Returns**: A tuple representing the orientation axes of the image.

---

### 7. `empty_img_like(ref: object, engine: str = 'nibabel') -> object`
Creates an empty NIfTI image with the same dimensions and affine transformation as the reference image, filled with zeros.

- **Parameters**:
  - `ref`: The reference image to copy dimensions and affine transformation from.
  - `engine`: Specifies whether the image is from `'nibabel'` or `'sitk'`.

- **Returns**: A new empty image with the same shape and metadata as the reference.

---
### 8. `split_image(img: object, parts: int, axis: int = 0) -> List[object]`
Splits a NIfTI image (nibabel or SimpleITK) into multiple parts along the specified axis.

- **Parameters**:
  - `img`: The NIfTI/ITK image to be split.
  - `parts`: The number of parts to split the image into.
  - `axis`: The axis along which to split. Default is 0 (the first axis).
    - For nibabel, axis=0 often corresponds to the x-dimension in 3D data.
    - For SimpleITK, axis=0 often corresponds to the z-dimension in 3D data.

- **Returns**: A list of sub-images, each representing a part of the original image.
  - If the input image is nibabel, the list contains nib.Nifti1Image objects.
  - If the input image is SimpleITK, the list contains sitk.Image objects.

---

## Example Usage

```python
import numpy as np
from optimed.wrappers.nifti import *

img = load_nifti('path_to_image.nii.gz', engine='nibabel')

canonical_img = as_closest_canonical(img, engine='nibabel')

converted_img = maybe_convert_nifti_image_in_dtype(canonical_img, to_dtype='int32', engine='nibabel')

save_nifti(converted_img.get_fdata(), canonical_img, np.int32, 'output_image.nii.gz', engine='nibabel')
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please feel free to submit pull requests or raise issues.