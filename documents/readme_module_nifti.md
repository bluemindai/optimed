# NIfTI Utilities

## Overview
This module provides a set of convenient functions for working with NIfTI images using either the `nibabel` or `SimpleITK` libraries. These functions simplify the process of loading, saving, converting, and manipulating NIfTI images. They also allow easy conversion between image orientations and handling metadata. The main goal of the library is to make the code simpler, more concise, and versatile, making it easier and faster to work with images.
This module now offers comprehensive utilities for working with NIfTI images using both `nibabel` and `SimpleITK`. It provides functions to load, save, convert, reorient, and manipulate NIfTI images, ensuring compatibility and ease of integration into various imaging pipelines.

## Functions

### 1. `load_nifti(file: str, canonical: bool=False, engine: str='nibabel') -> object`
Loads a NIfTI image from a specified file using either `nibabel` or `SimpleITK`.
- **Returns**: A `nibabel.Nifti1Image` or `SimpleITK.Image` object depending on the engine.

---

### 2. `save_nifti(ndarray: np.ndarray, ref_image: object, dtype: np.dtype, save_to: str, engine: str='nibabel') -> None`
Saves a NIfTI image to a specified file using metadata from a reference image.
- **Returns**: None

---

### 3. `as_closest_canonical(img: object, engine: str = 'nibabel') -> object`
Converts a NIfTI image to its closest canonical orientation (for both `nibabel` and `SimpleITK`).
- **Returns**: A new image transformed to the closest canonical orientation.

---

### 4. `undo_canonical(img_can: object, img_orig: object, engine: str = 'nibabel') -> object`
Reverts a NIfTI image from its canonical orientation back to its original orientation.
- **Returns**: The image reverted to its original orientation.

---

### 5. `maybe_convert_nifti_image_in_dtype(img: object, to_dtype: str, engine: str='nibabel') -> object`
Converts a NIfTI image to a specified data type if necessary.
- **Returns**: The converted image or the original if no conversion was needed.

---

### 6. `get_image_orientation(img: object, engine: str = 'nibabel') -> Tuple[str, str, str]`
Retrieves the orientation of a NIfTI image in terms of axes (e.g., `'R'`, `'A'`, `'S'` for Right-Anterior-Superior).
- **Returns**: A tuple representing the orientation axes of the image.

---

### 7. `empty_img_like(ref: object, engine: str = 'nibabel') -> object`
Creates an empty NIfTI image with the same dimensions and affine transformation as the reference image, filled with zeros.
- **Returns**: A new empty image with the same shape and metadata as the reference.

---
### 8. `split_image(img: object, parts: int, axis: int = 0) -> List[object]`
Splits a NIfTI image (nibabel or SimpleITK) into multiple parts along the specified axis.
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