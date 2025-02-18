# Crop Utilities

## Overview
This module provides functions to crop 3D NIfTI images based on masks or explicit boundaries. It includes utilities to compute bounding boxes from masks, crop images to those boxes, and restore the cropped images back to the original image dimensions.

## Functions

### 1. `get_bbox_from_mask(mask: np.ndarray, outside_value: int = -900, addon: int = 0, verbose: bool = True) -> list`
Computes the bounding box of the foreground in a mask.
- Additional Info: It computes the minimum and maximum indices along each axis with an optional margin.
- **Returns**: A list with bounding box coordinates.

---

### 2. `crop_to_bbox(data_or_image: Union[np.ndarray, nib.Nifti1Image], bbox: list, dtype: np.dtype = None) -> Union[np.ndarray, nib.Nifti1Image]`
Crops an image or array to the specified bounding box.
- Additional Info: For NIfTI images, the affine transformation is adapted to the cropped region.
- **Returns**: The cropped image or array.

---

### 3. `crop_to_mask(img_in: Union[str, nib.Nifti1Image], mask_img: Union[str, nib.Nifti1Image], addon: list = [0, 0, 0], dtype: np.dtype = None, save_to: str = None, verbose: bool = False) -> tuple`
Crops an image based on a mask and returns the cropped image along with its bounding box.
- Additional Info: Reads images from file paths if needed and optionally saves the output.
- **Returns**: A tuple containing the cropped image and the bounding box.

---

### 4. `undo_crop(img: Union[str, nib.Nifti1Image], ref_img: Union[str, nib.Nifti1Image], bbox: list, save_to: str = None) -> nib.Nifti1Image`
Restores a cropped image to its original dimensions using a reference image.
- Additional Info: Inserts the cropped image into a zero-filled array shaped as the original image.
- **Returns**: The image fitted back into the original space.

---

### 5. `crop_by_xyz_boudaries(img_in: Union[nib.Nifti1Image, str], save_to: str = None, x_start: int = 0, x_end: int = 512, y_start: int = 0, y_end: int = 512, z_start: int = 0, z_end: int = 50, dtype=np.int16) -> nib.Nifti1Image`
Crops an image along explicit x, y, and z boundaries.
- Additional Info: Useful for fixed region cropping by specifying indices.
- **Returns**: The cropped NIfTI image.

## Usage Example

```python
from optimed.processes.crop import crop_to_mask, undo_crop

# Crop an image based on a mask with an additional margin
cropped_img, bbox = crop_to_mask("input_image.nii.gz", "mask.nii.gz", addon=[10, 10, 10], verbose=True)

# Restore the cropped image back into the original image dimensions
restored_img = undo_crop(cropped_img, "input_image.nii.gz", bbox)
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please submit pull requests or raise issues.
