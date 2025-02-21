# Convert Utilities

## Overview
This module provides a set of functions to convert medical image formats between different standards. It includes conversions between SimpleITK and nibabel images, as well as utilities to convert DICOM series, NRRD, and MHA files to NIfTI.

## Functions

### 1. `convert_sitk_to_nibabel(sitk_image: sitk.Image) -> nib.Nifti1Image`
Converts a SimpleITK image to a NIfTI image.
- Additional Info: Extracts image data using SimpleITK and constructs a nibabel image with the appropriate affine.
- **Returns**: A NIfTI image.

---

### 2. `convert_nibabel_to_sitk(nibabel_image: nib.Nifti1Image) -> sitk.Image`
Converts a NIfTI image to a SimpleITK image.
- Additional Info: Restores origin, spacing, and direction from the nibabel affine.
- **Returns**: A SimpleITK image.

---

### 3. `convert_dcm_to_nifti(dicom_path: str, nifti_file: str, permute_axes: bool = False, return_object: bool = True, return_type: str = 'nibabel') -> Union[nib.Nifti1Image, sitk.Image]`
Converts a DICOM series to a NIfTI file.
- Additional Info: Uses SimpleITK ImageSeriesReader and optionally returns the image as a nibabel or SimpleITK object.
- **Returns**: A converted NIfTI image.

---

### 4. `convert_nrrd_to_nifti(nrrd_file: Union[str, object], nifti_file: str, return_object: bool = True, return_type: str = 'nibabel') -> Union[nib.Nifti1Image, sitk.Image]`
Converts a NRRD file to NIfTI format.
- Additional Info: Reads NRRD data and header to construct an affine; handles coordinate flipping if needed.
- **Returns**: A converted NIfTI image.

---

### 5. `conver_mha_to_nifti(mha_file: Union[str, object], nifti_file: str, return_object: bool = True, return_type: str = 'nibabel') -> Union[nib.Nifti1Image, sitk.Image]`
Converts a MHA file to NIfTI format.
- Additional Info: Reads the MHA image using SimpleITK and writes it as a NIfTI.
- **Returns**: A converted NIfTI image.

---

### 6. `convert_nifti_to_nrrd(input_nifti_filename: str, output_seg_filename: str, segment_metadata: list = None, verbose: bool = True) -> None`
Converts a plain NIfTI segmentation (label map) into a Slicer segmentation (NRRD) file with metadata.
- Additional Info:
    - Reads the NIfTI segmentation and optionally auto-generates segment metadata from unique labels (excluding background) if none are provided.
    - Uses the slicerio library for reading and writing segmentations.
- **Returns**: None.

## Usage Example

```python
from optimed.processes.convert import convert_dcm_to_nifti

# Convert a DICOM series from 'dicom_folder' to 'output_image.nii.gz'
nifti_img = convert_dcm_to_nifti("dicom_folder", "output_image.nii.gz", permute_axes=True)
print("Conversion successful!")
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please submit pull requests or raise issues.
