# Modules

## Table of Contents
1. [Wrappers](#wrappers)
2. [Processes](#processes)

## Wrappers

The `wrappers` module is a collection of wrapper functions built around core libraries (such as `nibabel`, `SimpleITK`, `scipy`, `os`, etc.). The primary goal of these functions is to standardize the code, making it more versatile, readable, and multifunctional.

### Available Wrappers
1. **NIfTI Wrappers**  
   Wrappers around `nibabel` for working with NIfTI image files.  
   [Read the documentation](readme_module_nifti.md)

2. **Calculations Wrappers**  
   Wrappers around `scipy` for enhanced array operations and data manipulation.  
   [Read the documentation](readme_module_calculations.md)

3. **Operations Wrappers**  
   Wrappers for file system operations (`os`), including file management and directory handling.  
   [Read the documentation](readme_module_operations.md)

## Processes

The processes module includes functions to crop, convert, resample, preview, and postprocess medical images. These submodules streamline various stages of image analysis and integrate seamlessly with the wrappers.

### Available Processes
1. **Crop Utilities**  
   Provides image cropping functionalities for isolating regions of interest.  
   [Read the documentation](readme_module_crop.md)

2. **Convert Utilities**  
   Contains tools for converting medical image formats (e.g., DICOM, NRRD, MHA) to NIfTI.  
   [Read the documentation](readme_module_convert.md)

3. **Resample Utilities**  
   Offers methods for resampling images to adjust resolution and spacing.  
   [Read the documentation](readme_module_resample.md)

4. **Preview Utilities**  
   Facilitates 3D visualization of medical image segmentations using VTK.  
   [Read the documentation](readme_module_preview.md)

5. **Postprocess Utilities**  
   Provides post-processing functions to clean and refine segmentation outputs.  
   [Read the documentation](readme_module_postprocess.md)