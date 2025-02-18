# Resample Utilities

## Overview
Модуль предоставляет функции для ресемплинга медицинских изображений с поддержкой CPU и GPU.

## Functions

1. `change_spacing`: Ресемплирует NIfTI изображение с новым spacing или целевой формой.  
2. `change_spacing_of_affine`: Изменяет affine матрицу с учётом заданного зума.  
3. `resample_img`: Ресемплирует NumPy изображение через SciPy с возможностью параллельной обработки.  
4. `resample_img_cucim`: Ресемплирует изображение с использованием cuCIM для ускорения на GPU.

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
