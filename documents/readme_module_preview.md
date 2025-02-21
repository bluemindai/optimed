# Preview Utilities

## Overview
This module offers functions to visualize 3D medical image segmentations. It converts NumPy images to VTK objects and renders 3D previews with segmentation overlays, captions, and custom view settings, saving the result as a PNG image.

## Functions

### 1. `preview_3d_image(input_path: Union[str, nib.Nifti1Image], output: str, segmentation_dict: Dict[int, Dict[str, Any]], view_direction: Tuple[str,list], smoothing: int = 20, shading: int = 20, background_color: Tuple[int,int,int] = (0,0,0), window_size: Tuple[int,int] = (800,800)) -> None`
Loads a NIfTI image, converts it to VTK format, applies segmentation overlays, and saves a 3D preview image.
- Additional Info: Integrates file checking, orientation conversion, and VTK rendering to produce a comprehensive preview.
- **Returns**: None.

### 2. `_numpy_to_vtk_image_data(np_data: np.ndarray, spacing=(1,1,1)) -> vtk.vtkImageData`
Converts a NumPy array (in RAS orientation) to a vtkImageData object.
- **Internal method**
- Additional Info: Facilitates the integration of image data into VTK pipelines by preserving voxel spacing.
- **Returns**: A vtkImageData object.

---

### 3. `_render_segmentation_to_image(vtk_image: vtk.vtkImageData, segmentation_dict: Dict[int, Dict[str, Any]], smoothing: int = 20, shading: int = 20, view_direction: str = 'A', background_color: Tuple[int,int,int] = (0,0,0), window_size: Tuple[int,int] = (800,800), output_filename: str = "segmentation_preview.png") -> None`
Renders the provided VTK image combined with segmentation information and saves the result as a PNG file.
- **Internal method**
- Additional Info: Uses marching cubes and smoothing filters to create surfaces, then overlays captions based on label properties.
- **Returns**: None.

---

## Usage Example

```python
from optimed.processes.preview import preview_3d_image

# Define segmentation properties for labels
segmentation_dict = {
    1: {'color': (255, 255, 0), 'opacity': 0.4, 'text': {'label': 'Liver', 'color': (0, 0, 0), 'size': 5}},
    2: {'color': (255, 0, 0), 'opacity': 1.0, 'text': {'label': 'Tumor', 'color': (0, 0, 0), 'size': 5}}
}

# Generate a 3D preview image from a NIfTI file
preview_3d_image(
    input_path="sample_image.nii.gz",
    output="preview_output.png",
    segmentation_dict=segmentation_dict,
    view_direction='A',
    smoothing=20,
    shading=20,
    background_color=(255,255,255),
    window_size=(1200,1200)
)
