import nibabel as nib
import numpy as np
import vtk
from vtk.util import numpy_support
from typing import Union, Dict, Tuple, Any
from optimed.wrappers.nifti import load_nifti, as_closest_canonical
from optimed.wrappers.operations import exists


def _numpy_to_vtk_image_data(np_data: np.ndarray, spacing=(1, 1, 1)) -> vtk.vtkImageData:
    """
    Converts a NumPy array (in RAS orientation, axis order (X, Y, Z))
    to a vtkImageData object.

    Parameters:
        np_data (np.ndarray): The input NumPy array.
        spacing (Tuple[float, float, float]): The voxel spacing.
    
    Returns:
        vtk.vtkImageData: The converted vtkImageData object.
    """
    x, y, z = np_data.shape
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(x, y, z)
    vtk_image.SetSpacing(spacing)
    
    flat_data = np.ravel(np_data, order='F')
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=flat_data,
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image


def _render_segmentation_to_image(
    vtk_image: vtk.vtkImageData,
    segmentation_dict: Dict[int, Dict[str, Any]],
    smoothing: int = 20,
    shading: int = 20,
    view_direction: str = 'A',
    background_color: Tuple[int, int, int] = (0, 0, 0),
    window_size: Tuple[int, int] = (800, 800),
    output_filename: str = "segmentation_preview.png"
) -> None:
    """
    Renders a 3D segmentation and saves it as a PNG file.
    If a label contains "text": {"label": ..., "color": ..., "size": ...},
    a caption is created whose bounding box is correctly centered relative
    to the base position determined by view_direction.

    Parameters:
        vtk_image (vtk.vtkImageData): The input vtkImageData object.
        segmentation_dict (Dict[int, Dict[str, Any]]): A dictionary where keys are label values
            and values are dictionaries with properties for each label.
        smoothing (int): The number of smoothing iterations.
        shading (int): The shading factor.
        view_direction (str): The view direction (R, L, A, P, S, I).
        background_color (Tuple[int, int, int]): The background color (RGB).
        window_size (Tuple[int, int]): The window size.
        output_filename (str): The output PNG file name.

    Returns:
        None
    """
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(*window_size)
    renderWindow.SetOffScreenRendering(True)
    
    camera = renderer.GetActiveCamera()

    # Create actors for each label
    for label_value, props in segmentation_dict.items():
        # Create the label surface using marching cubes
        contour = vtk.vtkDiscreteMarchingCubes()
        contour.SetInputData(vtk_image)
        contour.SetValue(0, label_value)
        contour.Update()

        poly_port = contour.GetOutputPort()
        if smoothing > 0:
            smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
            smooth_filter.SetInputConnection(poly_port)
            smooth_filter.SetNumberOfIterations(smoothing)
            smooth_filter.BoundarySmoothingOff()
            smooth_filter.FeatureEdgeSmoothingOff()
            smooth_filter.SetFeatureAngle(120.0)
            smooth_filter.SetPassBand(0.001)
            smooth_filter.NonManifoldSmoothingOn()
            smooth_filter.NormalizeCoordinatesOn()
            smooth_filter.Update()
            poly_port = smooth_filter.GetOutputPort()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(poly_port)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        color = props.get('color', (1, 1, 1))
        opacity = props.get('opacity', 1.0)
        if max(color) > 1:
            color = tuple(c / 255.0 for c in color)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)

        if shading > 0:
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetDiffuse(0.8)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(20)
        else:
            actor.GetProperty().LightingOff()

        renderer.AddActor(actor)

        # If text is specified for the label, create a caption
        text_info = props.get("text", None)
        if text_info is not None:
            text_label = text_info.get("label", "")
            text_color = text_info.get("color", (0, 0, 0))
            text_size = text_info.get("size", 12)  # font size

            # 1) Determine the "base" position for the text based on object bounds
            a_bounds = actor.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
            a_center = (
                (a_bounds[0] + a_bounds[1]) / 2.0,
                (a_bounds[2] + a_bounds[3]) / 2.0,
                (a_bounds[4] + a_bounds[5]) / 2.0
            )
            if view_direction in ['R', 'L']:
                margin = 0.05 * (a_bounds[1] - a_bounds[0])
            elif view_direction in ['A', 'P']:
                margin = 0.05 * (a_bounds[3] - a_bounds[2])
            elif view_direction in ['S', 'I']:
                margin = 0.05 * (a_bounds[5] - a_bounds[4])
            else:
                margin = 0.0

            if view_direction == 'R':
                base_pos = (a_bounds[1] + margin, a_center[1], a_center[2])
            elif view_direction == 'L':
                base_pos = (a_bounds[0] - margin, a_center[1], a_center[2])
            elif view_direction == 'A':
                base_pos = (a_center[0], a_bounds[3] + margin, a_center[2])
            elif view_direction == 'P':
                base_pos = (a_center[0], a_bounds[2] - margin, a_center[2])
            elif view_direction == 'S':
                base_pos = (a_center[0], a_center[1], a_bounds[5] + margin)
            elif view_direction == 'I':
                base_pos = (a_center[0], a_center[1], a_bounds[4] - margin)
            else:
                base_pos = a_center

            # 2) Create a text source and update its geometry
            text_source = vtk.vtkVectorText()
            text_source.SetText(text_label)
            text_source.Update()

            # 3) Obtain the bounding box of the text (in local coordinates)
            t_bounds = text_source.GetOutput().GetBounds()
            # Compute the center of the text
            tcx = (t_bounds[0] + t_bounds[1]) / 2.0
            tcy = (t_bounds[2] + t_bounds[3]) / 2.0
            tcz = (t_bounds[4] + t_bounds[5]) / 2.0

            # 4) Create a vtkFollower, set its origin, position, scale, and color
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())

            text_actor = vtk.vtkFollower()
            text_actor.SetMapper(text_mapper)
            if max(text_color) > 1:
                text_color = tuple(c / 255.0 for c in text_color)
            text_actor.GetProperty().SetColor(*text_color)

            # Scale the text (this will multiply the text coordinates)
            text_actor.SetScale(text_size, text_size, text_size)
            # Set the text origin equal to its center so that when setting the position the center
            # matches the base position
            text_actor.SetOrigin(tcx, tcy, tcz)
            text_actor.SetPosition(base_pos)
            text_actor.SetCamera(camera)

            renderer.AddActor(text_actor)

    # Set background color
    renderer.SetBackground(background_color[0]/255.0,
                           background_color[1]/255.0,
                           background_color[2]/255.0)

    # Render the scene
    renderWindow.Render()

    # Adjust the camera based on the overall bounding box of all objects
    renderer.ResetCamera()
    prop_bounds = [0, 0, 0, 0, 0, 0]
    renderer.ComputeVisiblePropBounds(prop_bounds)
    (xmin, xmax, ymin, ymax, zmin, zmax) = prop_bounds
    center = (
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0
    )
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    L = max(dx, dy, dz)
    
    fov = camera.GetViewAngle()  # Typically ~30°
    margin_factor = 1.2
    required_distance = L / (2 * np.tan(np.deg2rad(fov / 2))) * margin_factor

    if view_direction == 'R':
        camera.SetPosition(center[0] + required_distance, center[1], center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == 'L':
        camera.SetPosition(center[0] - required_distance, center[1], center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == 'A':
        camera.SetPosition(center[0], center[1] + required_distance, center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == 'P':
        camera.SetPosition(center[0], center[1] - required_distance, center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == 'S':
        camera.SetPosition(center[0], center[1], center[2] + required_distance)
        camera.SetViewUp(0, 1, 0)
    elif view_direction == 'I':
        camera.SetPosition(center[0], center[1], center[2] - required_distance)
        camera.SetViewUp(0, 1, 0)

    camera.SetFocalPoint(center)
    renderer.ResetCameraClippingRange()

    renderWindow.Render()

    # Save the result as a PNG file
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderWindow)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_filename)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"Image saved: {output_filename}")


def preview_3d_image(
    input_path: Union[str, nib.Nifti1Image],
    output: str,
    segmentation_dict: Dict[int, Dict[str, Any]],
    view_direction: str = 'A',
    smoothing: int = 20,
    shading: int = 20,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    window_size: Tuple[int, int] = (800, 800)
) -> None:
    """
    Loads a NIfTI file, converts it to RAS, creates a 3D segmentation, and saves a PNG
    with the specified orientation. If a label contains text: { 'label': str, 'color': (r, g, b), 'size': int },
    a caption is generated with its center calculated without intermediate rendering.

    Parameters:
        input_path (Union[str, nib.Nifti1Image]): The input NIfTI file path or Nifti1Image object.
        output (str): The output PNG file name.
        segmentation_dict (Dict[int, Dict[str, Any]]): A dictionary where keys are label values
            and values are dictionaries with properties for each label.
        view_direction (str): The view direction (R, L, A, P, S, I).
        smoothing (int): The number of smoothing iterations.
        shading (int): The shading factor.
        background_color (Tuple[int, int, int]): The background color (RGB).
        window_size (Tuple[int, int]): The window size.

    Returns:
        None
    """
    # Validate input parameters
    assert exists(input_path), "File not found."
    assert segmentation_dict and len(segmentation_dict) > 0, (
        "A non-empty segmentation dictionary is required.\n"
        "Example: {1: {'color': (255, 255, 0), 'opacity': 0.4, 'text': {'label': 'aorta', 'color': (0, 0, 0), 'size': 5}}, "
        "2: {'color': (255, 0, 255), 'opacity': 1.0}}"
    )
    for lbl, props in segmentation_dict.items():
        assert 'color' in props, f"'color' key is missing for label {lbl}."
        assert 'opacity' in props, f"'opacity' key is missing for label {lbl}."
        if 'text' in props:
            assert 'label' in props['text'], f"'label' key is missing in 'text' for label {lbl}."
            assert 'color' in props['text'], f"'color' key is missing in 'text' for label {lbl}."
            assert 'size' in props['text'], f"'size' key is missing in 'text' for label {lbl}."
    if smoothing:
        assert isinstance(smoothing, int), "Smoothing iterations must be an integer."
        assert smoothing >= 0, "Smoothing iterations must be >= 0."
    if shading:
        assert isinstance(shading, int), "Shading value must be an integer."
        assert shading >= 0, "Shading value must be >= 0."
    assert view_direction in ['R', 'L', 'A', 'P', 'S', 'I'], (
        "Invalid view_direction. Allowed values: R, L, A, P, S, I."
    )
    assert background_color and len(background_color) == 3, "background_color must be an RGB tuple."
    for color_channel in background_color:
        assert 0 <= color_channel <= 255, "Each color channel must be between 0 and 255."
    for window_dim in window_size:
        assert window_dim > 0, "Window dimensions must be positive numbers."
    assert output.endswith('.png'), "Output filename must end with .png."

    # Load the NIfTI image (use canonical orientation)
    if isinstance(input_path, str):
        img = load_nifti(input_path, canonical=True, engine='nibabel')
    elif isinstance(input_path, nib.Nifti1Image):
        img = as_closest_canonical(input_path)
    else:
        raise ValueError("input_path must be a string (file path) or Nifti1Image.")

    data = img.get_fdata()
    spacing = img.header.get_zooms()
    vtk_image = _numpy_to_vtk_image_data(data, spacing=spacing)

    _render_segmentation_to_image(
        vtk_image=vtk_image,
        segmentation_dict=segmentation_dict,
        smoothing=smoothing,
        shading=shading,
        view_direction=view_direction,
        background_color=background_color,
        window_size=window_size,
        output_filename=output
    )


# ------------------ USAGE EXAMPLE -------------------
if __name__ == "__main__":
    nifti_path = "/Users/eolika/Downloads/dataset_aorta/botkin_0127/aorta.nii.gz"
    segmentation_dict_example = {
        1: {'color': (255, 255, 0), 'opacity': 0.4, 'text': {'label': 'Aorta', 'color': (0, 0, 0), 'size': 5}},
        2: {'color': (255, 0, 255), 'opacity': 1.0, 'text': {'label': 'Label2','color': (0, 0, 0), 'size': 5}},
        3: {'color': (0, 255, 0), 'opacity': 1.0},
        4: {'color': (0, 255, 255), 'opacity': 1.0},
        5: {'color': (255, 0, 0), 'opacity': 1.0},
        6: {'color': (0, 255, 0), 'opacity': 1.0},
        7: {'color': (255, 255, 255), 'opacity': 1.0},
        8: {'color': (0, 255, 255), 'opacity': 1.0},
        9: {'color': (0, 255, 255), 'opacity': 1.0},
        10: {'color': (255, 0, 255), 'opacity': 1.0},
        11: {'color': (0, 255, 0), 'opacity': 1.0},
    }
    preview_3d_image(
        input_path=nifti_path,
        output="segmentation_preview.png",
        segmentation_dict=segmentation_dict_example,
        view_direction='A',  # Possible values: R, L, A, P, S, I
        background_color=(255, 255, 255),
        smoothing=20,
        shading=20,
        window_size=(1200, 1200)
    )
