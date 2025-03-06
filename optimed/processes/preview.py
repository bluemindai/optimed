import nibabel as nib
import numpy as np
import vtk
from vtk.util import numpy_support
from typing import Union, Dict, Tuple, Any, List
from optimed.wrappers.nifti import load_nifti, as_closest_canonical
from optimed.wrappers.operations import exists
import matplotlib.pyplot as plt
import math

vtk.vtkObject.GlobalWarningDisplayOff()  # Disable VTK warnings


def preview_3d_image(
    input: Union[str, nib.Nifti1Image],
    output: str,
    segmentation_dict: Dict[int, Dict[str, Any]],
    view_direction: Union[str, List[str]] = "A",
    smoothing: int = 20,
    shading: int = 20,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    window_size: Tuple[int, int] = (800, 800),
) -> None:
    """
    Loads a NIfTI file, converts it to a canonical orientation, creates a 3D segmentation
    and saves a PNG file for the specified view direction. If a text is provided for a label
    (e.g. { 'label': str, 'color': (r, g, b), 'size': int }), a caption is generated.

    If view_direction is passed as a list (e.g. ['A', 'I']), the final image is saved as subplots
    (maximum 2 images per row).

    Parameters:
        input (Union[str, nib.Nifti1Image]): Path to the NIfTI file or a Nifti1Image object.
        output (str): Output PNG filename.
        segmentation_dict (Dict[int, Dict[str, Any]]): Dictionary with parameters for each label.
        view_direction (str or List[str]): View direction (R, L, A, P, S, I) or list of directions.
        smoothing (int): Number of smoothing iterations.
        shading (int): Shading factor.
        background_color (Tuple[int, int, int]): Background color (RGB).
        window_size (Tuple[int, int]): Window size.

    Returns:
        None
    """
    # Input validation
    if isinstance(input, str):
        assert exists(input), "File not found."
    else:
        assert isinstance(input, nib.Nifti1Image), ValueError(
            "input_path must be a file path string or a Nifti1Image."
        )
    assert segmentation_dict and len(segmentation_dict) > 0, (
        "A non-empty segmentation dictionary is required.\n"
        "Example: {1: {'color': (255, 255, 0), 'opacity': 0.4, 'text': {'label': 'aorta', 'color': (0, 0, 0), 'size': 5}}, "
        "2: {'color': (255, 0, 255), 'opacity': 1.0}}"
    )
    for lbl, props in segmentation_dict.items():
        assert "color" in props, f"'color' is missing for label {lbl}."
        assert "opacity" in props, f"'opacity' is missing for label {lbl}."
        if "text" in props:
            assert (
                "label" in props["text"]
            ), f"'label' is missing in 'text' for label {lbl}."
            assert (
                "color" in props["text"]
            ), f"'color' is missing in 'text' for label {lbl}."
            assert (
                "size" in props["text"]
            ), f"'size' is missing in 'text' for label {lbl}."
    if smoothing:
        assert isinstance(smoothing, int), "Smoothing iterations must be an integer."
        assert smoothing >= 0, "Smoothing iterations must be >= 0."
    if shading:
        assert isinstance(shading, int), "Shading factor must be an integer."
        assert shading >= 0, "Shading factor must be >= 0."
    if isinstance(view_direction, list):
        for vd in view_direction:
            assert vd in [
                "R",
                "L",
                "A",
                "P",
                "S",
                "I",
            ], f"Invalid view_direction value: {vd}. Allowed values: R, L, A, P, S, I."
    else:
        assert view_direction in [
            "R",
            "L",
            "A",
            "P",
            "S",
            "I",
        ], "Invalid view_direction value. Allowed values: R, L, A, P, S, I."
    assert (
        background_color and len(background_color) == 3
    ), "background_color must be an RGB tuple."
    for color_channel in background_color:
        assert (
            0 <= color_channel <= 255
        ), "Each color channel must be between 0 and 255."
    for window_dim in window_size:
        assert window_dim > 0, "Window dimensions must be positive numbers."
    assert output.endswith(".png"), "Output filename must end with .png."

    # Load the NIfTI image (convert to canonical orientation)
    if isinstance(input, str):
        img = load_nifti(input, canonical=True, engine="nibabel")
    elif isinstance(input, nib.Nifti1Image):
        img = as_closest_canonical(input)

    data = img.get_fdata()
    spacing = img.header.get_zooms()
    vtk_img = _numpy_to_vtk_image_data(data, spacing=spacing)

    # If view_direction is a list, generate an image for each direction and combine them in subplots
    if isinstance(view_direction, list):
        images = []
        for vd in view_direction:
            img_arr = _render_segmentation_to_image(
                vtk_image=vtk_img,
                segmentation_dict=segmentation_dict,
                smoothing=smoothing,
                shading=shading,
                view_direction=vd,
                background_color=background_color,
                window_size=window_size,
                output_filename=None,
                return_image=True,
            )
            images.append(img_arr)

        n_images = len(images)
        ncols = min(2, n_images)
        nrows = math.ceil(n_images / ncols)

        # Choose figure size relative to window_size
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(window_size[0] / 100, window_size[1] / 100)
        )
        # Flatten axes array if necessary
        if n_images > 1:
            axes = np.atleast_1d(axes).flatten()
        else:
            axes = [axes]
        for ax, img in zip(axes, images):
            ax.imshow(np.flipud(img))
            ax.axis("off")
        # Hide unused subplots
        for ax in axes[len(images) :]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(output)
        plt.close()
    else:
        _render_segmentation_to_image(
            vtk_image=vtk_img,
            segmentation_dict=segmentation_dict,
            smoothing=smoothing,
            shading=shading,
            view_direction=view_direction,
            background_color=background_color,
            window_size=window_size,
            output_filename=output,
            return_image=False,
        )


def _numpy_to_vtk_image_data(
    np_data: np.ndarray, spacing=(1, 1, 1)
) -> vtk.vtkImageData:
    """
    Converts a NumPy array (in RAS orientation, axes (X, Y, Z))
    to a vtkImageData object.
    """
    x, y, z = np_data.shape
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(x, y, z)
    vtk_image.SetSpacing(spacing)

    flat_data = np.ravel(np_data, order="F")
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=flat_data, deep=True, array_type=vtk.VTK_FLOAT
    )
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image


def _render_segmentation_to_image(
    vtk_image: vtk.vtkImageData,
    segmentation_dict: Dict[int, Dict[str, Any]],
    smoothing: int = 20,
    shading: int = 20,
    view_direction: str = "A",
    background_color: Tuple[int, int, int] = (0, 0, 0),
    window_size: Tuple[int, int] = (800, 800),
    output_filename: Union[str, None] = "segmentation_preview.png",
    return_image: bool = False,
) -> Union[None, np.ndarray]:
    """
    Renders a 3D segmentation and either saves the result to a PNG file or returns
    the image as a numpy array. If a text is provided for a label, a caption is created
    with proper positioning relative to view_direction.

    Parameters:
        vtk_image (vtk.vtkImageData): Input vtkImageData object.
        segmentation_dict (Dict[int, Dict[str, Any]]): A dictionary with label values as keys
            and corresponding properties as values.
        smoothing (int): Number of smoothing iterations.
        shading (int): Shading factor.
        view_direction (str): View direction (R, L, A, P, S, I).
        background_color (Tuple[int, int, int]): Background color (RGB).
        window_size (Tuple[int, int]): Window size.
        output_filename (str or None): Output PNG filename.
        return_image (bool): If True, returns the image as a numpy array; otherwise saves the file.

    Returns:
        None or np.ndarray: Returns the image as a numpy array if return_image is True.
    """
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(*window_size)
    renderWindow.SetOffScreenRendering(True)

    camera = renderer.GetActiveCamera()

    # Creating actors for each label
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

        color = props.get("color", (1, 1, 1))
        opacity = props.get("opacity", 1.0)
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

        # If a text is provided for the label, create a caption
        text_info = props.get("text", None)
        if text_info is not None:
            text_label = text_info.get("label", "")
            text_color = text_info.get("color", (0, 0, 0))
            text_size = text_info.get("size", 12)  # font size

            # 1) Determine the base position for the text based on the actor bounds
            a_bounds = actor.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
            a_center = (
                (a_bounds[0] + a_bounds[1]) / 2.0,
                (a_bounds[2] + a_bounds[3]) / 2.0,
                (a_bounds[4] + a_bounds[5]) / 2.0,
            )
            if view_direction in ["R", "L"]:
                margin = 0.05 * (a_bounds[1] - a_bounds[0])
            elif view_direction in ["A", "P"]:
                margin = 0.05 * (a_bounds[3] - a_bounds[2])
            elif view_direction in ["S", "I"]:
                margin = 0.05 * (a_bounds[5] - a_bounds[4])
            else:
                margin = 0.0

            if view_direction == "R":
                base_pos = (a_bounds[1] + margin, a_center[1], a_center[2])
            elif view_direction == "L":
                base_pos = (a_bounds[0] - margin, a_center[1], a_center[2])
            elif view_direction == "A":
                base_pos = (a_center[0], a_bounds[3] + margin, a_center[2])
            elif view_direction == "P":
                base_pos = (a_center[0], a_bounds[2] - margin, a_center[2])
            elif view_direction == "S":
                base_pos = (a_center[0], a_center[1], a_bounds[5] + margin)
            elif view_direction == "I":
                base_pos = (a_center[0], a_center[1], a_bounds[4] - margin)
            else:
                base_pos = a_center

            # 2) Create the text source and update its geometry
            text_source = vtk.vtkVectorText()
            text_source.SetText(text_label)
            text_source.Update()

            # 3) Get the bounding box of the text (in local coordinates)
            t_bounds = text_source.GetOutput().GetBounds()
            # Calculate the text center
            tcx = (t_bounds[0] + t_bounds[1]) / 2.0
            tcy = (t_bounds[2] + t_bounds[3]) / 2.0
            tcz = (t_bounds[4] + t_bounds[5]) / 2.0

            # 4) Create a vtkFollower and set its properties
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())

            text_actor = vtk.vtkFollower()
            text_actor.SetMapper(text_mapper)
            if max(text_color) > 1:
                text_color = tuple(c / 255.0 for c in text_color)
            text_actor.GetProperty().SetColor(*text_color)

            text_actor.SetScale(text_size, text_size, text_size)
            text_actor.SetOrigin(tcx, tcy, tcz)
            text_actor.SetPosition(base_pos)
            text_actor.SetCamera(camera)

            renderer.AddActor(text_actor)

    # Set background color
    renderer.SetBackground(
        background_color[0] / 255.0,
        background_color[1] / 255.0,
        background_color[2] / 255.0,
    )

    # Render the scene
    renderWindow.Render()

    # Adjust the camera based on the overall object bounds
    renderer.ResetCamera()
    prop_bounds = [0, 0, 0, 0, 0, 0]
    renderer.ComputeVisiblePropBounds(prop_bounds)
    (xmin, xmax, ymin, ymax, zmin, zmax) = prop_bounds
    center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    L = max(dx, dy, dz)

    fov = camera.GetViewAngle()
    margin_factor = 1.2
    required_distance = L / (2 * np.tan(np.deg2rad(fov / 2))) * margin_factor

    if view_direction == "R":
        camera.SetPosition(center[0] + required_distance, center[1], center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == "L":
        camera.SetPosition(center[0] - required_distance, center[1], center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == "A":
        camera.SetPosition(center[0], center[1] + required_distance, center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == "P":
        camera.SetPosition(center[0], center[1] - required_distance, center[2])
        camera.SetViewUp(0, 0, 1)
    elif view_direction == "S":
        camera.SetPosition(center[0], center[1], center[2] + required_distance)
        camera.SetViewUp(0, 1, 0)
    elif view_direction == "I":
        camera.SetPosition(center[0], center[1], center[2] - required_distance)
        camera.SetViewUp(0, 1, 0)

    camera.SetFocalPoint(center)
    renderer.ResetCameraClippingRange()

    renderWindow.Render()

    # Capture the image from the render window
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderWindow)
    w2i.Update()

    if return_image:
        image_data = w2i.GetOutput()
        dims = image_data.GetDimensions()  # (width, height, 1)
        num_components = image_data.GetNumberOfScalarComponents()
        vtk_array = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
        # Reshape array to (height, width, num_components)
        image = vtk_array.reshape(dims[1], dims[0], num_components)
        return image
    else:
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_filename)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        return None


# ------------------ USAGE EXAMPLE -------------------
if __name__ == "__main__":
    nifti_path = "/Users/eolika/Downloads/dataset_aorta/botkin_0127/aorta.nii.gz"
    segmentation_dict_example = {
        1: {
            "color": (255, 255, 0),
            "opacity": 0.4,
            "text": {"label": "Aorta", "color": (0, 0, 0), "size": 5},
        },
        2: {
            "color": (255, 0, 255),
            "opacity": 1.0,
            "text": {"label": "Label2", "color": (0, 0, 0), "size": 5},
        },
        3: {"color": (0, 255, 0), "opacity": 1.0},
        4: {"color": (0, 255, 255), "opacity": 1.0},
        5: {"color": (255, 0, 0), "opacity": 1.0},
        6: {"color": (0, 255, 0), "opacity": 1.0},
        7: {"color": (255, 255, 255), "opacity": 1.0},
        8: {"color": (0, 255, 255), "opacity": 1.0},
        9: {"color": (0, 255, 255), "opacity": 1.0},
        10: {"color": (255, 0, 255), "opacity": 1.0},
        11: {"color": (0, 255, 0), "opacity": 1.0},
    }
    # Example with a single view direction (normal mode)
    preview_3d_image(
        input=nifti_path,
        output="segmentation_preview_single.png",
        segmentation_dict=segmentation_dict_example,
        view_direction="A",  # Allowed values: R, L, A, P, S, I
        background_color=(255, 255, 255),
        smoothing=20,
        shading=20,
        window_size=(1200, 1200),
    )

    # Example with multiple view directions (subplots)
    preview_3d_image(
        input=nifti_path,
        output="segmentation_preview_multi.png",
        segmentation_dict=segmentation_dict_example,
        view_direction=["A", "I", "R"],  # List of directions
        background_color=(255, 255, 255),
        smoothing=20,
        shading=20,
        window_size=(1200, 1200),
    )
