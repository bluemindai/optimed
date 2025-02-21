from optimed.wrappers.nifti import load_nifti
from optimed.wrappers.operations import exists
import SimpleITK as sitk
from typing import Union
import nibabel as nib
import numpy as np
import slicerio
import random
import nrrd
import gzip


def convert_sitk_to_nibabel(sitk_image: sitk.Image) -> nib.Nifti1Image:
    """
    Convert a SimpleITK image to a NIfTI image.

    Parameters:
        sitk_image (sitk.Image): The SimpleITK image object.

    Returns:
        nib.Nifti1Image: The NIfTI image object.
    """
    assert isinstance(sitk_image, sitk.Image), "Input must be a SimpleITK image."

    data = sitk.GetArrayFromImage(sitk_image)
    origin = np.array(sitk_image.GetOrigin())
    spacing = np.array(sitk_image.GetSpacing())

    # Build a 4x4 affine matrix: spacing along the diagonal, origin as translation.
    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = origin

    image = nib.Nifti1Image(data, affine)
    return image


def convert_nibabel_to_sitk(nibabel_image: nib.Nifti1Image) -> sitk.Image:
    """
    Convert a NIfTI image to a SimpleITK image.

    Parameters:
        nibabel_image (nib.Nifti1Image): The NIfTI image object.

    Returns:
        sitk.Image: The SimpleITK image object.
    """
    assert isinstance(nibabel_image, nib.Nifti1Image), "Input must be a NIfTI image."

    data = nibabel_image.get_fdata()
    affine = nibabel_image.affine

    origin = affine[:3, 3]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)

    image = sitk.GetImageFromArray(data)
    image.SetOrigin(tuple(origin))
    image.SetSpacing(tuple(spacing))

    return image


def convert_dcm_to_nifti(
    dicom_path: str,
    nifti_file: str,
    permute_axes: bool = False,
    return_object: bool = True,
    return_type: str = "nibabel",
) -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Converts a DICOM series to a NIfTI file.

    Parameters:
        dicom_path (str): Path to the input DICOM series.
        nifti_file (str): Path to the output NIfTI file.
        permute_axes (bool): If True, permute axes of the data.
        return_object (bool): If True, return the NIfTI object
        return_type (str): The return type of the function.

    Returns:
        nib.Nifti1Image: The SimpleITK image object.
    """
    assert exists(dicom_path), f"Input file not found: {dicom_path}"
    assert return_type in ["nibabel", "sitk"], "Invalid return type."

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if permute_axes:
        image = sitk.PermuteAxes(image, [2, 1, 0])
    sitk.WriteImage(image, nifti_file)

    if return_object:
        converted_image = load_nifti(nifti_file, engine=return_type)
        return converted_image


def convert_nrrd_to_nifti(
    nrrd_file: Union[str, object],
    nifti_file: str,
    return_object: bool = True,
    return_type: str = "nibabel",
) -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Convert a NRRD file to NIfTI format.

    Parameters:
        nrrd_file (Union[str, object]): Path to the input NRRD file or the data and header.
        nifti_file (str): Path to the output NIfTI file.
        return_object (bool): If True, return the NIfTI object.
        return_type (str): The return type of the function.

    Returns:
        nib.Nifti1Image: The NIfTI image object.
    """
    assert exists(nrrd_file), f"Input file not found: {nrrd_file}"
    assert return_type in ["nibabel", "sitk"], "Invalid return type."

    if isinstance(nrrd_file, str):
        data, header = nrrd.read(nrrd_file)
    else:
        data, header = nrrd_file

    # Extract the affine transformation matrix if available
    if "space directions" in header and "space origin" in header:
        affine = np.eye(4)
        affine[:3, :3] = header["space directions"]
        affine[:3, 3] = header["space origin"]

        # Check the space field for coordinate system information
        if "space" in header:
            space = header["space"].lower()
            if "left-posterior-superior" in space:
                # NIfTI uses RAS (Right-Anterior-Superior)
                # Flip the first two axes
                affine[0, :] *= -1
                affine[1, :] *= -1

    else:
        # Default to identity matrix if no affine info is available
        affine = np.eye(4)

    image = nib.Nifti1Image(data, affine)

    image.header.set_qform(affine)
    image.header.set_sform(affine)

    nib.save(image, nifti_file)

    if return_object:
        converted_image = load_nifti(nifti_file, engine=return_type)
        return converted_image


def conver_mha_to_nifti(
    mha_file: Union[str, object],
    nifti_file: str,
    return_object: bool = True,
    return_type: str = "nibabel",
) -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Convert a MHA file to NIfTI format.

    Parameters:
        mha_file (Union[str, object]): Path to the input MHA file or the SimpleITK image object.
        nifti_file (str): Path to the output NIfTI file.
        return_object (bool): If True, return the NIfTI object.
        return_type (str): The return type of the function.

    Returns:
        nib.Nifti1Image: The SimpleITK image object.
    """
    assert exists(mha_file), f"Input file not found: {mha_file}"
    assert return_type in ["nibabel", "sitk"], "Invalid return type."

    if isinstance(mha_file, str):
        image = sitk.ReadImage(mha_file)
    else:
        image = mha_file
    sitk.WriteImage(image, nifti_file)

    if return_object:
        converted_image = load_nifti(nifti_file, engine=return_type)
        return converted_image


def convert_nifti_to_nrrd(
    input_nifti_filename: str,
    output_seg_filename: str,
    segment_metadata: list = None,
    verbose=True,
) -> None:
    """
    Convert a plain NIfTI segmentation (label map) to a Slicer segmentation (.seg.nrrd)
    file with metadata. Optionally, provide segment metadata as a list of dicts.

    Parameters:
        input_nifti_filename (str):
            Path to the input NIfTI file (e.g., 'aorta.nii.gz').
        output_seg_filename (str):
            Path to the output segmentation file (e.g., 'aorta.seg.nrrd').
        segment_metadata (list of dict, optional):
            A list of dictionaries, one per segment, where each dict should contain:
                - "labelValue": int, the voxel label in the input image.
                - "name": str, human-readable segment name.
                - Optionally "color": list of 3 floats [R, G, B]. Values can be in the range 0-1 or 0-255.
                - Optionally "terminology": dict with further description.
            If not provided, metadata is auto-generated from the unique labels in the image.
        verbose (bool, optional):
            If True, print debug information.
    """
    assert exists(input_nifti_filename), f"Input file not found: {input_nifti_filename}"
    assert input_nifti_filename.endswith(".nii") or input_nifti_filename.endswith(
        ".nii.gz"
    ), "Input file must be a NIfTI file with .nii or .nii.gz extension."
    assert output_seg_filename.endswith(
        ".seg.nrrd"
    ), "Output file must be a Slicer segmentation file with .seg.nrrd extension."

    if segment_metadata is not None and not isinstance(segment_metadata, list):
        raise ValueError(
            "segment_metadata must be a list of dictionaries (example: [{'labelValue': 1, 'name': 'Segment 1'}])"
        )
    if segment_metadata:
        for seg in segment_metadata:
            if not isinstance(seg, dict):
                raise ValueError("Each segment metadata must be a dictionary.")
            if "labelValue" not in seg or "name" not in seg:
                raise ValueError(
                    "Each segment metadata dictionary must have 'labelValue' and 'name' keys."
                )
            if not isinstance(seg["labelValue"], int):
                raise ValueError(
                    "Each segment metadata 'labelValue' must be an integer."
                )
            if "color" in seg:
                if not isinstance(seg["color"], list):
                    raise ValueError(
                        "Segment metadata 'color' must be a list of 3 floats."
                    )
                if len(seg["color"]) != 3:
                    raise ValueError(
                        "Segment metadata 'color' must be a list of 3 floats."
                    )
                # Adapt color if values appear to be in the 0-255 range
                if any(c > 1 for c in seg["color"]):
                    seg["color"] = [round(c / 255.0, 2) for c in seg["color"]]

    def _random_color():
        """Generate a random RGB color with values between 0 and 1."""
        return [round(random.random(), 2) for _ in range(3)]

    def _read_nrrd_header(filename):
        """
        Read the header portion of a NRRD file.
        If the file is gzipped, use gzip to read it.
        """
        with open(filename, "rb") as f:
            magic = f.read(2)
        if magic == b"\x1f\x8b":  # gzip magic number
            open_func = gzip.open
        else:
            open_func = open
        header_lines = []
        with open_func(filename, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # NRRD header ends with an empty line
                if line.strip() == "":
                    break
                header_lines.append(line.strip())
        return header_lines

    segmentation = slicerio.read_segmentation(input_nifti_filename)

    # If no metadata provided, auto-generate it from the image labels (except background)
    if segment_metadata is None:
        img = sitk.ReadImage(input_nifti_filename)
        arr = sitk.GetArrayFromImage(img)
        unique_vals = np.unique(arr)
        if verbose:
            print("Unique label values in the input image:", unique_vals)

        segment_metadata = []
        for label in unique_vals:
            if label == 0:
                continue  # skip background
            segment_metadata.append(
                {
                    "labelValue": int(label),
                    "name": f"Segment {label}",
                    "color": _random_color(),
                }
            )
    else:
        # Ensure every segment has a "color" key; if missing, assign a random color
        for seg in segment_metadata:
            if "color" not in seg:
                seg["color"] = _random_color()

    # Assign metadata to the segmentation dictionary
    segmentation["segments"] = segment_metadata

    slicerio.write_segmentation(output_seg_filename, segmentation)
    if verbose:
        print("Conversion complete. Wrote segmentation to:", output_seg_filename)

    # Debug: Print header of the segmentation file
    if verbose:
        header_lines = _read_nrrd_header(output_seg_filename)
        print("Header of the segmentation file:")
        for line in header_lines:
            print(line)
