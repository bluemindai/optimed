import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform, io_orientation
import SimpleITK as sitk
import numpy as np
import xmltodict
import os

from typing import Tuple, Union, List
import warnings


def load_nifti(file: str, canonical: bool=False, engine: str='nibabel') -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Load a NIfTI image from a specified file using either nibabel or SimpleITK.

    Parameters:
        file : str
            The file path of the NIfTI image to be loaded. 
            The file must have a '.nii' or '.nii.gz' extension.
        canonical : bool, optional
            If True, the function will return the NIfTI image in its closest canonical orientation 
            (only applicable for nibabel), which is useful for standardizing the orientation of various images 
            for further processing. If False, the image will be returned in its original orientation 
            without any adjustments.
        engine : str, optional
            The engine to use for loading the NIfTI image. Can be 'nibabel' or 'sitk'. 
            Defaults to 'nibabel'.
        
    Raises:
        AssertionError
            If the input file does not exist, is not a file, has an invalid extension, 
            or if an invalid engine is specified.
        
    Returns:
        Union[nib.Nifti1Image, sitk.Image]
            A NIfTI image object loaded from the specified file using the selected engine.
    """

    assert os.path.exists(file), f"No such file: {file}"
    assert os.path.isfile(file), f"Is not a file: {file}"
    assert file.endswith('.nii.gz') or file.endswith('.nii'), "Invalid NIfTI extension."
    assert engine in ['nibabel', 'sitk'], "Invalid engine specified. Use 'nibabel' or 'sitk'."

    if engine == 'nibabel':
        if canonical:
            return nib.as_closest_canonical(nib.load(file))
        else:
            return nib.load(file)
    elif engine == 'sitk':
        if canonical:
            warnings.warn("Canonical orientation is not supported with the 'sitk' engine. The image will be loaded in its original orientation.", UserWarning)
        return sitk.ReadImage(file)


def load_multilabel_nifti(img: Union[str, nib.Nifti1Image]) -> Tuple[nib.Nifti1Image, dict]:
    """
    Load a NIfTI image with multiple labels and extract the label map.

    Parameters:
        img : Union[str, nib.Nifti1Image]
            Path to the NIfTI image file or a nibabel.Nifti1Image object.

    Returns:
        Tuple[nib.Nifti1Image, dict]
            A tuple containing the NIfTI image and a dictionary with label IDs and names.
    """

    # Handle both file path and nifti image input
    if isinstance(img, str):
        img = load_nifti(img, engine='nibabel')
    
    ext_header = img.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"]["Label"]
    
    # If only one label, ext_header is a dict instead of a list (because of xmltodict.parse()) -> convert to list
    if isinstance(ext_header, dict):
        ext_header = [ext_header]
        
    label_map = {e["#text"]: int(e["@Key"]) for e in ext_header}
    return img, label_map


def save_nifti(ndarray: np.ndarray, 
               ref_image: Union[nib.Nifti1Image, sitk.Image], 
               save_to: str,
               dtype: np.dtype = np.uint8,
               engine: str='nibabel') -> None:
    """
    Save a NIfTI image to a specified file using either nibabel or SimpleITK, based on a reference image.

    Parameters:
        ndarray : np.ndarray
            A numpy array representing the image data to be saved.
        ref_image : Union[nib.Nifti1Image, sitk.Image]
            A reference image object from which to extract metadata. This can be either a nibabel.Nifti1Image 
            or a SimpleITK.Image.
        save_to : str
            The file path where the NIfTI image will be saved.
        dtype : np.dtype, optional
            The NumPy data type to which the image data should be converted. 
            Defaults to np.uint8.
        engine : str, optional
            The engine to use for saving the NIfTI image. Can be 'nibabel' or 'sitk'. 
            Defaults to 'nibabel'.

    Raises:
        AssertionError
            If the input parameters do not meet the specified criteria.
        ValueError
            If the numpy array cannot be converted to the specified dtype.
        TypeError
            If the reference image is of an unsupported type.
    
    Returns:
        None
    """
    
    # Validate input types
    assert isinstance(ndarray, np.ndarray), "Invalid input: ndarray must be a numpy array."
    assert engine in ['nibabel', 'sitk'], "Invalid engine specified. Use 'nibabel' or 'sitk'."
    
    valid_dtypes = [np.uint8, np.int16, np.int32, np.float16, np.float32, np.float64]
    assert dtype in valid_dtypes, f"Invalid dtype: must be one of {valid_dtypes}."

    try:
        converted_array = ndarray.astype(dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to convert ndarray to dtype {dtype}: {e}")

    if engine == 'nibabel':
        if not isinstance(ref_image, nib.Nifti1Image):
            raise TypeError("For nibabel, ref_image must be a nibabel.Nifti1Image object.")
        
        affine = ref_image.affine
        header = ref_image.header.copy()
        header.set_data_dtype(dtype)

        img = nib.Nifti1Image(converted_array, affine=affine, header=header)
        nib.save(img, save_to)
    
    elif engine == 'sitk':
        if not isinstance(ref_image, sitk.Image):
            raise TypeError("For sitk, ref_image must be a SimpleITK.Image object.")
        
        if dtype == np.float16:
            warnings.warn("SimpleITK does not support float16. Converting to float32.", UserWarning)
            converted_array = converted_array.astype(np.float32)
        
        origin = ref_image.GetOrigin()
        direction = ref_image.GetDirection()

        img = sitk.GetImageFromArray(converted_array)
        img.SetOrigin(origin)
        img.SetDirection(direction)
        sitk.WriteImage(img, save_to)


def save_multilabel_nifti(ndarray: np.ndarray,
                            ref_image: nib.Nifti1Image,
                            metadata: dict,
                            save_to: str,
                            dtype: np.dtype = np.uint8) -> None:
    """
    Save a NIfTI image with multiple labels to a specified file using nibabel.

    Parameters:
        ndarray : np.ndarray
            A numpy array representing the image data to be saved.
        ref_image : nibabel.Nifti1Image
            A reference image object from which to extract metadata.
        metadata : dict
           A dictionary generated by dcmqi (https://qiicr.org/dcmqi/#/seg) containing segmentation metadata
        save_to : str
            The file path where the NIfTI image will be saved.
        dtype : np.dtype, optional
            The NumPy data type to which the image data should be converted. 
            Defaults to np.uint8.
    """
    assert isinstance(ndarray, np.ndarray), "Invalid input: ndarray must be a numpy array."

    if not isinstance(ref_image, nib.Nifti1Image):
        raise TypeError("Only nibabel.Nifti1Image is supported.")
    
    valid_dtypes = [np.uint8, np.int16, np.int32, np.float16, np.float32, np.float64]
    assert dtype in valid_dtypes, f"Invalid dtype: must be one of {valid_dtypes}."

    try:
        converted_array = ndarray.astype(dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to convert ndarray to dtype {dtype}: {e}")
    
    affine = ref_image.affine
    header = ref_image.header.copy()
    header.set_data_dtype(dtype)

    img = nib.Nifti1Image(converted_array, affine=affine, header=header)
    img = add_metadata_to_nifti(img, metadata)
    nib.save(img, save_to)


def add_metadata_to_nifti(img_in: nib.Nifti1Image, meta: dict):
    """
    Attach segmentation label/color information from meta to the NIfTI image header.
    
    Parameters
    ----------
    img_in : nibabel.Nifti1Image
        NIfTI image to which we will add the label map extension.
    meta : dict
        A dictionary generated by dcmqi (https://qiicr.org/dcmqi/#/seg) containing 
        segmentation metadata. This dictionary includes a list of segment definitions, 
        where each definition contains:
        - labelID (int): The label identifier.
        - SegmentDescription (str): A description of the segment.
        - recommendedDisplayRGBValue (list of int): A list of three integers representing 
            the recommended display color in RGB format, with values ranging from 0 to 255.
    
    Returns
    -------
    nibabel.Nifti1Image
        The same NIfTI image with its header extended by the label metadata.
    """

    assert isinstance(img_in, nib.Nifti1Image), f"Input must be a nibabel.Nifti1Image, got {type(img_in)}"

    data = img_in.get_fdata()
    valid_labels = set(np.unique(data).astype(int))

    # XML preamble
    xml_pre = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        ' <CaretExtension>'
        '  <Date><![CDATA[2013-07-14T05:45:09]]></Date>'
        '   <VolumeInformation Index="0">'
        '   <LabelTable>'
    )

    body = ""

    # segmentAttributes is typically a list of lists; iterate over each sub-list and segment
    for segment_list in meta["segmentAttributes"]:
        for seg_info in segment_list:
            label_id = seg_info["labelID"]

            # skip if the label is not present in the image
            if label_id not in valid_labels:
                continue

            label_name = seg_info["SegmentDescription"]
            # Convert [0..255] into [0..1]
            rgb = [c / 255.0 for c in seg_info["recommendedDisplayRGBValue"]]

            body += (
                f'<Label Key="{label_id}" '
                f'Red="{rgb[0]:.3f}" Green="{rgb[1]:.3f}" Blue="{rgb[2]:.3f}" '
                'Alpha="1">'
                f'<![CDATA[{label_name}]]>'
                '</Label>\n'
            )

    # Close out the XML
    xml_post = (
        '  </LabelTable>'
        '  <StudyMetaDataLinkSet>'
        '  </StudyMetaDataLinkSet>'
        '  <VolumeType><![CDATA[Label]]></VolumeType>'
        '   </VolumeInformation>'
        '</CaretExtension>'
    )

    # Combine everything into one XML string
    xml_str = xml_pre + "\n" + body + "\n" + xml_post + "\n"

    img_in.header.extensions.append(
        nib.nifti1.Nifti1Extension(0, xml_str.encode("utf-8"))
    )

    return img_in


def sitk_to_nibabel(img_sitk: sitk.Image) -> nib.Nifti1Image:
    """
    Converts a SimpleITK.Image to a nibabel.Nifti1Image, correctly forming the affine matrix.
    """
    data = sitk.GetArrayFromImage(img_sitk)  # shape: [D, H, W]

    # Extract parameters from SimpleITK
    origin = img_sitk.GetOrigin()      # (Ox, Oy, Oz)
    spacing = img_sitk.GetSpacing()    # (Sx, Sy, Sz)
    direction = img_sitk.GetDirection()  # length 9 (3x3)

    # Create the affine matrix:
    # direction = [d00, d01, d02, d10, d11, d12, d20, d21, d22]
    # We need a 4x4 matrix:
    # [ d00*Sx, d01*Sy, d02*Sz, Ox ]
    # [ d10*Sx, d11*Sy, d12*Sz, Oy ]
    # [ d20*Sx, d21*Sy, d22*Sz, Oz ]
    # [   0,      0,      0,    1 ]
    affine = np.array([
        [direction[0] * spacing[0], direction[3] * spacing[1], direction[6] * spacing[2], origin[0]],
        [direction[1] * spacing[0], direction[4] * spacing[1], direction[7] * spacing[2], origin[1]],
        [direction[2] * spacing[0], direction[5] * spacing[1], direction[8] * spacing[2], origin[2]],
        [0,                        0,                        0,                        1]
    ], dtype=np.float64)

    img_nib = nib.Nifti1Image(data, affine)
    return img_nib


def nibabel_to_sitk(img_nib: nib.Nifti1Image) -> sitk.Image:
    """
    Converts a nibabel.Nifti1Image to a SimpleITK.Image, restoring origin, spacing, and direction.
    """
    data = img_nib.get_fdata(dtype=np.float32)
    affine = img_nib.affine

    # Extract spacing, direction, and origin from the affine matrix:
    origin = affine[:3, 3]

    # Extract direction vectors
    dxx, dxy, dxz = affine[0, 0], affine[0, 1], affine[0, 2]
    dyx, dyy, dyz = affine[1, 0], affine[1, 1], affine[1, 2]
    dzx, dzy, dzz = affine[2, 0], affine[2, 1], affine[2, 2]

    # Calculate spacing as the magnitude of the vectors
    sx = np.sqrt(dxx**2 + dyx**2 + dzx**2)
    sy = np.sqrt(dxy**2 + dyy**2 + dzy**2)
    sz = np.sqrt(dxz**2 + dyz**2 + dzz**2)

    spacing = (sx, sy, sz)

    # Normalize direction vectors
    if sx != 0:
        dxx /= sx
        dyx /= sx
        dzx /= sx
    if sy != 0:
        dxy /= sy
        dyy /= sy
        dzy /= sy
    if sz != 0:
        dxz /= sz
        dyz /= sz
        dzz /= sz

    direction = [dxx, dxy, dxz,
                 dyx, dyy, dyz,
                 dzx, dzy, dzz]

    img_sitk = sitk.GetImageFromArray(data)  # SimpleITK always uses [z, y, x]
    img_sitk.SetOrigin(origin)
    img_sitk.SetSpacing(spacing)
    img_sitk.SetDirection(direction)

    return img_sitk


def as_closest_canonical(img: Union[nib.Nifti1Image, sitk.Image], engine: str = 'nibabel') -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Convert a NIfTI image to its closest canonical orientation.

    This function works for both nibabel and SimpleITK images. 
    In the case of nibabel, it uses the built-in `as_closest_canonical` function. 
    For SimpleITK, it adjusts the direction matrix to align with the canonical orientation.

    Parameters:
        img : Union[nib.Nifti1Image, sitk.Image]
            The NIfTI image to be converted. This can be either a nibabel.Nifti1Image 
            or a SimpleITK.Image object.
        engine : str, optional
            The engine to use, either 'nibabel' or 'sitk'. Defaults to 'nibabel'.

    Returns:
        Union[nib.Nifti1Image, sitk.Image]
            A new image that has been transformed to the closest canonical orientation.
    """

    if engine == 'nibabel':
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError("Expected a nibabel.Nifti1Image for engine 'nibabel'")
        return nib.as_closest_canonical(img)

    elif engine == 'sitk':
        if not isinstance(img, sitk.Image):
            raise TypeError("Expected a SimpleITK.Image for engine 'sitk'")

        # For SimpleITK, we assume the canonical orientation is RAS (Right-Anterior-Superior).
        # This involves setting the direction matrix to the identity matrix.
        img_canonical = sitk.Image(img)
        identity_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # RAS direction matrix
        img_canonical.SetDirection(identity_direction)

        return img_canonical

    else:
        raise ValueError("Unsupported engine. Use 'nibabel' or 'sitk'.")


def undo_canonical(img_can: Union[nib.Nifti1Image, sitk.Image], img_orig: Union[nib.Nifti1Image, sitk.Image], engine: str = 'nibabel') -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Invert the canonical transformation of a NIfTI image.

    This function works for both nibabel and SimpleITK images. 
    In the case of nibabel, it reverts the canonical transformation using `as_reoriented`.
    For SimpleITK, it restores the original direction matrix.

    Parameters:
        img_can : Union[nib.Nifti1Image, sitk.Image]
            The canonical image to be reverted. This can be either a nibabel.Nifti1Image 
            or a SimpleITK.Image.
        img_orig : Union[nib.Nifti1Image, sitk.Image]
            The original image before canonical transformation. Can be either a 
            nibabel.Nifti1Image or a SimpleITK.Image.
        engine : str
            The engine to use, either 'nibabel' or 'sitk'. Defaults to 'nibabel'.

    Returns:
        Union[nib.Nifti1Image, sitk.Image]
            The image reverted to its original orientation.
    """

    if engine == 'nibabel':
        if not isinstance(img_can, nib.Nifti1Image) or not isinstance(img_orig, nib.Nifti1Image):
            raise TypeError("Expected both images to be nibabel.Nifti1Image for engine 'nibabel'")

        img_ornt = io_orientation(img_orig.affine)
        ras_ornt = axcodes2ornt("RAS")

        from_canonical = ornt_transform(ras_ornt, img_ornt)

        return img_can.as_reoriented(from_canonical)

    elif engine == 'sitk':
        if not isinstance(img_can, sitk.Image) or not isinstance(img_orig, sitk.Image):
            raise TypeError("Expected both images to be SimpleITK.Image for engine 'sitk'")

        img_reverted = sitk.Image(img_can)

        # Restore the original direction matrix from the original image
        img_reverted.SetDirection(img_orig.GetDirection())

        return img_reverted

    else:
        raise ValueError("Unsupported engine. Use 'nibabel' or 'sitk'.")


def maybe_convert_nifti_image_in_dtype(img: Union[nib.Nifti1Image, sitk.Image], 
                                       to_dtype: str, 
                                       engine: str='nibabel') -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Convert a NIfTI image (from nibabel or SimpleITK) to a specified data type only if its current data type 
    is different from the target data type.

    Parameters:
        img : Union[nib.Nifti1Image, sitk.Image]
            The NIfTI image object to be converted. Can be either nibabel.Nifti1Image or SimpleITK.Image.
        to_dtype : str
            The desired data type to convert the image data to.
        engine : str, optional
            The engine of the image. Can be 'nibabel' or 'sitk'. Defaults to 'nibabel'.

    Raises:
        AssertionError
            If the provided to_dtype is not valid.
        TypeError
            If the provided image is not of the expected type.

    Returns:
        Union[nib.Nifti1Image, sitk.Image]
            The converted image if conversion was performed, or the original image if no conversion was needed.
    """

    dtype_mapping = {
        'int16': np.int16,
        'int32': np.int32,
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64,
        'uint8': np.uint8
    }

    assert to_dtype in dtype_mapping, f"Invalid to_dtype format. Must be one of {list(dtype_mapping.keys())}"

    if engine == 'nibabel':
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError("Expected a nibabel.Nifti1Image for engine 'nibabel'")

        current_dtype = str(img.header.get_data_dtype())
        if current_dtype == to_dtype:
            return img  # No conversion needed

        converted_data = img.get_fdata().astype(dtype_mapping[to_dtype])
        new_img = nib.Nifti1Image(converted_data, img.affine, img.header)
        new_img.header.set_data_dtype(to_dtype)
        return new_img

    elif engine == 'sitk':
        if not isinstance(img, sitk.Image):
            raise TypeError("Expected a SimpleITK.Image for engine 'sitk'")

        sitk_dtype_mapping = {
            '16-bit signed integer': 'int16',
            '32-bit signed integer': 'int32',
            '32-bit float': 'float32',
            '64-bit float': 'float64',
            '8-bit unsigned integer': 'uint8'
        }

        current_dtype = img.GetPixelIDTypeAsString()
        mapped_current_dtype = sitk_dtype_mapping.get(current_dtype)

        if mapped_current_dtype == to_dtype:
            return img  # No conversion needed

        converted_data = sitk.GetArrayFromImage(img).astype(dtype_mapping[to_dtype])
        new_img = sitk.GetImageFromArray(converted_data)
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetDirection(img.GetDirection())
        return new_img

    return img


def get_image_orientation(img: Union[nib.Nifti1Image, sitk.Image], engine: str = 'nibabel') -> Tuple[str, str, str]:
    """
    Retrieves the orientation of a NIfTI image based on its affine transformation matrix.
    
    Works for both nibabel and SimpleITK images.

    Parameters:
        img : Union[nib.Nifti1Image, sitk.Image]
            The NIfTI image from which to extract the orientation. Can be either nibabel.Nifti1Image or SimpleITK.Image.
        engine : str
            The engine used to load the image ('nibabel' or 'sitk').

    Returns:
        Tuple[str, str, str]
            A tuple representing the orientation of the image 
            in terms of the axes (e.g., ('R', 'A', 'S')).
    """

    if engine == 'nibabel':
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError("Expected a nibabel.Nifti1Image for engine 'nibabel'")

        affine = img.affine

    elif engine == 'sitk':
        if not isinstance(img, sitk.Image):
            raise TypeError("Expected a SimpleITK.Image for engine 'sitk'")

        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = img.GetDirection()

        affine = [[direction[0] * spacing[0], direction[3] * spacing[1], direction[6] * spacing[2], origin[0]],
                  [direction[1] * spacing[0], direction[4] * spacing[1], direction[7] * spacing[2], origin[1]],
                  [direction[2] * spacing[0], direction[5] * spacing[1], direction[8] * spacing[2], origin[2]],
                  [0, 0, 0, 1]]

    else:
        raise ValueError("Unsupported engine. Use 'nibabel' or 'sitk'.")

    orientation = aff2axcodes(affine)

    return orientation


def change_image_orientation(img: Union[nib.Nifti1Image, sitk.Image], target_orientation: str = "RAS", engine: str = "nibabel") -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Reorients a NIfTI image to a specified orientation (e.g., 'RAS', 'LPS', 'LAS').

    Parameters:
        img : Union[nib.Nifti1Image, sitk.Image]
            Image in the format of nibabel.Nifti1Image or SimpleITK.Image.
        target_orientation : str
            Target three-letter orientation code (e.g., 'RAS', 'LPS', 'LAS').
        engine : str
            'nibabel' or 'sitk'. Specifies how the image was loaded.
    
    Returns:
        Union[nib.Nifti1Image, sitk.Image]
            The reoriented image (same type as the input image).
    """
    if engine == "nibabel":
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError("For engine='nibabel', the input must be a nibabel.Nifti1Image.")

        # Current orientation
        current_ornt = io_orientation(img.affine)
        # Target orientation in ornt format
        desired_ornt = axcodes2ornt(target_orientation)
        # Transformation matrix
        transform = ornt_transform(current_ornt, desired_ornt)
        # Apply transformation
        reoriented_img = img.as_reoriented(transform)

        return reoriented_img

    elif engine == "sitk":
        if not isinstance(img, sitk.Image):
            raise TypeError("For engine='sitk', the input must be a SimpleITK.Image.")

        # 1) Convert SimpleITK.Image -> nibabel.Nifti1Image
        img_nib = sitk_to_nibabel(img)

        # 2) Reorient using nibabel
        current_ornt = io_orientation(img_nib.affine)
        desired_ornt = axcodes2ornt(target_orientation)
        transform = ornt_transform(current_ornt, desired_ornt)
        img_nib_reoriented = img_nib.as_reoriented(transform)

        # 3) Convert the result back to SimpleITK.Image
        img_sitk_reoriented = nibabel_to_sitk(img_nib_reoriented)
        return img_sitk_reoriented

    else:
        raise ValueError("Supported engines are 'nibabel' and 'sitk'.")


def empty_img_like(ref: Union[nib.Nifti1Image, sitk.Image], engine: str = 'nibabel') -> Union[nib.Nifti1Image, sitk.Image]:
    """
    Create an empty NIfTI image with the same dimensions and affine transformation 
    as the reference image, filled with zeros. Supports both nibabel and SimpleITK.

    Parameters:
        ref : Union[nib.Nifti1Image, sitk.Image]
            The reference NIfTI image (nibabel.Nifti1Image or SimpleITK.Image) 
            from which to derive dimensions and affine.
        engine : str
            The engine used ('nibabel' or 'sitk'). Defaults to 'nibabel'.

    Returns:
        Union[nib.Nifti1Image, sitk.Image]
            A new NIfTI image with all voxel values set to zero.
    """

    if engine == 'nibabel':
        if not isinstance(ref, nib.Nifti1Image):
            raise TypeError("Expected a nibabel.Nifti1Image for engine 'nibabel'")

        empty_data = np.zeros_like(ref.get_fdata())
        empty_img = nib.Nifti1Image(
            empty_data.astype(ref.get_data_dtype()), 
            affine=ref.affine, 
            header=ref.header
        )
        return empty_img

    elif engine == 'sitk':
        if not isinstance(ref, sitk.Image):
            raise TypeError("Expected a SimpleITK.Image for engine 'sitk'")

        empty_data = np.zeros(ref.GetSize()[::-1], dtype=sitk.GetArrayFromImage(ref).dtype)
        
        empty_img = sitk.GetImageFromArray(empty_data)
        
        empty_img.SetOrigin(ref.GetOrigin())
        empty_img.SetSpacing(ref.GetSpacing())
        empty_img.SetDirection(ref.GetDirection())
        
        return empty_img

    else:
        raise ValueError("Unsupported engine. Use 'nibabel' or 'sitk'.")


def split_image(
    img: Union[nib.Nifti1Image, sitk.Image], 
    parts: int, 
    axis: int = 0,
) -> List[Union[nib.Nifti1Image, sitk.Image]]:
    """
    Split an image (nibabel or SimpleITK) into multiple parts along a specified axis.

    Parameters
    ----------
    img : Union[nib.Nifti1Image, sitk.Image]
        The NIfTI/ITK image to be split.
    parts : int
        Number of parts to split the image into.
    axis : int, optional
        The axis along which to split. Default is 0 (the first dimension).
        - For nibabel, axis=0 often corresponds to the x-dimension in 3D data.
        - For SimpleITK, axis=0 often corresponds to the z-dimension in 3D data.

    Returns
    -------
    List[Union[nib.Nifti1Image, sitk.Image]]
        A list of sub-images, each representing part of the original image.
        - If the input is nibabel, the output list contains nib.Nifti1Image objects.
        - If the input is SimpleITK, the output list contains sitk.Image objects.
    """
    if parts <= 0:
        raise ValueError("Number of parts must be greater than 0.")

    # --- Nibabel
    if isinstance(img, nib.Nifti1Image):
        # Preserve original dtype instead of forcing float64
        data = np.asanyarray(img.dataobj)
        shape = data.shape

        if axis < 0 or axis >= len(shape):
            raise ValueError(f"Axis {axis} is out of range for data shape {shape}.")

        axis_length = shape[axis]
        # Calculate boundaries (e.g. if axis_length=10 and parts=3 => [0,3,6,10])
        boundaries = np.linspace(0, axis_length, parts + 1, dtype=int)

        orig_affine = img.affine.copy()
        # For convenience, separate rotation/shear from translation
        R = orig_affine[:3, :3]
        t_orig = orig_affine[:3, 3]

        sub_images = []
        for i in range(parts):
            start_idx = boundaries[i]
            end_idx   = boundaries[i+1]

            # Build a slice object to split along `axis`
            # e.g. if axis=0, we slice [start_idx:end_idx, :, :, ...]
            slice_obj = [slice(None)] * len(shape)
            slice_obj[axis] = slice(start_idx, end_idx)
            slice_obj = tuple(slice_obj)

            sub_data = data[slice_obj]

            new_affine = orig_affine.copy()
            
            # offset_vec: how many voxels we moved along 'axis'
            offset_vec = np.zeros(3)
            # If axis < 3, we can apply an offset in real space
            # (for 4D images, axis=3 might be time, which doesn't affect spatial offset)
            if axis < 3:
                offset_vec[axis] = start_idx
                delta_t = R @ offset_vec
                new_affine[:3, 3] = t_orig + delta_t

            new_header = img.header.copy()
            new_header.set_data_shape(sub_data.shape)
            sub_img = nib.Nifti1Image(sub_data, affine=new_affine, header=new_header)
            sub_images.append(sub_img)

        return sub_images

    # --- SimpleITK
    elif isinstance(img, sitk.Image):
        # SITK array is typically in [z, y, x] order for a 3D image,
        # so axis=0 = z, axis=1 = y, axis=2 = x (by default).
        data = sitk.GetArrayFromImage(img)
        shape = data.shape
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"Axis {axis} is out of range for data shape {shape}.")

        axis_length = shape[axis]
        boundaries = np.linspace(0, axis_length, parts + 1, dtype=int)

        origin = np.array(img.GetOrigin())        # (Ox, Oy, Oz) in 3D
        spacing = np.array(img.GetSpacing())      # (Sz, Sy, Sx) in 3D
        direction = np.array(img.GetDirection())  # flattened direction cosines
        dim = img.GetDimension()

        # direction matrix: e.g., if 3D => reshape to (3,3)
        dir_mat = direction.reshape(dim, dim)

        sub_images = []
        for i in range(parts):
            start_idx = boundaries[i]
            end_idx   = boundaries[i+1]

            slice_obj = [slice(None)] * len(shape)
            slice_obj[axis] = slice(start_idx, end_idx)
            slice_obj = tuple(slice_obj)

            sub_data = data[slice_obj]

            sub_img = sitk.GetImageFromArray(sub_data)


            # The offset in voxel space along `axis` is `start_idx`
            # Convert that to a physical offset by direction * spacing
            # SITK's axis=0 => row in direction matrix = dir_mat[:, 0], etc.
            new_origin = origin.copy()
            if axis < dim:  # only apply offset if axis < 3 in a 3D scenario
                offset_vec = dir_mat[:, axis] * (start_idx * spacing[axis])
                new_origin = origin + offset_vec

            sub_img.SetSpacing(tuple(spacing))
            sub_img.SetDirection(tuple(direction))
            sub_img.SetOrigin(tuple(new_origin))

            sub_images.append(sub_img)

        return sub_images

    else:
        raise TypeError("Unsupported image type. Please provide a nib.Nifti1Image or a sitk.Image.")
