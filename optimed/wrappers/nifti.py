import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform, io_orientation
import SimpleITK as sitk
import numpy as np
import os

from typing import Tuple
import warnings


def load_nifti(file: str, canonical: bool=False, engine: str='nibabel') -> object:
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
        nib.Nifti1Image or sitk.Image
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


def save_nifti(ndarray: np.ndarray, 
               ref_image: object, 
               dtype: np.dtype, 
               save_to: str,
               engine: str='nibabel') -> None:
    """
    Save a NIfTI image to a specified file using either nibabel or SimpleITK, based on a reference image.

    Parameters:
        ndarray : np.ndarray
            A numpy array representing the image data to be saved.
        ref_image : object
            A reference image object from which to extract metadata. This can be either a nibabel.Nifti1Image 
            or a SimpleITK.Image.
        dtype : np.dtype
            The NumPy data type to which the image data should be converted.
        save_to : str
            The file path where the NIfTI image will be saved.
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


def as_closest_canonical(img: object, engine: str = 'nibabel') -> object:
    """
    Convert a NIfTI image to its closest canonical orientation.

    This function works for both nibabel and SimpleITK images. 
    In the case of nibabel, it uses the built-in `as_closest_canonical` function. 
    For SimpleITK, it adjusts the direction matrix to align with the canonical orientation.

    Parameters:
        img : object
            The NIfTI image to be converted. This can be either a nibabel.Nifti1Image 
            or a SimpleITK.Image object.
        engine : str, optional
            The engine to use, either 'nibabel' or 'sitk'. Defaults to 'nibabel'.

    Returns:
        object:
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


def undo_canonical(img_can: object, img_orig: object, engine: str = 'nibabel') -> object:
    """
    Invert the canonical transformation of a NIfTI image.

    This function works for both nibabel and SimpleITK images. 
    In the case of nibabel, it reverts the canonical transformation using `as_reoriented`.
    For SimpleITK, it restores the original direction matrix.

    Parameters:
        img_can (object): The canonical image to be reverted. This can be either a nibabel.Nifti1Image 
                          or a SimpleITK.Image.
        img_orig (object): The original image before canonical transformation. Can be either a 
                           nibabel.Nifti1Image or a SimpleITK.Image.
        engine (str): The engine to use, either 'nibabel' or 'sitk'. Defaults to 'nibabel'.

    Returns:
        object: The image reverted to its original orientation.
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


def maybe_convert_nifti_image_in_dtype(img: object, 
                                        to_dtype: str, 
                                        engine: str='nibabel') -> object:
    """
    Convert a NIfTI image (from nibabel or SimpleITK) to a specified data type only if its current data type 
    is different from the target data type.

    Parameters:
        img : object
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
        object
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


def get_image_orientation(img: object, engine: str = 'nibabel') -> Tuple[str, str, str]:
    """
    Retrieves the orientation of a NIfTI image based on its affine transformation matrix.
    
    Works for both nibabel and SimpleITK images.

    Parameters:
    img (object): The NIfTI image from which to extract the orientation. Can be either nibabel.Nifti1Image or SimpleITK.Image.
    engine (str): The engine used to load the image ('nibabel' or 'sitk').

    Returns:
    Tuple[str, str, str]: A tuple representing the orientation of the image 
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


def change_image_orientation(img: object, target_orientation: str = "RAS", engine: str = "nibabel") -> object:
    """
    Reorients a NIfTI image to a specified orientation (e.g., 'RAS', 'LPS', 'LAS').

    Parameters:
    img : object
        Image in the format of nibabel.Nifti1Image or SimpleITK.Image.
    target_orientation : str
        Target three-letter orientation code (e.g., 'RAS', 'LPS', 'LAS').
    engine : str
        'nibabel' or 'sitk'. Specifies how the image was loaded.
    
    Returns:
    object:
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


def empty_img_like(ref: object, engine: str = 'nibabel') -> object:
    """
    Create an empty NIfTI image with the same dimensions and affine transformation 
    as the reference image, filled with zeros. Supports both nibabel and SimpleITK.

    Parameters:
        ref : object
            The reference NIfTI image (nibabel.Nifti1Image or SimpleITK.Image) 
            from which to derive dimensions and affine.
        engine : str
            The engine used ('nibabel' or 'sitk'). Defaults to 'nibabel'.

    Returns:
        object
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
