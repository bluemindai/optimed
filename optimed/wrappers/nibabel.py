import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform, io_orientation
import numpy as np
import os

from typing import Tuple


def load_nifti(file: str, canonical: bool=False) -> nib.Nifti1Image:
    """
    Load a NIfTI image from a specified file.

    Parameters:
        file : str
            The file path of the NIfTI image to be loaded. 
            The file must have a '.nii' or '.nii.gz' extension.
        canonical : bool, optional
            If True, the function will return the NIfTI image in its closest canonical orientation, 
            which is useful for standardizing the orientation of various images for further processing. 
            If False, the image will be returned in its original orientation without any adjustments.

    Raises:
        AssertionError
            If the input file does not exist, is not a file, or has an invalid extension.

    Returns:
        nib.Nifti1Image
            A NIfTI image object loaded from the specified file.
    """
    
    assert os.path.exists(file), f"No such file: {file}"
    assert os.path.isfile(file), f"Is not a file: {file}"
    assert file.endswith('.nii.gz') or file.endswith('.nii'), "Invalid NIfTI extension."

    if canonical:
        return nib.as_closest_canonical(nib.load(file))
    else:
        return nib.load(file)


def save_nifti(ndarray: np.ndarray, 
               affine: np.ndarray, 
               header: nib.Nifti1Header, 
               dtype: np.dtype, 
               save_to: str) -> None:
    """
    Save a NIfTI image to a specified file.

    Parameters:
        ndarray : np.ndarray
            A numpy array representing the image data to be saved.
        affine : np.ndarray
            A 4x4 numpy array representing the affine transformation matrix.
        header : nib.Nifti1Header
            A NIfTI header object containing metadata for the NIfTI image.
        dtype : np.dtype
            The NumPy data type to which the image data should be converted.
        save_to : str
            The file path where the NIfTI image will be saved.

    Raises:
        AssertionError
            If the input parameters do not meet the specified criteria.
        ValueError
            If the numpy array cannot be converted to the specified dtype.
    
    Returns:
        None
    """
    
    # Validate input types
    assert isinstance(ndarray, np.ndarray), "Invalid input: ndarray must be a numpy array."
    assert isinstance(affine, np.ndarray) and affine.shape == (4, 4), "Invalid affine: must be a 4x4 numpy array."
    assert isinstance(header, nib.Nifti1Header), "Invalid header: must be a Nifti1Header."
    
    valid_dtypes = [np.int16, np.int32, np.float16, np.float32, np.float64]
    assert dtype in valid_dtypes, f"Invalid dtype: must be one of {valid_dtypes}."

    try:
        converted_array = ndarray.astype(dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to convert ndarray to dtype {dtype}: {e}")
    
    new_header = header.copy()
    new_header.set_data_dtype(dtype)

    # Create Nifti image and save
    img = nib.Nifti1Image(converted_array, affine=affine, header=new_header)
    nib.save(img, save_to)


def as_closest_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Convert a NIfTI image to its closest canonical orientation.

    This function takes a NIfTI image and transforms it into a standard orientation
    that is consistent with the canonical frame of reference. This is particularly useful
    when processing multiple images with potentially different orientations, ensuring 
    that they are aligned in a uniform manner for further analysis.

    Parameters:
        img : nib.Nifti1Image
            The NIfTI image to be converted to its closest canonical orientation.

    Returns:
        nib.Nifti1Image
            A new NIfTI image object that has been transformed to the closest canonical orientation.
    """
    return nib.as_closest_canonical(img)


def undo_canonical(img_can: nib.Nifti1Image, img_orig: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Invert the canonical transformation of a NIfTI image.

    Parameters:
        img_can (nib.Nifti1Image): The canonical image to be reverted.
        img_orig (nib.Nifti1Image): The original image before canonical transformation.

    Returns:
        nib.Nifti1Image: The image reverted to its original orientation.
    """

    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")

    to_canonical = img_ornt
    from_canonical = ornt_transform(ras_ornt, img_ornt)

    return img_can.as_reoriented(from_canonical)


def maybe_convert_nifti_image_in_dtype(img: nib.Nifti1Image, 
                                        from_dtype: str, 
                                        to_dtype: str) -> nib.Nifti1Image:
    """
    Convert a NIfTI image to a specified data type if its current data type matches the source data type.

    Parameters:
        img : nib.Nifti1Image
            The NIfTI image object to be converted.
        from_dtype : str
            The current data type of the image data as a string.
        to_dtype : str
            The desired data type to convert the image data to.

    Raises:
        AssertionError
            If the provided from_dtype or to_dtype is not valid.

    Returns:
        nib.Nifti1Image
            The converted NIfTI image if conversion was performed, or the original image if no conversion was needed.
    """
    
    # Mapping from string representations to numpy data types
    dtype_mapping = {
        'int16': np.int16,
        'int32': np.int32,
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64
    }

    # Validate input dtypes
    assert from_dtype in dtype_mapping, f"Invalid from_dtype format. Must be one of {list(dtype_mapping.keys())}"
    assert to_dtype in dtype_mapping, f"Invalid to_dtype format. Must be one of {list(dtype_mapping.keys())}"

    current_dtype = str(img.header.get_data_dtype())

    # If the current data type matches the from_dtype, perform conversion
    if current_dtype == from_dtype:
        converted_data = img.get_fdata()
        new_data = converted_data.astype(dtype_mapping[to_dtype])
        new_img = nib.Nifti1Image(new_data, img.affine, img.header)
        new_img.header.set_data_dtype(to_dtype)
        return new_img 

    return img


def get_image_orientation(img: nib.Nifti1Image) -> Tuple[str, str, str]:
    """
    Retrieves the orientation of a NIfTI image based on its affine transformation matrix.

    Parameters:
    img (nib.Nifti1Image): The NIfTI image from which to extract the orientation.

    Returns:
    Tuple[str, str, str]: A tuple representing the orientation of the image 
                           in terms of the axes (e.g., ('R', 'A', 'S')).
    """

    affine = img.affine
    orientation = aff2axcodes(affine)

    return orientation


def empty_img(ref: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Create an empty NIfTI image with the same dimensions and affine transformation 
    as the reference image, filled with zeros.

    Parameters:
        ref : nib.Nifti1Image
            The reference NIfTI image from which to derive dimensions and affine.

    Returns:
        nib.Nifti1Image
            A new NIfTI image with all voxel values set to zero.
    """

    empty_data = np.zeros_like(ref.get_fdata())
    
    empty_img = nib.Nifti1Image(
        empty_data.astype(ref.get_data_dtype()), 
        affine=ref.affine, 
        header=ref.header
    )

    return empty_img
