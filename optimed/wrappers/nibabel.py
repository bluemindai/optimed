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


def as_closest_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """

    FOR NIBABEL ONLY
    
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

    FOR NIBABEL ONLY

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


def empty_img_like(ref: nib.Nifti1Image) -> nib.Nifti1Image:
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
