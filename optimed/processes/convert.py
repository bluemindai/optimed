from optimed.wrappers.nifti import load_nifti
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import nrrd
from typing import Union


def convert_sitk_to_nibabel(
    sitk_image: sitk.Image
) -> nib.Nifti1Image:
    """
    Convert a SimpleITK image to a NIfTI image.

    Parameters:
        sitk_image (sitk.Image): The SimpleITK image object.

    Returns:
        nib.Nifti1Image: The NIfTI image object.
    """
    data = sitk.GetArrayFromImage(sitk_image)
    affine = sitk_image.GetOrigin() + sitk_image.GetSpacing()
    image = nib.Nifti1Image(data, affine)

    return image


def convert_nibabel_to_sitk(
    nibabel_image: nib.Nifti1Image
) -> sitk.Image:
    """
    Convert a NIfTI image to a SimpleITK image.

    Parameters:
        nibabel_image (nib.Nifti1Image): The NIfTI image object.

    Returns:
        sitk.Image: The SimpleITK image object.
    """
    data = nibabel_image.get_fdata()
    affine = nibabel_image.affine
    image = sitk.GetImageFromArray(data)
    image.SetOrigin(affine[:3, 3])
    image.SetSpacing(affine[:3, :3])

    return image
        

def convert_dcm_to_nifti(
    dicom_path: str,
    nifti_file: str,
    permute_axes: bool = False,
    return_object: bool = True,
    return_type: str = 'nibabel'
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
    assert return_type in ['nibabel', 'sitk'], 'Invalid return type.'

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
    return_type: str = 'nibabel'
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
    assert return_type in ['nibabel', 'sitk'], 'Invalid return type.'

    if isinstance(nrrd_file, str):
        data, header = nrrd.read(nrrd_file)
    else:
        data, header = nrrd_file

    # Extract the affine transformation matrix if available
    if 'space directions' in header and 'space origin' in header:
        affine = np.eye(4)
        affine[:3, :3] = header['space directions']
        affine[:3, 3] = header['space origin']
        
        # Check the space field for coordinate system information
        if 'space' in header:
            space = header['space'].lower()
            if 'left-posterior-superior' in space:
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
    return_type: str = 'nibabel'
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
    assert return_type in ['nibabel', 'sitk'], 'Invalid return type.'

    if isinstance(mha_file, str):
        image = sitk.ReadImage(mha_file)
    else:
        image = mha_file
    sitk.WriteImage(image, nifti_file)

    if return_object:
        converted_image = load_nifti(nifti_file, engine=return_type)
        return converted_image
