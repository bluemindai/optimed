from optimed.wrappers.nifti import load_nifti
import nibabel as nib
import numpy as np
from typing import Union

# Some code snippets are taken from TotalSegmentator (https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/cropping.py)


def get_bbox_from_mask(
    mask: np.ndarray, outside_value: int = -900, addon: int = 0, verbose: bool = True
) -> list:
    """
    Get bounding box coordinates from a mask.

    Parameters:
        mask (np.ndarray): The mask array.
        outside_value (int): The value outside the region of interest. Default is -900.
        addon (int): Additional margin to add to the bounding box. Default is 0.
        verbose (bool): Verbose. Default is True.

    Returns:
        list: The bounding box coordinates.
    """
    if type(addon) is int:
        addon = [addon] * 3
    if (mask > outside_value).sum() == 0:
        if verbose:
            print("WARNING: Could not crop because no foreground detected")
        minzidx, maxzidx = 0, mask.shape[0]
        minxidx, maxxidx = 0, mask.shape[1]
        minyidx, maxyidx = 0, mask.shape[2]
    else:
        mask_voxel_coords = np.where(mask > outside_value)
        minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
        minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
        minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

    # Avoid bbox to get out of image size
    s = mask.shape
    minzidx = max(0, minzidx)
    maxzidx = min(s[0], maxzidx)
    minxidx = max(0, minxidx)
    maxxidx = min(s[1], maxxidx)
    minyidx = max(0, minyidx)
    maxyidx = min(s[2], maxyidx)

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(
    data_or_image: Union[np.ndarray, nib.Nifti1Image],
    bbox: list,
    dtype: np.dtype = None,
) -> Union[np.ndarray, nib.Nifti1Image]:
    """
    Crop either a NumPy array or a NIfTI image to a bounding box and adapt the affine if needed.

    Parameters:
        data_or_image (np.ndarray or nib.Nifti1Image): The input data or image.
        bbox (list): The bounding box coordinates.
        dtype (np.dtype): The data type for the output image.

    Returns:
        np.ndarray or nib.Nifti1Image: The cropped data or image.
    """
    if isinstance(data_or_image, nib.Nifti1Image):
        data = data_or_image.get_fdata()
        data_cropped = data[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ]
        affine = np.copy(data_or_image.affine)
        affine[:3, 3] = np.dot(
            affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1])
        )[:3]
        out_dtype = data_or_image.dataobj.dtype if dtype is None else dtype
        return nib.Nifti1Image(data_cropped.astype(out_dtype), affine)
    else:
        assert data_or_image.ndim == 3, "only supports 3d images"
        return data_or_image[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ]


def crop_to_mask(
    img_in: Union[str, nib.Nifti1Image],
    mask_img: Union[str, nib.Nifti1Image],
    addon: list = [0, 0, 0],
    dtype: np.dtype = None,
    save_to: str = None,
    verbose: bool = False,
) -> tuple:
    """
    Crop a NIfTI image to a mask and adapt the affine accordingly.

    Parameters:
        img_in (str or nib.Nifti1Image): The input NIfTI image.
        mask_img (str or nib.Nifti1Image): The mask image.
        addon (list): Additional margin to add to the bounding box.
        dtype (np.dtype): The data type for the output image.
        save_to (str): If provided, save the cropped image to this path.
        verbose (bool): If True, print progress information.

    Returns:
        tuple: The cropped NIfTI image and the bounding box coordinates.
    """

    if isinstance(img_in, str):
        img_in = load_nifti(img_in, engine="nibabel")
    if isinstance(mask_img, str):
        mask_img = load_nifti(mask_img, engine="nibabel")

    mask = mask_img.get_fdata()

    addon = (np.array(addon) / img_in.header.get_zooms()).astype(int)  # mm to voxels
    bbox = get_bbox_from_mask(mask, outside_value=0, addon=addon, verbose=verbose)

    img_out = crop_to_bbox(img_in, bbox, dtype)

    if save_to is not None:
        nib.save(img_out, save_to)

    return img_out, bbox


def undo_crop(
    img: Union[str, nib.Nifti1Image],
    ref_img: Union[str, nib.Nifti1Image],
    bbox: list,
    save_to: str = None,
) -> nib.Nifti1Image:
    """
    Fit the image which was cropped by bbox back into the shape of ref_img.

    Parameters:
        img (str or nib.Nifti1Image): The cropped image.
        ref_img (str or nib.Nifti1Image): The reference image.
        bbox (list): The bounding box coordinates.
        save_to (str): If provided, save the fitted image to this path.

    Returns:
        nib.Nifti1Image: The image fitted back into the original shape.
    """
    if isinstance(img, str):
        img = load_nifti(img, engine="nibabel")
    if isinstance(ref_img, str):
        ref_img = load_nifti(ref_img, engine="nibabel")

    img_out = np.zeros(ref_img.shape)
    img_out[
        bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
    ] = img.get_fdata()

    img_out = nib.Nifti1Image(img_out, ref_img.affine)

    if save_to is not None:
        nib.save(img_out, save_to)
    return img_out


def crop_by_xyz_boudaries(
    img_in: Union[nib.Nifti1Image, str],
    save_to: str = None,
    x_start: int = 0,
    x_end: int = 512,
    y_start: int = 0,
    y_end: int = 512,
    z_start: int = 0,
    z_end: int = 50,
    dtype=np.int16,
) -> nib.Nifti1Image:
    """
    Crop a NIfTI image or a file path to an image along the x, y, and z axes.
    If save_to is provided, save the result.

    Parameters:
        img_in (nib.Nifti1Image or str): The input NIfTI image or file path.
        save_to (str): If provided, save the cropped image to this path.
        x_start (int): The starting index along the x-axis.
        x_end (int): The ending index along the x-axis.
        y_start (int): The starting index along the y-axis.
        y_end (int): The ending index along the y-axis.
        z_start (int): The starting index along the z-axis.
        z_end (int): The ending index along the z-axis.
        dtype (np.dtype): The data type for the output image.

    Returns:
        nib.Nifti1Image: The cropped NIfTI image.
    """
    if isinstance(img_in, str):
        img_in = load_nifti(img_in, canonical=True, engine="nibabel")

    data = img_in.get_fdata()
    affine = img_in.affine
    cropped_data = data[x_start:x_end, y_start:y_end, z_start:z_end]
    cropped_img = nib.Nifti1Image(cropped_data.astype(dtype), affine)

    if save_to:
        nib.save(cropped_img, save_to)
    return cropped_img
