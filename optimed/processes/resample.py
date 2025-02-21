from optimed import _cupy_available, _cucim_available
from joblib import Parallel, delayed
from scipy import ndimage
import nibabel as nib
import numpy as np
import psutil

# Some code snippets are taken from TotalSegmentator (https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/resampling.py)


def change_spacing(
    img_in: nib.Nifti1Image,
    new_spacing: float = 1.25,
    target_shape: tuple = None,
    order: int = 0,
    nr_cpus: int = 1,
    dtype: np.dtype = None,
    remove_negative: bool = False,
    force_affine: np.ndarray = None,
    use_gpu: bool = True,
) -> nib.Nifti1Image:
    """
    Resample a NIfTI image to a new spacing or target shape.

    Parameters:
        img_in (nib.Nifti1Image): The input NIfTI image to be resampled.
        new_spacing (float or sequence of float): The new spacing to which the image should be resampled.
        target_shape (sequence of int, optional): The target shape for the output image.
        order (int, optional): Resampling order. Default is 0.
        nr_cpus (int, optional): Number of CPU cores to use for resampling. Default is 1.
        dtype (np.dtype, optional): Output data type.
        remove_negative (bool, optional): If True, set all negative values to 0.
        force_affine (np.ndarray, optional): If provided, this affine will be used for the output image.
        use_gpu (bool, optional): If True, use GPU for resampling.

    Returns:
        nib.Nifti1Image: The resampled NIfTI image.
    """
    data = img_in.get_fdata()  # quite slow
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())

    if len(img_spacing) == 4:
        img_spacing = img_spacing[:3]  # for 4D images only use spacing of first 3 dims

    if type(new_spacing) is float:
        new_spacing = [
            new_spacing,
        ] * 3  # for 3D and 4D
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        img_spacing = np.array(
            list(img_spacing)
            + [
                new_spacing[2],
            ]
        )

    if target_shape is not None:
        # Find the right zoom to exactly reach the target_shape.
        # We also have to adapt the spacing to this new zoom.
        zoom = np.array(target_shape) / old_shape
        new_spacing = img_spacing / zoom
    else:
        zoom = img_spacing / new_spacing

    if np.array_equal(img_spacing, new_spacing):
        # Input spacing is equal to new spacing. Return image without resampling.
        return img_in

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)

    # This is only correct if all off-diagonal elements are 0
    # new_affine[0, 0] = new_spacing[0] if img_in.affine[0, 0] > 0 else -new_spacing[0]
    # new_affine[1, 1] = new_spacing[1] if img_in.affine[1, 1] > 0 else -new_spacing[1]
    # new_affine[2, 2] = new_spacing[2] if img_in.affine[2, 2] > 0 else -new_spacing[2]

    # This is the proper solution
    # Scale each column vector by the zoom of this dimension
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    # Just for information: How to get spacing from affine with rotation:
    # Calc length of each column vector:
    # vecs = affine[:3, :3]
    # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

    if (_cupy_available and _cucim_available) and use_gpu:
        new_data = resample_img_cucim(
            data, zoom=zoom, order=order, nr_cpus=nr_cpus
        )  # gpu resampling
    else:
        new_data = resample_img(
            data, zoom=zoom, order=order, nr_cpus=nr_cpus
        )  # cpu resampling

    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)

    if force_affine is not None:
        new_affine = force_affine

    new_header = img_in.header.copy()
    new_header.set_data_dtype(dtype)

    return nib.Nifti1Image(new_data, affine=new_affine, header=new_header)


def change_spacing_of_affine(affine: np.ndarray, zoom: float = 0.5) -> np.ndarray:
    """
    Change the spacing of an affine matrix.

    Parameters:
        affine (np.ndarray): The input affine matrix.
        zoom (float): The zoom factor.

    Returns:
        np.ndarray: The updated affine matrix.
    """
    new_affine = np.copy(affine)
    for i in range(3):
        new_affine[i, i] /= zoom
    return new_affine


def resample_img(
    img: np.ndarray, zoom: float = 0.5, order: int = 0, nr_cpus: int = -1
) -> np.ndarray:
    """
    Resize a numpy image array to a new size by resampling.

    Parameters:
        img (np.ndarray): The input image array. It can be 2D, 3D, or 4D.
        zoom (float): The zoom factor. A value of 0.5 will halve the image resolution, making it smaller.
        order (int): The order of interpolation. Default is 0 (nearest-neighbor).
        nr_cpus (int): The number of CPU cores to use for parallel processing. Default is -1 to use all available cores.

    Returns:
        np.ndarray: The resampled image array.

    Notes:
    - Works for 2D, 3D, and 4D images.
    """

    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    dim = len(img.shape)

    # Add dimensions to make each input 4D
    if dim == 2:
        img = img[..., None, None]
    if dim == 3:
        img = img[..., None]

    nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(
        delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3])
    )
    img_sm = np.array(img_sm).transpose(
        1, 2, 3, 0
    )  # grads channel was in front -> put to back
    if dim == 3:
        img_sm = img_sm[:, :, :, 0]
    if dim == 2:
        img_sm = img_sm[:, :, 0, 0]
    return img_sm


def resample_img_cucim(
    img: np.ndarray,
    zoom: float = 0.5,
    order: int = 0,
) -> np.ndarray:
    """
    Resample a CuPy-based numpy image array using cuCIM for potential speedup.

    Parameters:
        img (np.ndarray): The input image array as a CuPy array.
        zoom (float): The zoom factor. A value of 0.5 will halve the image resolution, making it smaller.
        order (int): The order of interpolation. Default is 0 (nearest-neighbor).

    Returns:
        np.ndarray: The resampled image array.

    Notes:
    - This function may provide a significant speedup for large images when using cuCIM.
    - The speedup is less noticeable for small images.
    - cuCIM may not always be faster depending on the environment.
    """
    from cucim.skimage.transform import resize

    img = np.asarray(img)  # slow
    new_shape = (np.array(img.shape) * zoom).round().astype(np.int32)
    resampled_img = resize(
        img, output_shape=new_shape, order=order, mode="edge", anti_aliasing=False
    )  # very fast
    resampled_img = np.asnumpy(
        resampled_img
    )  # Alternative: img_arr = np.float32(resampled_img.get())   # very fast
    return resampled_img
