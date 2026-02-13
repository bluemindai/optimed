import skimage
import numpy as np
from scipy.ndimage import gaussian_filter
from optimed.wrappers.calculations import scipy_binary_dilation
from optimed.processes.crop import get_bbox_from_mask
from numpy.typing import DTypeLike


def rescale_intensity(
    img: np.ndarray,
    lower: float = -125.0,
    upper: float = 225.0,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """
    Apply intensity rescaling to an image (also known as windowing).

    Parameters:
        img (np.ndarray): Input image.
        lower (float): Lower bound.
        upper (float): Upper bound.
        dtype (np.dtype): Data type.

    Returns:
        np.ndarray: Rescaled image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if dtype not in [np.float32, np.float64]:
        raise TypeError("dtype must be a float type")

    clipped = np.clip(img, lower, upper).astype(dtype)
    return skimage.exposure.rescale_intensity(
        clipped, in_range=(lower, upper), out_range=(0, 255)
    )


def denoise_image(
    img: np.ndarray,
    sigma: float = 0.5,
    mode: str = "gaussian",
    channel_axis: int = None,
) -> np.ndarray:
    """
    Denoise an image using a Gaussian filter.

    Parameters:
        img (np.ndarray): Input image.
        sigma (float): Standard deviation for Gaussian kernel.
        mode (str): Type of denoising filter.
        channel_axis (int, optional): If not None, the axis of img corresponding to channels.

    Returns:
        np.ndarray: Denoised image.
    """
    if mode not in ["gaussian", "bilateral"]:
        raise ValueError('mode must be either "gaussian" or "bilateral"')
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 3:
        raise ValueError("only supports 3D images")

    if mode == "gaussian":
        return gaussian_filter(img, sigma=sigma)
    else:  # bilateral
        return skimage.restoration.denoise_bilateral(
            img, sigma_color=sigma, channel_axis=channel_axis
        )


def get_mask_by_threshold(img: np.ndarray, threshold: float, above: bool = True):
    """
    Generate a binary mask by thresholding the input image.

    Parameters:
        img (np.ndarray): Input image.
        threshold (float): The threshold value.
        above (bool): If True, returns mask where img > threshold,
                      else returns mask where img < threshold.

    Returns:
        np.ndarray: A binary mask.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 3:
        raise ValueError("only supports 3D images")

    if above:
        mask = img > threshold
    else:
        mask = img < threshold
    return mask


def blur_inside_roi(
    img: np.ndarray,
    roi_mask: np.ndarray,
    sigma: int = 5,
    addon: int = 0,
    use_bbox: bool = False,
):
    """
    Blur the inside of a region of interest (ROI) in an image.
    This function applies a Gaussian blur only to the ROI (or to the bounding box of the ROI
    if use_bbox is True), leaving the rest of the image intact.

    Parameters:
        img (np.ndarray): The input image.
        roi_mask (np.ndarray): The binary mask representing the ROI.
        sigma (int): The sigma value for the Gaussian filter.
        addon (int): Additional margin to add to the bounding box if use_bbox is True.
        use_bbox (bool): If True, use the bounding box derived from roi_mask instead of the exact mask.

    Returns:
        np.ndarray: The image with the ROI blurred.
    """
    if not isinstance(img, np.ndarray) or not isinstance(roi_mask, np.ndarray):
        raise TypeError("img and roi_mask must be NumPy arrays")
    if img.ndim != 3 or roi_mask.ndim != 3:
        raise ValueError("only supports 3D images")
    if img.shape != roi_mask.shape:
        raise ValueError("img and roi_mask must have the same shape")

    blurred_data = gaussian_filter(img, sigma=sigma)

    if use_bbox:
        bbox = get_bbox_from_mask(roi_mask, outside_value=0, addon=addon, verbose=False)
        new_mask = np.zeros_like(roi_mask, dtype=bool)
        new_mask[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ] = True
        mask_to_use = new_mask
    else:
        mask_to_use = roi_mask.astype(bool)

    result = np.where(mask_to_use, blurred_data, img)
    return result


def blur_outside_roi(
    img: np.ndarray,
    roi_mask: np.ndarray,
    sigma: int = 5,
    addon: int = 0,
    use_bbox: bool = False,
):
    """
    Blur the outside of a region of interest (ROI) in an image.
    Useful for anonymizing data outside the ROI.

    Parameters:
        img (np.ndarray): The input image.
        roi_mask (np.ndarray): The ROI mask.
        sigma (int): The sigma value for the Gaussian filter. Default is 5.
        addon (int): Additional margin to add to the bounding box. Default is 0.
        use_bbox (bool): Use bounding box. Default is False.

    Returns:
        np.ndarray: The image with the outside of the ROI blurred.
    """
    if not isinstance(img, np.ndarray) or not isinstance(roi_mask, np.ndarray):
        raise TypeError("img and roi_mask must be NumPy arrays")
    if img.ndim != 3 or roi_mask.ndim != 3:
        raise ValueError("only supports 3D images")
    if img.shape != roi_mask.shape:
        raise ValueError("img and roi_mask must have the same shape")

    blurred_data = gaussian_filter(img, sigma=sigma)

    if use_bbox:
        bbox = get_bbox_from_mask(roi_mask, outside_value=0, addon=addon, verbose=False)
        new_mask = np.zeros_like(roi_mask, dtype=bool)
        new_mask[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ] = True
        mask_to_use = new_mask
    else:
        if addon > 0:
            structure = np.ones((addon, addon, addon))
            roi_mask = scipy_binary_dilation(roi_mask, structure, iterations=5)
        mask_to_use = roi_mask.astype(bool)

    result = np.where(mask_to_use, img, blurred_data)

    return result
