from optimed.wrappers.calculations import (
    scipy_sum,
    scipy_label,
    scipy_minimum,
    scipy_center_of_mass,
    scipy_binary_dilation,
    scipy_distance_transform_edt,
)
from optimed import _cupy_available
from typing import Union
import numpy as np
import skimage

try:
    import cupy as cp
except ImportError:
    # Fallback to CPU versions
    _cupy_available = None  # noqa


def if_touches_border_3d(mask: np.ndarray, use_gpu: bool = True) -> bool:
    """
    Check if the 3d mask touches any of the borders.
    If the mask touches any border, it's considered incomplete.

    Parameters:
        mask (np.ndarray): The binary mask array.
        use_gpu (bool): If True, use GPU for computation. Default is True.

    Returns:
        bool: True if the mask touches any border, otherwise False.
    """
    if use_gpu and _cupy_available:
        mask = cp.array(mask)
    else:
        xp = np

    if mask.ndim != 3:
        raise ValueError("Mask must be 3-dimensional.")

    for axis in range(3):
        if xp.any(mask.take(0, axis=axis)) or xp.any(mask.take(-1, axis=axis)):
            return True
    return False


def if_touches_border_2d(mask: np.ndarray, use_gpu: bool = True) -> bool:
    """
    Check if the 2d mask touches any of the borders.
    If the mask touches any border, it's considered incomplete.

    Parameters:
        mask (np.ndarray): The binary mask array.
        use_gpu (bool): If True, use GPU for computation. Default is True.

    Returns:
        bool: True if the mask touches any border, otherwise False.
    """
    if use_gpu and _cupy_available:
        mask = cp.array(mask)
    else:
        xp = np

    if mask.ndim != 2:
        raise ValueError("Mask must be 2-dimensional.")

    if (
        xp.any(mask[0, :])
        or xp.any(mask[-1, :])
        or xp.any(mask[:, 0])
        or xp.any(mask[:, -1])
    ):
        return True
    return False


def find_component_by_centroid(
    components: np.ndarray,
    target_centroid: list,
    num_components: int,
    atol: int = 3,
    use_gpu: bool = True,
) -> Union[int, None]:
    """
    Find a component in a labeled image based on its centroid coordinates.

    This function searches through labeled components to find one whose centroid matches
    the target centroid coordinates within a small tolerance.

    Parameters:
        components (ndarray): Labeled image array where each unique value represents a different component.
        target_centroid (array-like): Target centroid coordinates to match against.
        num_components (int): Number of unique components in labeled image (excluding background).
        atol (int, optional): Absolute tolerance for centroid matching, by default 3.
        use_gpu (bool, optional): Whether to use GPU acceleration if available, by default True.

    Returns:
        int or None: Index of matching component if found, None otherwise

    Notes
    -----
    The function uses a tolerance when comparing centroids to allow for small variations
    in the exact centroid location.
    """
    centroids = scipy_center_of_mass(
        components > 0,
        labels=components,
        index=range(1, num_components + 1),
        use_gpu=use_gpu,
    )
    for comp_idx, centroid in enumerate(centroids, start=1):
        if np.allclose(centroid, target_centroid, atol=atol):
            return comp_idx
    return None


def find_component_by_point(
    components: np.ndarray, target_point: list, use_gpu: bool = True
) -> Union[int, None]:
    """
    Find a component in a labeled image based on a point within the component.

    This function searches through labeled components to find one that contains
    the target point coordinates within a small tolerance.

    Parameters:
        components (ndarray): Labeled image array where each unique value represents a different component.
        target_point (array-like): Target point coordinates to match against.
        use_gpu (bool, optional): Whether to use GPU acceleration if available, by default True.

    Returns:
        int or None: Index of matching component if found, None otherwise.

    Notes
    -----
    The function uses a tolerance when comparing points to allow for small variations
    in the exact point location.
    """
    if use_gpu and _cupy_available:
        components = cp.array(components)

    target_point = tuple(int(coord) for coord in target_point)
    label_at_point = components[target_point]
    if label_at_point != 0:
        return label_at_point
    return None


def delete_small_segments(
    binary_mask: np.ndarray,
    interval: list = [10, np.inf],
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Find blobs/clusters of same label. Remove all blobs which have a size which is outside of the interval.

    Parameters:
        binary_mask (ndarray) : Binary image.
        interval (list) : Boundaries of the sizes to remove.
        use_gpu (bool, optional): Whether to use GPU acceleration if available.
        verbose (bool, optional): Show debug information.
    Returns:
        np.ndarray: Filtered blobs.
    """
    if verbose:
        print("[postprocessing] deleting small components")

    if not use_gpu or not _cupy_available:
        xp = np
    else:
        xp = cp
        binary_mask = cp.array(binary_mask)

    labeled, num_labels = scipy_label(binary_mask, use_gpu=use_gpu)
    if num_labels == 0:
        return binary_mask

    counts = xp.bincount(xp.array(labeled).ravel())
    remove_labels = xp.where((counts <= interval[0]) | (counts > interval[1]))[0]

    if verbose:
        print(f"[delete_small_segments] Number of blobs before: {num_labels}")
        print(f"[delete_small_segments] Removing labels: {remove_labels}")

    binary_mask[xp.isin(xp.array(labeled), remove_labels)] = 0
    binary_mask = binary_mask.astype(bool)

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_segments_disconnected_from_point(
    binary_mask: np.ndarray,
    target_point: list,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Deletes all segments from binary_mask that are not connected to the specified target_point.
    Returns a mask where only the component containing the target_point remains
    (if it exceeds the size_threshold).

    Parameters:
        binary_mask (np.ndarray): Binary mask from which segments are removed.
        target_point (list): [row, col], coordinates of the target point (in pixels).
        size_threshold (int, optional): Minimum size to retain the component.
        use_gpu (bool, optional): Whether to use GPU acceleration (via CuPy) if available.
        verbose (bool, optional): Whether to print detailed progress information.

    Returns:
        np.ndarray: Updated binary_mask where only the segment with the target point remains
                    (if its size >= size_threshold), otherwise returns an empty mask.
    """
    if not use_gpu or not _cupy_available:
        xp = np
    else:
        xp = cp
        binary_mask = cp.array(binary_mask)

    labeled, num_labels = scipy_label(binary_mask, use_gpu=use_gpu)
    if num_labels == 0:
        return binary_mask

    target_point = tuple(int(coord) for coord in target_point)
    label_at_point = labeled[target_point]
    if label_at_point == 0:
        if verbose:
            print(
                "[delete_segments_disconnected_from_point] Target point is not in any segment."
            )
        return xp.zeros_like(binary_mask)

    keep_mask = labeled == label_at_point
    component_size = xp.count_nonzero(xp.array(keep_mask))
    if component_size < size_threshold:
        if verbose:
            print(
                f"[delete_segments_disconnected_from_point] Component size {component_size} < {size_threshold}."
            )
        return xp.zeros_like(binary_mask)

    binary_mask = keep_mask.astype(binary_mask.dtype)

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_segments_disconnected_from_parent(
    binary_mask: np.ndarray,
    parent_mask: np.ndarray,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Remove segments (in binary_mask) corresponding to connected regions in binary_mask
    that are either completely disconnected from parent_mask or are too small.

    Parameters:
        binary_mask (np.ndarray): Binary mask with disconnected segments.
        parent_mask (np.ndarray): Reference mask.
        size_threshold (int): Minimum allowed component size for those that intersect the parent_mask.
        use_gpu (bool, optional): If True, use GPU acceleration.
        verbose (bool, optional): If True, print progress information.

    Returns:
        np.ndarray: Updated binary_mask with disconnected segments removed.
    """
    if not use_gpu or not _cupy_available:
        xp = np
    else:
        xp = cp
        binary_mask = cp.array(binary_mask)
        parent_mask = cp.array(parent_mask)

    if not xp.any(binary_mask) or not xp.any(parent_mask):
        return xp.zeros_like(binary_mask)

    labeled, num_labels = scipy_label(binary_mask, use_gpu=use_gpu)
    if num_labels == 0:
        return binary_mask

    intersections = scipy_sum(
        parent_mask, labels=labeled, index=np.arange(1, num_labels + 1), use_gpu=use_gpu
    )
    sizes = xp.bincount(xp.array(labeled).ravel())[1:]
    remove_labels = (
        xp.where(
            (intersections == 0) | ((intersections > 0) & (sizes < size_threshold))
        )[0]
        + 1
    )

    if verbose:
        for lbl in remove_labels:
            idx = lbl - 1
            if intersections[idx] == 0:
                print(
                    f"[delete_segments_disconnected_from_parent] Deleting component {lbl} (size={sizes[idx]}) - no intersection."
                )
            else:
                print(
                    f"[delete_segments_disconnected_from_parent] Deleting small component {lbl} (size={sizes[idx]})."
                )

    binary_mask[xp.isin(xp.array(labeled), remove_labels)] = 0

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_segments_distant_from_point(
    binary_mask: np.ndarray,
    target_point: list,
    distance_threshold: int = 5,
    size_threshold: int = 10,
    keep_large: bool = True,
    large_threshold: int = 100000,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Delete segments from binary_mask that are distant from the target point.
    Segments whose minimum distance exceeds distance_threshold are deleted,
    unless they are large (when keep_large=True) or their size is smaller than size_threshold.

    Parameters:
        binary_mask (np.ndarray): Binary mask from which segments are removed.
        target_point (list): Coordinates of the target point [row, col].
        distance_threshold (int, optional): Maximum allowable minimum distance to the target point.
        size_threshold (int, optional): Minimum size of a segment to retain if it is close.
        keep_large (bool, optional): If True, do not delete large components even if they are distant.
        large_threshold (int, optional): Size threshold for large components.
        use_gpu (bool, optional): Whether to use GPU acceleration if available.
        verbose (bool, optional): Whether to print detailed progress information.

    Returns:
        np.ndarray: Updated binary_mask with segments not meeting the condition removed.
    """
    if not use_gpu or not _cupy_available:
        xp = np
    else:
        xp = cp
        binary_mask = cp.array(binary_mask)

    point_mask = xp.zeros_like(binary_mask, dtype=bool)
    point_mask[tuple(int(coord) for coord in target_point)] = True
    point_mask = scipy_binary_dilation(point_mask, iterations=1, use_gpu=use_gpu)
    dt = scipy_distance_transform_edt(~point_mask, use_gpu=use_gpu)

    labeled, num_labels = scipy_label(binary_mask, use_gpu=use_gpu)
    if num_labels == 0:
        return binary_mask

    min_dists = scipy_minimum(
        dt, labels=labeled, index=xp.arange(1, num_labels + 1), use_gpu=use_gpu
    )
    sizes = xp.bincount(xp.array(labeled).ravel())[1:]
    remove_labels = (
        xp.where(
            (
                (min_dists > distance_threshold)
                & (sizes <= large_threshold if keep_large else True)
            )
            | ((min_dists <= distance_threshold) & (sizes < size_threshold))
        )[0]
        + 1
    )

    if verbose:
        for lbl in remove_labels:
            idx = lbl - 1
            if min_dists[idx] > distance_threshold:
                print(
                    f"[delete_distant_point_segments] Deleting component {lbl} (size: {sizes[idx]}, dist: {min_dists[idx]:.2f})."
                )
            else:
                print(
                    f"[delete_distant_point_segments] Deleting small component {lbl} (size: {sizes[idx]})."
                )

    binary_mask[xp.isin(xp.array(labeled), remove_labels)] = 0

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_nearby_segments_with_buffer(
    binary_mask: np.ndarray,
    parent_mask: np.ndarray,
    distance_threshold: int = 5,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Delete segments that are too close to the main label based on distance threshold.

    Parameters:
        binary_mask (np.ndarray): Binary mask with disconnected segments.
        parent_mask (np.ndarray): Reference mask.
        distance_threshold (float, optional): Minimum distance required between a segment and the main label.
        size_threshold (int, optional): Size threshold for retaining segments.
        use_gpu (bool, optional): If True, use GPU acceleration.
        verbose (bool, optional): If True, print progress information.

    Returns:
        np.ndarray: Binary mask with nearby segments removed.
    """
    if verbose:
        print("[postprocessing] deleting close components")

    if not use_gpu or not _cupy_available:
        xp = np

    if not xp.any(binary_mask):
        return binary_mask

    if not xp.any(parent_mask):
        return binary_mask

    connected_components, num_components = scipy_label(
        xp.array(binary_mask), use_gpu=use_gpu
    )
    for component_idx in range(1, num_components + 1):  # Skip background label 0
        component_mask = connected_components == component_idx
        component_size = xp.count_nonzero(xp.array(component_mask))

        if verbose:
            print(f"\tAnalysing component {component_idx}/{num_components}...")

        if not isinstance(component_mask, np.ndarray):
            component_mask = component_mask.get()

        buffered_mask = skimage.segmentation.expand_labels(
            component_mask, distance_threshold
        )

        if xp.sum(xp.array(buffered_mask) * xp.array(parent_mask)) > 0:
            binary_mask[component_mask] = 0
            if verbose:
                print(
                    f"\t\tDeleting component {component_idx} due to proximity to parent component."
                )
            continue
        if component_size < size_threshold:
            binary_mask[component_mask] = 0
            if verbose:
                print(
                    f"\t\tDeleting small component {component_idx} (size: {component_size})."
                )

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_touching_border_segments(
    binary_mask: np.ndarray,
    size_threshold: int = 100,
    use_gpu: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Delete segments touching the border of the image that are smaller than a specified size threshold.

    Parameters:
        binary_mask (np.ndarray): Binary mask containing all segments.
        size_threshold (int, optional): Minimum size for segments to be retained. If -1, threshold is ignored.
        use_gpu (bool, optional): If True, uses GPU acceleration.
        verbose (bool, optional): If True, outputs detailed processing information.

    Returns:
        np.ndarray: Modified binary mask with small border-touching segments removed.

    The function identifies connected components in the binary_mask, checks whether they touch
    the image border, and then removes them only if their size is below the size_threshold.
    Components that are larger than or equal to size_threshold are preserved even if they touch the borders.
    """
    if not use_gpu or not _cupy_available:
        xp = np
    else:
        xp = cp
        binary_mask = cp.array(binary_mask)

    labeled, num_labels = scipy_label(binary_mask, use_gpu=use_gpu)
    if num_labels == 0:
        return binary_mask

    for lbl in range(1, num_labels + 1):
        comp_mask = labeled == lbl
        touches_border = (
            if_touches_border_3d(comp_mask, use_gpu)
            if binary_mask.ndim == 3
            else if_touches_border_2d(comp_mask, use_gpu)
        )
        comp_size = xp.count_nonzero(xp.array(comp_mask))
        if touches_border and (size_threshold == -1 or comp_size < size_threshold):
            binary_mask[xp.array(comp_mask)] = 0
            if verbose:
                print(
                    f"[delete_touching_border_segments] Deleting border-touching component {lbl} (size: {comp_size})."
                )

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def fill_holes_2d(
    binary_mask: np.ndarray, max_hole_size: int = 100, use_gpu: bool = True
) -> np.ndarray:
    """
    Fill small holes in a 2D binary mask.

    Parameters:
        binary_mask (np.ndarray): The 2D binary mask.
        max_hole_size (int): Maximum size of holes to fill.
        use_gpu (bool): If True, use GPU if available.

    Returns:
        np.ndarray: The 2D binary mask with small holes filled.
    """
    if binary_mask.ndim != 2:
        raise ValueError("Mask must be 2-dimensional.")

    if not use_gpu or not _cupy_available:
        xp = np

    holes_labeled, num_labels = scipy_label(~xp.array(binary_mask), use_gpu=use_gpu)
    counts = xp.bincount(xp.array(holes_labeled).ravel())
    remove_labels = xp.where(counts <= max_hole_size)[0]
    binary_mask[xp.isin(xp.array(holes_labeled), remove_labels)] = 1

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def fill_holes_3d(
    binary_mask: np.ndarray, max_hole_size: int = 100, use_gpu: bool = True
) -> np.ndarray:
    """
    Fill small holes in a 3D binary mask.

    Parameters:
        binary_mask (np.ndarray): The 3D binary mask.
        max_hole_size (int): Maximum size of holes to fill.
        use_gpu (bool): If True, use GPU if available.

    Returns:
        np.ndarray: The 3D binary mask with small holes filled.
    """
    if binary_mask.ndim != 3:
        raise ValueError("Mask must be 3-dimensional.")

    if not use_gpu or not _cupy_available:
        xp = np
    else:
        xp = cp
        binary_mask = cp.array(binary_mask)

    holes_labeled, num_labels = scipy_label(~binary_mask, use_gpu=use_gpu)
    counts = xp.bincount(xp.array(holes_labeled).ravel())
    remove_labels = xp.where(counts <= max_hole_size)[0]
    binary_mask[xp.isin(xp.array(holes_labeled), remove_labels)] = 1

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask
