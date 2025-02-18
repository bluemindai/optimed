from optimed.wrappers.calculations import (
    scipy_sum,
    scipy_label,
    scipy_minimum,
    scipy_binary_dilation,
    scipy_distance_transform_edt
)
from optimed import _cupy_available
from typing import Union
import numpy as np
import skimage

try:
    import cupy as cp
    xp = cp
except ImportError as exc:
    xp = np


def if_touches_border_3d(
    mask: np.ndarray,
    use_gpu: bool = True
) -> bool:
    """
    Check if the 3d mask touches any of the borders.
    If the mask touches any border, it's considered incomplete.

    Parameters:
        mask (np.ndarray): The binary mask array.
        use_gpu (bool): If True, use GPU for computation. Default is True.

    Returns:
        bool: True if the mask touches any border, otherwise False.
    """
    if not use_gpu or not _cupy_available:
        xp = np

    mask = xp.array(mask)
    if xp.any(mask[1, :, :]) or xp.any(mask[-2, :, :]):
        return True
    if xp.any(mask[:, 1, :]) or xp.any(mask[:, -2, :]):
        return True
    if xp.any(mask[:, :, 1]) or xp.any(mask[:, :, -2]):
        return True
    
    if not isinstance(mask, np.ndarray):
        mask = mask.get()

    return False


def if_touches_border_2d(
    mask: np.ndarray,
    use_gpu: bool = True
) -> bool:
    """
    Check if the 2d mask touches any of the borders.
    If the mask touches any border, it's considered incomplete.

    Parameters:
        mask (np.ndarray): The binary mask array.
        use_gpu (bool): If True, use GPU for computation. Default is True.

    Returns:
        bool: True if the mask touches any border, otherwise False.
    """
    if not use_gpu or not _cupy_available:
        xp = np

    mask = xp.array(mask)
    if xp.any(mask[1, :]) or xp.any(mask[-2, :]):
        return True
    if xp.any(mask[:, 1]) or xp.any(mask[:, -2]):
        return True

    if not isinstance(mask, np.ndarray):
        mask = mask.get()

    return False


def find_component_by_centroid(
    components: np.ndarray,
    target_centroid: list,
    num_components: int,
    atol: int = 3,
    use_gpu: bool = True
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

    if not use_gpu or not _cupy_available:
        xp = np

    for comp_idx in range(1, num_components + 1):
        comp_mask = xp.array(components == comp_idx)
        if xp.any(comp_mask):
            coords = xp.argwhere(comp_mask)
            center = coords.mean(axis=0).astype(int)
            if xp.allclose(center, target_centroid, atol=atol):
                return comp_idx
    return None


def find_component_by_point(
    components: np.ndarray,
    target_point: list,
    num_components: int,
    atol: int = 3,
    use_gpu: bool = True
) -> Union[int, None]:
    """
    Find a component in a labeled image based on a point within the component.

    This function searches through labeled components to find one that contains 
    the target point coordinates within a small tolerance.

    Parameters:
        components (ndarray): Labeled image array where each unique value represents a different component.
        target_point (array-like): Target point coordinates to match against.
        num_components (int): Number of unique components in labeled image (excluding background).
        atol (int, optional): Absolute tolerance for point matching, by default 3.
        use_gpu (bool, optional): Whether to use GPU acceleration if available, by default True.

    Returns:
        int or None: Index of matching component if found, None otherwise.

    Notes
    -----
    The function uses a tolerance when comparing points to allow for small variations
    in the exact point location.
    """

    if not use_gpu or not _cupy_available:
        xp = np

    target_point = xp.array(target_point)
    for comp_idx in range(1, num_components + 1):
        comp_mask = xp.array(components == comp_idx)
        if xp.any(comp_mask):
            coords = xp.argwhere(comp_mask)
            # Compute Euclidean distances from each coordinate to the target point.
            distances = xp.linalg.norm(coords - target_point, axis=1)
            if xp.any(distances <= atol):
                return comp_idx
    return None


def delete_small_segments(
    binary_mask: np.ndarray,
    interval: list = [10, np.inf],
    use_gpu: bool = True,
    verbose: bool = False
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
        print(f"[postprocessing] deleting small components")

    if not use_gpu or not _cupy_available:
        xp = np

    mask, number_of_blobs = scipy_label(xp.array(binary_mask), use_gpu=use_gpu)
    mask = xp.array(mask)

    if verbose: 
        print('[delete_small_segments] Number of blobs before: ' + str(number_of_blobs))

    counts = xp.bincount(mask.flatten())  # number of pixels in each blob

    # If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1: return binary_mask

    remove = xp.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = xp.nonzero(remove)[0]
    mask[xp.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1  # set everything else to 1

    if verbose:
        print(f"\t[delete_small_segments] counts: {sorted(counts)[::-1]}")
        _, number_of_blobs_after = scipy_label(mask, use_gpu=use_gpu)
        print('[delete_small_segments] Number of blobs after: ' + str(number_of_blobs_after))

    if not isinstance(mask, np.ndarray):
        mask = mask.get()

    return mask


def delete_segments_disconnected_from_point(
    binary_mask: np.ndarray,
    target_point: list,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True
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

    binary_mask_xp = xp.array(binary_mask)

    connected_components, num_components = scipy_label(binary_mask_xp, use_gpu=use_gpu)
    if verbose:
        print(f"[delete_segments_disconnected_from_point] Found {num_components} components in binary_mask.")

    # Determine which component contains the target_point
    component_label = find_component_by_point(
        connected_components, target_point, num_components, atol=3, use_gpu=use_gpu
    )
    if component_label is None:
        # If the point is not in any segment, return an empty mask
        if verbose:
            print("[delete_segments_disconnected_from_point] No component contains the target point. Returning empty mask.")
        return np.zeros_like(binary_mask)

    if verbose:
        print(f"[delete_segments_disconnected_from_point] Target point is in component {component_label}.")

    # Form a mask with only this component
    keep_mask = (connected_components == component_label)

    main_component_size = xp.count_nonzero(keep_mask)
    if main_component_size < size_threshold:
        if verbose:
            print(
                f"[delete_segments_disconnected_from_point] Main component (label {component_label}) is too small "
                f"(size: {main_component_size}). Removing it."
            )
        # If too small, return an empty mask
        result = xp.zeros_like(binary_mask_xp)
    else:
        result = xp.where(keep_mask, 1, 0)

    if not isinstance(result, np.ndarray):
        result = result.get()

    return result


def delete_segments_disconnected_from_parent(
    binary_mask: np.ndarray,
    parent_mask: np.ndarray,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True
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

    if not xp.any(binary_mask):
        return binary_mask
    
    if not xp.any(parent_mask):
        return binary_mask

    binary_mask = xp.array(binary_mask)
    parent_mask = xp.array(parent_mask)

    if verbose:
        print("[postprocessing] deleting disconnected components...")

    binary_mask = xp.array(binary_mask)
    parent_mask = xp.array(parent_mask)

    connected_components, num_components = scipy_label(binary_mask, use_gpu=use_gpu)
    connected_components = xp.array(connected_components)
    if verbose:
        print(f"\tFound {num_components} components.")

    # Compute component sizes (include background label 0)
    comp_sizes = xp.bincount(connected_components.ravel())
    # Compute for each component the number of pixels that intersect the parent_mask.
    # For each pixel in connected_components, if parent_mask is True, we add 1.
    parent_intersections = scipy_sum(parent_mask, labels=connected_components, index=np.arange(comp_sizes.size), use_gpu=use_gpu)
    parent_intersections = xp.array(parent_intersections)
    # We ignore label 0 (background) in the removal decision.
    comp_labels = xp.arange(1, num_components + 1)
    comp_sizes = comp_sizes[1:]
    parent_intersections = parent_intersections[1:]

    # Determine which labels to remove:
    # - Remove a component if it does not intersect with the parent_mask.
    # - For components that do intersect, remove if the size is smaller than the size_threshold.
    remove_labels = comp_labels[(parent_intersections == 0) | ((parent_intersections > 0) & (comp_sizes < size_threshold))]

    if verbose:
        for lbl in remove_labels:
            # Use 1-indexed labels. Since we've already removed background, subtract one for indexing.
            idx = lbl - 1
            if parent_intersections[idx] == 0:
                print(
                    f"\tDeleting component {lbl} (size: {comp_sizes[idx]}) "
                    "with no intersection with parent mask."
                )
            else:
                print(f"\tDeleting small component {lbl} (size: {comp_sizes[idx]}).")

    # Remove the marked components from binary_mask in one vectorized step.
    mask_to_remove = xp.isin(connected_components, remove_labels)
    binary_mask[mask_to_remove] = 0

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_segments_distant_from_point(
    binary_mask: np.ndarray,
    target_point: list,
    distance_threshold: int = 5,
    size_threshold: int = 10,
    keep_large: bool = True,
    use_gpu: bool = True,
    verbose: bool = True
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
        use_gpu (bool, optional): Whether to use GPU acceleration if available.
        verbose (bool, optional): Whether to print detailed progress information.

    Returns:
        np.ndarray: Updated binary_mask with segments not meeting the condition removed.
    """
    if not use_gpu or not _cupy_available:
        xp = np

    binary_mask = xp.array(binary_mask)

    point_mask = xp.zeros_like(binary_mask, dtype=bool)

    point_coords = tuple(int(coord) for coord in target_point)
    point_mask[point_coords] = True

    if verbose:
        print("[delete_distant_point_segments] Created target point mask.")

    point_mask = scipy_binary_dilation(point_mask, iterations=1, use_gpu=use_gpu)
    point_mask = xp.array(point_mask)

    dt = scipy_distance_transform_edt(~point_mask, use_gpu=use_gpu)
    dt = xp.array(dt)

    connected_components, num_components = scipy_label(binary_mask, use_gpu=use_gpu)
    if verbose:
        print(f"[delete_distant_point_segments] Found {num_components} components.")

    # Compute component sizes (ignore background, label=0)
    sizes = xp.bincount(connected_components.ravel())
    comp_labels = xp.arange(1, num_components + 1)
    comp_sizes = sizes[1:]

    # Compute the minimum distance for each component to the target point
    min_dists = scipy_minimum(dt, labels=connected_components, index=comp_labels, use_gpu=use_gpu)
    min_dists = xp.array(min_dists)

    large_threshold = 100000

    if keep_large:
        remove_far = (min_dists > distance_threshold) & (comp_sizes <= large_threshold)
    else:
        remove_far = (min_dists > distance_threshold)
    remove_small = (min_dists <= distance_threshold) & (comp_sizes < size_threshold)

    remove_mask = remove_far | remove_small
    remove_labels = comp_labels[remove_mask]

    if verbose:
        for i, label_val in enumerate(comp_labels):
            if remove_mask[i]:
                if min_dists[i] > distance_threshold:
                    print(
                        f"[delete_distant_point_segments] Deleting component {label_val} (size: {comp_sizes[i]}) "
                        f"because its minimum distance to the target point is {min_dists[i]:.2f} (> {distance_threshold})."
                    )
                else:
                    print(f"[delete_distant_point_segments] Deleting small component {label_val} (size: {comp_sizes[i]}).")

    if remove_labels.size > 0:
        binary_mask[xp.isin(connected_components, remove_labels)] = 0

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_segments_disconnected_from_parent(
    binary_mask: np.ndarray,
    parent_mask: np.ndarray,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True
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

    if not xp.any(binary_mask):
        return binary_mask
    
    if not xp.any(parent_mask):
        return binary_mask

    binary_mask = xp.array(binary_mask)
    parent_mask = xp.array(parent_mask)

    if verbose:
        print("[postprocessing] Deleting disconnected components...")

    connected_components, num_components = scipy_label(binary_mask, use_gpu=use_gpu)
    if verbose:
        print(f"\tFound {num_components} components.")

    comp_sizes = xp.bincount(connected_components.ravel())

    parent_intersections = scipy_sum(
        parent_mask, labels=connected_components,
        index=xp.arange(comp_sizes.size), use_gpu=use_gpu
    )

    # Analyze only components with label 1..num_components (skip background)
    comp_labels = xp.arange(1, num_components + 1)
    comp_sizes = comp_sizes[1:]
    parent_intersections = parent_intersections[1:]

    # Conditions for removal:
    # 1) No intersection with parent (parent_intersections == 0)
    # 2) Intersection exists, but component size is below size_threshold
    remove_labels = comp_labels[
        (parent_intersections == 0) |
        ((parent_intersections > 0) & (comp_sizes < size_threshold))
    ]

    if verbose:
        for lbl in remove_labels:
            idx = lbl - 1
            if parent_intersections[idx] == 0:
                print(f"\tDeleting component {lbl} (size={comp_sizes[idx]}) - no intersection with parent.")
            else:
                print(f"\tDeleting small component {lbl} (size={comp_sizes[idx]}) - below threshold.")

    mask_to_remove = xp.isin(connected_components, remove_labels)
    binary_mask[mask_to_remove] = 0

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_nearby_segments_with_buffer(
    binary_mask: np.ndarray,
    parent_mask: np.ndarray,
    distance_threshold: int = 5,
    size_threshold: int = 10,
    use_gpu: bool = True,
    verbose: bool = True
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
    if verbose: print(f"[postprocessing] deleting close components")

    if not use_gpu or not _cupy_available:
        xp = np

    if not xp.any(binary_mask):
        return binary_mask

    if not xp.any(parent_mask):
        return binary_mask

    connected_components, num_components = scipy_label(xp.array(binary_mask), use_gpu=use_gpu)
    for component_idx in range(1, num_components + 1):  # Skip background label 0
        component_mask = connected_components == component_idx
        component_size = xp.count_nonzero(xp.array(component_mask))

        if verbose:
            print(f"\tAnalysing component {component_idx}/{num_components}...")

        if not isinstance(component_mask, np.ndarray):
            component_mask = component_mask.get()

        buffered_mask = skimage.segmentation.expand_labels(
            component_mask, 
            distance_threshold
        )

        if xp.sum(xp.array(buffered_mask) * xp.array(parent_mask)) > 0:
            binary_mask[component_mask] = 0
            if verbose: print(f"\t\tDeleting component {component_idx} due to proximity to parent component.")
            continue
        if component_size < size_threshold:
            binary_mask[component_mask] = 0
            if verbose: print(f"\t\tDeleting small component {component_idx} (size: {component_size}).")

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def delete_touching_border_segments(
    binary_mask: np.ndarray,
    size_threshold: int = 100,
    use_gpu: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Delete segments touching the border of the image that are smaller than a specified size threshold.

    Parameters:
        binary_mask (np.ndarray): Binary mask containing all segments.
        size_threshold (int, optional): Minimum size for segments to be retained.
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

    if verbose:
        print(f"[postprocessing] Deleting border-touching segments smaller than {size_threshold}...")

    connected_components, num_components = scipy_label(xp.array(binary_mask), use_gpu=use_gpu)

    for component_idx in range(1, num_components + 1):
        component_mask = (connected_components == component_idx)
        component_size = xp.count_nonzero(component_mask)

        if verbose:
            print(f"\tAnalyzing component {component_idx}/{num_components} (size: {component_size})...")

        if not isinstance(component_mask, np.ndarray):
            component_mask = component_mask.get()

        if if_touches_border_3d(component_mask):
            # Remove the component only if it is smaller than the threshold.
            if component_size < size_threshold:
                binary_mask[component_mask] = 0
                if verbose:
                    print(f"\tRemoving border-touching component {component_idx} (size: {component_size}).")
            else:
                if verbose:
                    print(
                        f"\tComponent {component_idx} touches the border "
                        f"but is kept (size: {component_size} >= {size_threshold})."
                    )

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def fill_holes_2d(
    binary_mask: np.ndarray,
    max_hole_size: int = 100,
    use_gpu: bool = True
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
    if not use_gpu or not _cupy_available:
        xp = np

    holes_labeled, num_labels = scipy_label(~xp.array(binary_mask), use_gpu=use_gpu)
    counts = xp.bincount(holes_labeled.ravel())
    remove_labels = xp.where(counts <= max_hole_size)[0]
    binary_mask[xp.isin(holes_labeled, remove_labels)] = 1

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask


def fill_holes_3d(
    binary_mask: np.ndarray,
    max_hole_size: int = 100,
    use_gpu: bool = True
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
    if not use_gpu or not _cupy_available:
        xp = np

    holes_labeled, num_labels = scipy_label(~xp.array(binary_mask), use_gpu=use_gpu)
    counts = xp.bincount(holes_labeled.ravel())
    remove_labels = xp.where(counts <= max_hole_size)[0]
    binary_mask[xp.isin(holes_labeled, remove_labels)] = 1

    if not isinstance(binary_mask, np.ndarray):
        binary_mask = binary_mask.get()

    return binary_mask
