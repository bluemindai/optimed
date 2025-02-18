import unittest
import numpy as np
from optimed.processes.postprocess import (
    if_touches_border_3d,
    if_touches_border_2d,
    find_component_by_centroid,
    find_component_by_point,
    delete_small_segments,
    delete_segments_disconnected_from_point,
    delete_segments_disconnected_from_parent,
    delete_segments_distant_from_point,
    delete_nearby_segments_with_buffer,
    delete_touching_border_segments,
    fill_holes_2d,
    fill_holes_3d
)

class TestPostprocessFunctions(unittest.TestCase):

    def test_find_component_by_centroid_found(self):
        labeled = np.zeros((5, 5), dtype=np.int32)
        labeled[2:4, 2:4] = 1  # Component 1
        comp = find_component_by_centroid(labeled, target_centroid=[2, 2], num_components=1, atol=1, use_gpu=False)
        self.assertEqual(comp, 1)

    def test_find_component_by_centroid_not_found(self):
        labeled = np.zeros((5, 5), dtype=np.int32)
        labeled[2:4, 2:4] = 1  # Component 1, center (2,2)
        comp = find_component_by_centroid(labeled, target_centroid=[0, 0], num_components=1, atol=1, use_gpu=False)
        self.assertIsNone(comp)

    def test_find_component_by_point_found(self):
        labeled = np.zeros((5, 5), dtype=np.int32)
        labeled[2:4, 2:4] = 1
        comp = find_component_by_point(labeled, target_point=[2, 2], num_components=1, atol=1, use_gpu=False)
        self.assertEqual(comp, 1)

    def test_find_component_by_point_not_found(self):
        labeled = np.zeros((5, 5), dtype=np.int32)
        labeled[2:4, 2:4] = 1
        comp = find_component_by_point(labeled, target_point=[0, 0], num_components=1, atol=1, use_gpu=False)
        self.assertIsNone(comp)

    def test_if_touches_border_3d_true(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1, 2, 2] = 1
        self.assertTrue(if_touches_border_3d(mask, use_gpu=False))

    def test_if_touches_border_3d_false(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[2, 2, 2] = 1
        self.assertFalse(if_touches_border_3d(mask, use_gpu=False))

    def test_if_touches_border_2d_true(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1, 3] = 1
        self.assertTrue(if_touches_border_2d(mask, use_gpu=False))

    def test_if_touches_border_2d_false(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        self.assertFalse(if_touches_border_2d(mask, use_gpu=False))

    def test_delete_small_segments(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        # Small component (3 pixels)
        mask[1, 1] = 1
        mask[1, 2] = 1
        mask[2, 1] = 1
        # Large component (4x4 = 16 pixels)
        mask[5:9, 5:9] = 1
        result = delete_small_segments(mask, interval=[4, np.inf], use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 16)

    def test_delete_segments_disconnected_from_point(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 1
        mask[6:10, 6:10] = 1
        result = delete_segments_disconnected_from_point(mask, target_point=[7, 7], size_threshold=5, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 16)

    def test_delete_segments_disconnected_from_point_small(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        result = delete_segments_disconnected_from_point(mask, target_point=[0, 0], size_threshold=5, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 0)

    def test_delete_segments_disconnected_from_parent(self):
        # Scenario where both components are present and the parent mask overlaps one of them.
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 1    # Component 1 (9 pixels)
        mask[6:10, 6:10] = 1  # Component 2 (16 pixels)
        parent_mask = np.zeros((10, 10), dtype=np.uint8)
        parent_mask[7:9, 7:9] = 1  # Covers only part of component 2
        result = delete_segments_disconnected_from_parent(mask, parent_mask, size_threshold=5, use_gpu=False, verbose=False)
        # Expect component 1 to be removed, and component 2 to remain (16 pixels)
        self.assertEqual(np.count_nonzero(result), 16)

    def test_delete_segments_disconnected_from_parent_empty_binary(self):
        # Case where binary_mask is empty.
        mask = np.zeros((10, 10), dtype=np.uint8)
        parent_mask = np.ones((10, 10), dtype=np.uint8)
        result = delete_segments_disconnected_from_parent(mask, parent_mask, size_threshold=5, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 0)

    def test_delete_segments_disconnected_from_parent_empty_parent(self):
        # Case where parent_mask is empty: function should return the original binary_mask.
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 1
        parent_mask = np.zeros((10, 10), dtype=np.uint8)
        result = delete_segments_disconnected_from_parent(mask, parent_mask, size_threshold=5, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), np.count_nonzero(mask))

    def test_delete_segments_disconnected_from_parent_partial_removal(self):
        # Additional test: two components, one of which is below the threshold and does not intersect with parent_mask.
        mask = np.zeros((10, 10), dtype=np.uint8)
        # Component 1: top left corner (3x3 = 9 pixels)
        mask[0:3, 0:3] = 1
        # Component 2: bottom right corner (4x4 = 16 pixels)
        mask[6:10, 6:10] = 1
        # Parent mask covers only component 2.
        parent_mask = np.zeros((10, 10), dtype=np.uint8)
        parent_mask[7:9, 7:9] = 1
        # With size_threshold=10, component 1 (9 pixels) is below the threshold and should be removed.
        result = delete_segments_disconnected_from_parent(mask, parent_mask, size_threshold=10, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 16)

    def test_delete_segments_distant_from_point(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4:7, 4:7] = 1
        mask[0:2, 0:2] = 1
        target_point = [5, 5]
        result = delete_segments_distant_from_point(mask, target_point,
                                                    distance_threshold=2, size_threshold=5,
                                                    keep_large=True, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 9)

    def test_delete_nearby_segments_with_buffer(self):
        # 2D mask: two components that do not merge.
        mask = np.zeros((10, 10), dtype=np.uint8)
        # Large component: 4x4 block at the bottom (rows 4–7, columns 4–7) – 16 pixels
        mask[4:8, 4:8] = 1
        # Small component: 2x2 block at the top (rows 0–2, columns 4–6) – 4 pixels
        mask[0:2, 4:6] = 1
        # Parent mask covers the area of the small component (intersects only with it)
        parent_mask = np.zeros((10, 10), dtype=np.uint8)
        parent_mask[0:3, 4:7] = 1
        result = delete_nearby_segments_with_buffer(mask, parent_mask,
                                                     distance_threshold=1, size_threshold=5,
                                                     use_gpu=False, verbose=False)
        # Expect: small component removed, large remains (16 pixels)
        self.assertEqual(np.count_nonzero(result), 16)

    def test_delete_nearby_segments_with_buffer_small_component(self):
        # Test for branch: if component_size < size_threshold:
        # Create a small component that does not intersect with the parent mask.
        mask = np.zeros((10, 10), dtype=np.uint8)
        # Small component: 2x2 block in the bottom right (4 pixels)
        mask[8:10, 8:10] = 1
        # Parent mask is non-zero but does not intersect with the component
        parent_mask = np.zeros((10, 10), dtype=np.uint8)
        parent_mask[0:3, 0:3] = 1
        result = delete_nearby_segments_with_buffer(mask.copy(), parent_mask,
                                                     distance_threshold=1, size_threshold=5,
                                                     use_gpu=False, verbose=False)
        # Expect the small component to be removed
        self.assertEqual(np.count_nonzero(result), 0)

    def test_delete_touching_border_segments(self):
        # 3D mask of size 7x7x7, so components do not merge.
        mask = np.zeros((7, 7, 7), dtype=np.uint8)
        # Small component: one pixel at position [1,1,1] (touching the border by if_touches_border_3d logic)
        mask[1, 1, 1] = 1
        # Large component: 2x2x2 block in the center (rows 3–4, columns 3–4, slices 3–4) – 8 pixels
        mask[3:5, 3:5, 3:5] = 1
        # size_threshold=2: small (1 pixel) is removed, large (8 pixels) remains
        result = delete_touching_border_segments(mask, size_threshold=2, use_gpu=False, verbose=False)
        self.assertEqual(np.count_nonzero(result), 8)

    def test_fill_holes_2d(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False  # small hole
        result = fill_holes_2d(mask.copy(), max_hole_size=2, use_gpu=False)
        self.assertTrue(np.all(result))

    def test_fill_holes_3d(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = False  # small hole
        result = fill_holes_3d(mask.copy(), max_hole_size=2, use_gpu=False)
        self.assertTrue(np.all(result))


if __name__ == "__main__":
    unittest.main()
