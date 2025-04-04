import unittest
import numpy as np
from optimed import _cupy_available
from optimed.wrappers.calculations import (
    scipy_label,
    scipy_binary_dilation,
    scipy_binary_closing,
    scipy_binary_erosion,
    scipy_distance_transform_edt,
    scipy_minimum,
    scipy_sum,
    filter_mask,
    scipy_binary_opening,
    scipy_binary_fill_holes,
    scipy_median_filter
)

class TestArrayFunctions(unittest.TestCase):

    # ========================
    # Tests for scipy_label
    # ========================
    def test_scipy_label_cpu(self):
        input_array = np.array([
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=bool)

        labeled, num_labels = scipy_label(input_array, use_gpu=False)
        self.assertEqual(num_labels, 3, f"Expected 3 labels, got {num_labels}")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_label.")
    def test_scipy_label_gpu(self):
        input_array = np.array([
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=bool)

        labeled, num_labels = scipy_label(input_array, use_gpu=True)
        self.assertEqual(num_labels, 3, f"GPU: Expected 3 labels, got {num_labels}")

    # ========================
    # Tests for scipy_binary_dilation
    # ========================
    def test_scipy_binary_dilation_cpu(self):
        input_array = np.zeros((5, 5), dtype=bool)
        input_array[2, 2] = True  # center pixel True
        structure = np.ones((3, 3), dtype=bool)

        dilated = scipy_binary_dilation(input_array, structure=structure, iterations=1, use_gpu=False)
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        self.assertTrue(np.array_equal(dilated, expected), "CPU binary_dilation did not match expected 3x3 block.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_binary_dilation.")
    def test_scipy_binary_dilation_gpu(self):
        input_array = np.zeros((5, 5), dtype=bool)
        input_array[2, 2] = True
        structure = np.ones((3, 3), dtype=bool)

        dilated = scipy_binary_dilation(input_array, structure=structure, iterations=1, use_gpu=True)
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        self.assertTrue(np.array_equal(dilated, expected), "GPU binary_dilation did not match expected 3x3 block.")

    # ========================
    # Tests for scipy_binary_closing
    # ========================
    def test_scipy_binary_closing_cpu(self):
        # Create an input with a small gap.
        input_array = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=bool)
        structure = np.ones((3, 3), dtype=bool)
        closed = scipy_binary_closing(input_array, structure=structure, iterations=1, use_gpu=False)
        # With default constant-padding, the dilation fills the array, but erosion only keeps the center.
        expected = np.array([
            [False, False, False],
            [False, True, False],
            [False, False, False]
        ], dtype=bool)
        self.assertTrue(np.array_equal(closed, expected), "CPU binary_closing did not match expected result.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_binary_closing.")
    def test_scipy_binary_closing_gpu(self):
        input_array = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=bool)
        structure = np.ones((3, 3), dtype=bool)
        closed = scipy_binary_closing(input_array, structure=structure, iterations=1, use_gpu=True)
        expected = np.array([
            [False, False, False],
            [False, True, False],
            [False, False, False]
        ], dtype=bool)
        self.assertTrue(np.array_equal(closed, expected), "GPU binary_closing did not match expected result.")

    # ========================
    # Tests for scipy_binary_erosion
    # ========================
    def test_scipy_binary_erosion_cpu(self):
        # Create an input with a block of ones.
        input_array = np.array([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0]
        ], dtype=bool)
        structure = np.ones((3, 3), dtype=bool)
        eroded = scipy_binary_erosion(input_array, structure=structure, iterations=1, use_gpu=False)
        # With default constant-padding, none of the pixels have a full 3x3 neighborhood of True.
        expected = np.zeros((4, 4), dtype=bool)
        self.assertTrue(np.array_equal(eroded, expected), "CPU binary_erosion did not match expected result.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_binary_erosion.")
    def test_scipy_binary_erosion_gpu(self):
        input_array = np.array([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0]
        ], dtype=bool)
        structure = np.ones((3, 3), dtype=bool)
        eroded = scipy_binary_erosion(input_array, structure=structure, iterations=1, use_gpu=True)
        expected = np.zeros((4, 4), dtype=bool)
        self.assertTrue(np.array_equal(eroded, expected), "GPU binary_erosion did not match expected result.")

    # ========================
    # Tests for scipy_distance_transform_edt
    # ========================
    def test_scipy_distance_transform_edt_cpu(self):
        """
        To compute the distance from every pixel to the nearest True pixel,
        we invert the input (since EDT computes distances for zeros).
        """
        input_array = np.zeros((5, 5), dtype=bool)
        input_array[2, :] = True  # middle row True
        input_array[:, 2] = True  # middle column True

        # Invert the array so that originally True become False.
        dist_trans = scipy_distance_transform_edt(~input_array, use_gpu=False)
        # For the top-left corner (0,0), the nearest originally True pixel is at (0,2) or (2,0): distance = 2.
        self.assertAlmostEqual(dist_trans[0, 0], 2.0, places=3)
        # The center (2,2) should have a distance of 0.
        self.assertAlmostEqual(dist_trans[2, 2], 0.0, places=3, msg="Center point distance should be 0.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_distance_transform_edt.")
    def test_scipy_distance_transform_edt_gpu(self):
        input_array = np.zeros((5, 5), dtype=bool)
        input_array[2, :] = True
        input_array[:, 2] = True

        dist_trans = scipy_distance_transform_edt(~input_array, use_gpu=True)
        self.assertAlmostEqual(dist_trans[0, 0], 2.0, places=3)
        self.assertAlmostEqual(dist_trans[2, 2], 0.0, places=3)

    # ========================
    # Tests for scipy_minimum
    # ========================
    def test_scipy_minimum_cpu(self):
        input_array = np.array([
            [3, 4, 5],
            [7, 1, 2],
            [9, 8, 6]
        ], dtype=np.int32)
        label_array = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ], dtype=np.int32)
        min_val_region1 = scipy_minimum(input_array, label_array, 1, use_gpu=False)
        min_val_region2 = scipy_minimum(input_array, label_array, 2, use_gpu=False)
        self.assertEqual(min_val_region1, 1, f"Expected min of 1 for region 1, got {min_val_region1}")
        self.assertEqual(min_val_region2, 2, f"Expected min of 2 for region 2, got {min_val_region2}")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_minimum.")
    def test_scipy_minimum_gpu(self):
        input_array = np.array([
            [3, 4, 5],
            [7, 1, 2],
            [9, 8, 6]
        ], dtype=np.int32)
        label_array = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ], dtype=np.int32)
        min_val_region1 = scipy_minimum(input_array, label_array, 1, use_gpu=True)
        min_val_region2 = scipy_minimum(input_array, label_array, 2, use_gpu=True)
        self.assertEqual(min_val_region1, 1)
        self.assertEqual(min_val_region2, 2)

    # ========================
    # Tests for scipy_sum
    # ========================
    def test_scipy_sum_cpu(self):
        input_array = np.array([
            [3, 4, 5],
            [7, 1, 2],
            [9, 8, 6]
        ], dtype=np.int32)
        label_array = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ], dtype=np.int32)
        sum_val_region1 = scipy_sum(input_array, label_array, 1, use_gpu=False)
        sum_val_region2 = scipy_sum(input_array, label_array, 2, use_gpu=False)
        self.assertEqual(sum_val_region1, 15, f"Expected sum of 15 for region 1, got {sum_val_region1}")
        self.assertEqual(sum_val_region2, 30, f"Expected sum of 30 for region 2, got {sum_val_region2}")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_sum.")
    def test_scipy_sum_gpu(self):
        input_array = np.array([
            [3, 4, 5],
            [7, 1, 2],
            [9, 8, 6]
        ], dtype=np.int32)
        label_array = np.array([
            [1, 1, 2],
            [1, 1, 2],
            [2, 2, 2]
        ], dtype=np.int32)
        sum_val_region1 = scipy_sum(input_array, label_array, 1, use_gpu=True)
        sum_val_region2 = scipy_sum(input_array, label_array, 2, use_gpu=True)
        self.assertEqual(sum_val_region1, 15)
        self.assertEqual(sum_val_region2, 30)

    # ========================
    # Tests for filter_mask
    # ========================
    def test_filter_mask_cpu(self):
        mask = np.array([
            [0, 1, 2],
            [3, 0, 4],
            [5, 1, 1]
        ], dtype=np.int32)
        lbls_to_keep = [1, 5]
        filtered = filter_mask(mask, lbls_to_keep, use_gpu=False)
        expected = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [5, 1, 1]
        ], dtype=np.int32)
        self.assertTrue(np.array_equal(filtered, expected), "filter_mask CPU result is incorrect.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for filter_mask.")
    def test_filter_mask_gpu(self):
        mask = np.array([
            [0, 1, 2],
            [3, 0, 4],
            [5, 1, 1]
        ], dtype=np.int32)
        lbls_to_keep = [1, 5]
        filtered = filter_mask(mask, lbls_to_keep, use_gpu=True)
        expected = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [5, 1, 1]
        ], dtype=np.int32)
        self.assertTrue(np.array_equal(filtered, expected), "filter_mask GPU result is incorrect.")

    # ========================
    # Tests for scipy_binary_opening
    # ========================
    def test_scipy_binary_opening_cpu(self):
        input_array = np.array([
            [False, False, False],
            [False, True, False],
            [False, False, False]
        ], dtype=bool)
        structure = np.ones((3,3), dtype=bool)
        opened = scipy_binary_opening(input_array, structure=structure, iterations=1, use_gpu=False)
        expected = np.zeros((3,3), dtype=bool)
        self.assertTrue(np.array_equal(opened, expected), "CPU binary_opening did not remove isolated pixel.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_binary_opening.")
    def test_scipy_binary_opening_gpu(self):
        input_array = np.array([
            [False, False, False],
            [False, True, False],
            [False, False, False]
        ], dtype=bool)
        structure = np.ones((3,3), dtype=bool)
        opened = scipy_binary_opening(input_array, structure=structure, iterations=1, use_gpu=True)
        expected = np.zeros((3,3), dtype=bool)
        self.assertTrue(np.array_equal(opened, expected), "GPU binary_opening did not remove isolated pixel.")

    # ========================
    # Tests for scipy_binary_fill_holes
    # ========================
    def test_scipy_binary_fill_holes_cpu(self):
        input_array = np.array([
            [True, True, True],
            [True, False, True],
            [True, True, True]
        ], dtype=bool)
        filled = scipy_binary_fill_holes(input_array, use_gpu=False)
        expected = np.array([
            [True, True, True],
            [True, True, True],
            [True, True, True]
        ], dtype=bool)
        self.assertTrue(np.array_equal(filled, expected), "CPU binary_fill_holes did not fill hole.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_binary_fill_holes.")
    def test_scipy_binary_fill_holes_gpu(self):
        input_array = np.array([
            [True, True, True],
            [True, False, True],
            [True, True, True]
        ], dtype=bool)
        filled = scipy_binary_fill_holes(input_array, use_gpu=True)
        expected = np.array([
            [True, True, True],
            [True, True, True],
            [True, True, True]
        ], dtype=bool)
        self.assertTrue(np.array_equal(filled, expected), "GPU binary_fill_holes did not fill hole.")

    # ========================
    # Tests for scipy_median_filter
    # ========================
    def test_scipy_median_filter_cpu(self):
        input_array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32)
        filtered = scipy_median_filter(input_array, size=(3,3), use_gpu=False)
        self.assertEqual(filtered[1,1], 5, "CPU median_filter center value mismatch.")
        self.assertEqual(filtered.shape, input_array.shape, "CPU median_filter changed shape.")

    @unittest.skipUnless(_cupy_available, "cupy not installed. Skipping GPU test for scipy_median_filter.")
    def test_scipy_median_filter_gpu(self):
        input_array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.int32)
        filtered = scipy_median_filter(input_array, size=(3,3), use_gpu=True)
        self.assertEqual(filtered[1,1], 5, "GPU median_filter center value mismatch.")
        self.assertEqual(filtered.shape, input_array.shape, "GPU median_filter changed shape.")


if __name__ == '__main__':
    unittest.main()
