import unittest
import numpy as np
from optimed.wrappers.numpy import *

class TestArrayFunctions(unittest.TestCase):

    def test_mean(self):
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(mean(arr), 3.0)
        self.assertEqual(mean(arr, exclude_nan=True), 3.0)

        arr_with_nan = np.array([1, 2, np.nan, 4, 5])
        self.assertTrue(np.isnan(mean(arr_with_nan)))
        self.assertEqual(mean(arr_with_nan, exclude_nan=True), 3.0)

    def test_sum(self):
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(sum(arr), 15)
        self.assertTrue(np.array_equal(sum(arr, keepdims=True), np.array([15])))

    def test_median(self):
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(median(arr), 3)

        arr_with_nan = np.array([1, 2, np.nan, 4, 5])
        self.assertTrue(np.isnan(median(arr_with_nan)))
        self.assertEqual(median(arr_with_nan, exclude_nan=True), 3)

    def test_std(self):
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(std(arr), np.std(arr))

        arr_with_nan = np.array([1, 2, np.nan, 4, 5])
        self.assertTrue(np.isnan(std(arr_with_nan)))
        self.assertEqual(std(arr_with_nan, exclude_nan=True), np.std([1, 2, 4, 5]))

    def test_reshape(self):
        arr = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(reshape(arr, (2, 3)), arr.reshape(2, 3)))
        with self.assertRaises(ValueError):
            reshape(arr, (3, 3))

    def test_concatenate(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        self.assertTrue(np.array_equal(concatenate([arr1, arr2]), np.concatenate([arr1, arr2])))

    def test_linspace(self):
        result = linspace(0, 1, num=5)
        expected = np.linspace(0, 1, num=5)
        self.assertTrue(np.array_equal(result, expected))

        with self.assertRaises(ValueError):
            linspace(0, 1, num=0)

    def test_argmax(self):
        arr = np.array([1, 3, 2, 5, 4])
        self.assertEqual(argmax(arr), 3)

    def test_clip(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = clip(arr, 2, 4)
        self.assertTrue(np.array_equal(result, np.array([2, 2, 3, 4, 4])))

    def test_unique(self):
        arr = np.array([1, 2, 2, 3, 3, 3])
        unique_arr = unique(arr)
        self.assertTrue(np.array_equal(unique_arr, np.array([1, 2, 3])))

        unique_arr, counts = unique(arr, return_counts=True)
        self.assertTrue(np.array_equal(unique_arr, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(counts, np.array([1, 2, 3])))

    def test_zeros_like(self):
        arr = np.array([1, 2, 3])
        result = zeros_like(arr)
        self.assertTrue(np.array_equal(result, np.zeros_like(arr)))

    def test_ones_like(self):
        arr = np.array([1, 2, 3])
        result = ones_like(arr)
        self.assertTrue(np.array_equal(result, np.ones_like(arr)))

    def test_full_like(self):
        arr = np.array([1, 2, 3])
        result = full_like(arr, 7)
        self.assertTrue(np.array_equal(result, np.full_like(arr, 7)))


if __name__ == '__main__':
    unittest.main()
