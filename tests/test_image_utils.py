import unittest
import numpy as np
from optimed.processes.image import (
    rescale_intensity,
    denoise_image,
    get_mask_by_threshold,
    blur_inside_roi,
    blur_outside_roi
)

class TestImageFunctions(unittest.TestCase):
    def test_rescale_intensity(self):
        img = np.array([-200, -125, 100, 300], dtype=np.float32)
        out = rescale_intensity(img, lower=-125, upper=225)
        self.assertTrue(np.all((out >= 0) & (out <= 255)))

    def test_denoise_image(self):
        img = np.random.rand(5, 5, 5).astype(np.float32)
        out = denoise_image(img, sigma=1.0, mode='gaussian')
        self.assertEqual(out.shape, img.shape)

    def test_get_mask_by_threshold(self):
        img = np.array([[[1, 3], [5, 10]]], dtype=np.float32)
        mask = get_mask_by_threshold(img, threshold=4, above=True)
        self.assertTrue(mask[0, 0, 0] == False and mask[0, 1, 1] == True)

    def test_blur_inside_roi(self):
        img = np.zeros((5, 5, 5), dtype=np.float32)
        img[2, 2, 2] = 10.0
        roi = np.zeros_like(img, dtype=bool)
        roi[2, 2, 2] = True
        out = blur_inside_roi(img, roi, sigma=2)
        self.assertFalse(np.allclose(img[2, 2, 2], out[2, 2, 2]))

    def test_blur_outside_roi(self):
        img = np.zeros((5, 5, 5), dtype=np.float32)
        img[2, 2, 2] = 10.0
        roi = np.zeros_like(img, dtype=bool)
        roi[2, 2, 2] = True
        out = blur_outside_roi(img, roi, sigma=2)
        self.assertTrue(np.allclose(img[2, 2, 2], out[2, 2, 2]))
        self.assertFalse(np.allclose(img[0, 0, 0], out[0, 0, 0]))

if __name__ == "__main__":
    unittest.main()
