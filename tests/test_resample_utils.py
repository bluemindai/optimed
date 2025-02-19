import unittest
import numpy as np
import nibabel as nib
from optimed.processes.resample import (
    change_spacing,
    change_spacing_of_affine,
    resample_img,
    resample_img_cucim
)
from optimed import _cucim_available

class TestResample(unittest.TestCase):
    def setUp(self):
        # Create a test 3D image of size 10x10x10 with an identity affine matrix
        self.data = np.random.rand(10, 10, 10)
        self.affine = np.eye(4)
        self.img = nib.Nifti1Image(self.data, self.affine)

    def test_change_spacing(self):
        # Change spacing from 1.0 to 2.0 -> expect shape (5, 5, 5)
        new_spacing = 2.0
        img2 = change_spacing(self.img, new_spacing=new_spacing, order=0, nr_cpus=1)
        self.assertEqual(img2.shape, (5, 5, 5))

    def test_change_spacing_with_target_shape(self):
        # Specify target_shape for precise shape change
        target_shape = (8, 8, 8)
        img2 = change_spacing(self.img, new_spacing=2.0, target_shape=target_shape, order=0, nr_cpus=1)
        self.assertEqual(img2.shape, target_shape)

    def test_change_spacing_without_resampling(self):
        # If spacing remains unchanged, returns the original image
        img2 = change_spacing(self.img, new_spacing=1.0, order=0, nr_cpus=1)
        np.testing.assert_array_equal(img2.get_fdata(), self.img.get_fdata())

    def test_change_spacing_of_affine(self):
        # Check that change_spacing_of_affine scales the diagonal by the zoom factor
        zoom = 2.0
        new_affine = change_spacing_of_affine(self.affine, zoom=zoom)
        self.assertAlmostEqual(new_affine[0, 0], 1.0 / zoom)
        self.assertAlmostEqual(new_affine[1, 1], 1.0 / zoom)
        self.assertAlmostEqual(new_affine[2, 2], 1.0 / zoom)

    def test_resample_img(self):
        # Test resample_img with zoom=[0.5, 0.5, 0.5]
        zoom = np.array([0.5, 0.5, 0.5])
        resampled = resample_img(self.data, zoom=zoom, order=0, nr_cpus=1)
        expected_shape = tuple((np.array(self.data.shape) * zoom).round().astype(int))
        self.assertEqual(resampled.shape, expected_shape)

    @unittest.skipUnless(_cucim_available, "CuCIM is not available")
    def test_resample_img_cucim(self):
        # Test resample_img_cucim with zoom=0.5 scaling (scalar value)
        zoom = 0.5
        resampled = resample_img_cucim(self.data, zoom=zoom, order=0)
        expected_shape = tuple((np.array(self.data.shape) * zoom).round().astype(int))
        self.assertEqual(resampled.shape, expected_shape)

if __name__ == "__main__":
    unittest.main()
