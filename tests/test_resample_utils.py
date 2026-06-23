import unittest
import numpy as np
import nibabel as nib
import sys
import types
from unittest.mock import patch
from optimed.processes.resample import (
    change_spacing,
    change_spacing_of_affine,
    resample_img,
    resample_img_cucim,
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
        img2 = change_spacing(
            self.img, new_spacing=2.0, target_shape=target_shape, order=0, nr_cpus=1
        )
        self.assertEqual(img2.shape, target_shape)

    def test_change_spacing_2d_with_target_shape_without_new_spacing(self):
        data = np.random.rand(10, 10)
        img = nib.Nifti1Image(data, np.eye(4))

        img2 = change_spacing(
            img, new_spacing=None, target_shape=(5, 5), order=0, nr_cpus=1
        )

        self.assertEqual(img2.shape, (5, 5))

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

    def test_change_spacing_gpu_does_not_forward_nr_cpus(self):
        with (
            patch("optimed.processes.resample._cupy_available", True),
            patch("optimed.processes.resample._cucim_available", True),
            patch(
                "optimed.processes.resample.resample_img_cucim",
                return_value=np.zeros((5, 5, 5)),
            ) as resample_gpu,
        ):
            img2 = change_spacing(
                self.img, new_spacing=2.0, order=0, nr_cpus=1, use_gpu=True
            )

        self.assertEqual(img2.shape, (5, 5, 5))
        _, kwargs = resample_gpu.call_args
        self.assertEqual(set(kwargs), {"zoom", "order"})

    def test_resample_img_cucim_handles_2d_and_4d_shapes(self):
        cupy = types.ModuleType("cupy")
        cupy.asarray = np.asarray
        cupy.asnumpy = np.asarray
        cupy.stack = np.stack

        transform = types.ModuleType("cucim.skimage.transform")
        transform.resize = lambda img, output_shape, **kwargs: np.zeros(
            output_shape, dtype=np.asarray(img).dtype
        )
        cucim = types.ModuleType("cucim")
        skimage = types.ModuleType("cucim.skimage")

        modules = {
            "cupy": cupy,
            "cucim": cucim,
            "cucim.skimage": skimage,
            "cucim.skimage.transform": transform,
        }
        with (
            patch.dict(sys.modules, modules),
            patch("optimed.processes.resample._cupy_available", True),
            patch("optimed.processes.resample._cucim_available", True),
        ):
            resampled_2d = resample_img_cucim(
                np.zeros((10, 10)), zoom=np.array([0.5, 0.5, 1.0]), order=0
            )
            resampled_4d = resample_img_cucim(
                np.zeros((10, 10, 10, 2)), zoom=np.array([0.5, 0.5, 0.5]), order=0
            )

        self.assertEqual(resampled_2d.shape, (5, 5))
        self.assertEqual(resampled_4d.shape, (5, 5, 5, 2))

    @unittest.skipUnless(_cucim_available, "CuCIM is not available")
    def test_resample_img_cucim(self):
        # Test resample_img_cucim with zoom=0.5 scaling (scalar value)
        zoom = 0.5
        resampled = resample_img_cucim(self.data, zoom=zoom, order=0)
        expected_shape = tuple((np.array(self.data.shape) * zoom).round().astype(int))
        self.assertEqual(resampled.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
