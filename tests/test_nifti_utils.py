import unittest
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os
import warnings
from optimed.wrappers.nifti import *
from tempfile import TemporaryDirectory

class TestNiftiFunctions(unittest.TestCase):

    def setUp(self):
        self.test_dir = TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_nifti_nibabel(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        
        save_path = os.path.join(self.test_dir.name, 'test_nibabel.nii.gz')
        nib.save(img, save_path)

        loaded_img = load_nifti(save_path, canonical=False, engine='nibabel')
        self.assertIsInstance(loaded_img, nib.Nifti1Image)
        np.testing.assert_array_equal(loaded_img.get_fdata(), data)

    def test_load_nifti_sitk(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        img = sitk.GetImageFromArray(data)
        
        save_path = os.path.join(self.test_dir.name, 'test_sitk.nii.gz')
        sitk.WriteImage(img, save_path)

        loaded_img = load_nifti(save_path, canonical=False, engine='sitk')
        self.assertIsInstance(loaded_img, sitk.Image)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded_img), data)

    def test_save_nifti_nibabel(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        
        affine = np.eye(4)
        ref_img = nib.Nifti1Image(data, affine)
        
        save_path = os.path.join(self.test_dir.name, 'saved_nibabel.nii.gz')
        save_nifti(ndarray=data, ref_image=ref_img, dtype=np.float32, save_to=save_path, engine='nibabel')

        loaded_img = nib.load(save_path)
        np.testing.assert_array_equal(loaded_img.get_fdata(), data)

    def test_save_nifti_sitk(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        
        ref_img = sitk.GetImageFromArray(data)

        save_path = os.path.join(self.test_dir.name, 'saved_sitk.nii.gz')
        save_nifti(ndarray=data, ref_image=ref_img, dtype=np.float32, save_to=save_path, engine='sitk')

        loaded_img = sitk.ReadImage(save_path)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded_img), data)

    def test_save_nifti_with_uint8(self):
        data = np.random.randint(0, 256, size=(5, 5, 5)).astype(np.uint8)
        
        ref_img_nib = nib.Nifti1Image(data, np.eye(4))
        save_path_nib = os.path.join(self.test_dir.name, 'saved_nibabel_uint8.nii.gz')
        save_nifti(ndarray=data, ref_image=ref_img_nib, dtype=np.uint8, save_to=save_path_nib, engine='nibabel')
        loaded_img_nib = nib.load(save_path_nib)
        np.testing.assert_array_equal(loaded_img_nib.get_fdata(), data)

        ref_img_sitk = sitk.GetImageFromArray(data)
        save_path_sitk = os.path.join(self.test_dir.name, 'saved_sitk_uint8.nii.gz')
        save_nifti(ndarray=data, ref_image=ref_img_sitk, dtype=np.uint8, save_to=save_path_sitk, engine='sitk')
        loaded_img_sitk = sitk.ReadImage(save_path_sitk)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded_img_sitk), data)

    def test_maybe_convert_nifti_image_in_dtype(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        ref_img_nib = nib.Nifti1Image(data, np.eye(4))

        converted_img_nib = maybe_convert_nifti_image_in_dtype(ref_img_nib, to_dtype='int32', engine='nibabel')
        self.assertEqual(str(converted_img_nib.header.get_data_dtype()), 'int32')

        ref_img_sitk = sitk.GetImageFromArray(data)
        converted_img_sitk = maybe_convert_nifti_image_in_dtype(ref_img_sitk, to_dtype='int32', engine='sitk')
        self.assertEqual(converted_img_sitk.GetPixelIDTypeAsString(), '32-bit signed integer')

    def test_get_image_orientation(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        ref_img_nib = nib.Nifti1Image(data, np.eye(4))
        orientation_nib = get_image_orientation(ref_img_nib, engine='nibabel')
        self.assertEqual(orientation_nib, ('R', 'A', 'S'))

        ref_img_sitk = sitk.GetImageFromArray(data)
        ref_img_sitk.SetOrigin([0, 0, 0])
        ref_img_sitk.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        orientation_sitk = get_image_orientation(ref_img_sitk, engine='sitk')
        self.assertEqual(orientation_sitk, ('R', 'A', 'S'))

    def test_empty_img_like(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        ref_img_nib = nib.Nifti1Image(data, np.eye(4))
        empty_img_nib = empty_img_like(ref_img_nib, engine='nibabel')
        self.assertTrue(np.all(empty_img_nib.get_fdata() == 0))

        ref_img_sitk = sitk.GetImageFromArray(data)
        empty_img_sitk = empty_img_like(ref_img_sitk, engine='sitk')
        empty_img_sitk_data = sitk.GetArrayFromImage(empty_img_sitk)
        self.assertTrue(np.all(empty_img_sitk_data == 0))

    def test_as_closest_canonical_and_undo_canonical(self):
        data = np.random.rand(5, 5, 5).astype(np.float32)
        affine = np.array([[0, -1, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        img_nib = nib.Nifti1Image(data, affine)
        canonical_img = as_closest_canonical(img_nib)
        reverted_img = undo_canonical(canonical_img, img_nib)
        
        np.testing.assert_array_equal(reverted_img.get_fdata(), img_nib.get_fdata())
        np.testing.assert_array_equal(reverted_img.affine, img_nib.affine)

        ref_img_sitk = sitk.GetImageFromArray(data)
        ref_img_sitk.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        canonical_img_sitk = as_closest_canonical(ref_img_sitk, engine='sitk')
        reverted_img_sitk = undo_canonical(canonical_img_sitk, ref_img_sitk, engine='sitk')

        np.testing.assert_array_equal(sitk.GetArrayFromImage(reverted_img_sitk), sitk.GetArrayFromImage(ref_img_sitk))
        self.assertEqual(reverted_img_sitk.GetDirection(), ref_img_sitk.GetDirection())

    def test_invalid_engine(self):
        with self.assertRaises(AssertionError):
            load_nifti("fake_path.nii.gz", engine="invalid_engine")

        data = np.random.rand(5, 5, 5).astype(np.float32)
        ref_img = nib.Nifti1Image(data, np.eye(4))
        with self.assertRaises(AssertionError):
            save_nifti(data, ref_img, np.float32, "fake_path.nii.gz", engine="invalid_engine")

    def test_split_image_nibabel(self):
        data = np.random.rand(10, 10, 10).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        
        parts = 2
        split_imgs = split_image(img, parts, axis=0)
        
        self.assertEqual(len(split_imgs), parts)
        for i, split_img in enumerate(split_imgs):
            self.assertIsInstance(split_img, nib.Nifti1Image)
            expected_shape = list(data.shape)
            expected_shape[0] = expected_shape[0] // parts
            self.assertEqual(split_img.shape, tuple(expected_shape))

    def test_split_image_sitk(self):
        data = np.random.rand(10, 10, 10).astype(np.float32)
        img = sitk.GetImageFromArray(data)
        
        parts = 2
        split_imgs = split_image(img, parts, axis=0)
        
        self.assertEqual(len(split_imgs), parts)
        for i, split_img in enumerate(split_imgs):
            self.assertIsInstance(split_img, sitk.Image)
            expected_shape = list(data.shape)
            expected_shape[0] = expected_shape[0] // parts
            self.assertEqual(sitk.GetArrayFromImage(split_img).shape, tuple(expected_shape))


if __name__ == '__main__':
    unittest.main()
