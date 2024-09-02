import unittest
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os
import warnings

from optimed.wrappers.nifti import *

from tempfile import TemporaryDirectory

# Assume the load_nifti and save_nifti functions are defined in the module 'nifti_utils'
# from nifti_utils import load_nifti, save_nifti

class TestNiftiFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to save test files
        self.test_dir = TemporaryDirectory()

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_load_nifti_nibabel(self):
        # Create a dummy nibabel image
        data = np.random.rand(5, 5, 5).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        
        # Save the image to the temporary directory
        save_path = os.path.join(self.test_dir.name, 'test_nibabel.nii.gz')
        nib.save(img, save_path)

        # Load the image using load_nifti with nibabel
        loaded_img = load_nifti(save_path, canonical=False, engine='nibabel')
        self.assertIsInstance(loaded_img, nib.Nifti1Image)
        np.testing.assert_array_equal(loaded_img.get_fdata(), data)

    def test_load_nifti_sitk(self):
        # Create a dummy SimpleITK image
        data = np.random.rand(5, 5, 5).astype(np.float32)
        img = sitk.GetImageFromArray(data)
        
        # Save the image to the temporary directory
        save_path = os.path.join(self.test_dir.name, 'test_sitk.nii.gz')
        sitk.WriteImage(img, save_path)

        # Load the image using load_nifti with SimpleITK
        loaded_img = load_nifti(save_path, canonical=False, engine='sitk')
        self.assertIsInstance(loaded_img, sitk.Image)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded_img), data)

    def test_save_nifti_nibabel(self):
        # Create a dummy numpy array
        data = np.random.rand(5, 5, 5).astype(np.float32)
        
        # Create a reference nibabel image
        affine = np.eye(4)
        ref_img = nib.Nifti1Image(data, affine)
        
        # Save the array as a NIfTI file using nibabel
        save_path = os.path.join(self.test_dir.name, 'saved_nibabel.nii.gz')
        save_nifti(data, ref_img, np.float32, save_path, engine='nibabel')

        # Load the saved image and check its contents
        loaded_img = nib.load(save_path)
        np.testing.assert_array_equal(loaded_img.get_fdata(), data)

    def test_save_nifti_sitk(self):
        # Create a dummy numpy array
        data = np.random.rand(5, 5, 5).astype(np.float32)
        
        # Create a reference SimpleITK image
        ref_img = sitk.GetImageFromArray(data)
        
        # Save the array as a NIfTI file using SimpleITK
        save_path = os.path.join(self.test_dir.name, 'saved_sitk.nii.gz')
        save_nifti(data, ref_img, np.float32, save_path, engine='sitk')

        # Load the saved image and check its contents
        loaded_img = sitk.ReadImage(save_path)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded_img), data)

    def test_save_nifti_with_uint8(self):
        # Test saving and loading with uint8 dtype for both nibabel and sitk
        data = np.random.randint(0, 256, size=(5, 5, 5)).astype(np.uint8)
        
        # Test with nibabel
        ref_img_nib = nib.Nifti1Image(data, np.eye(4))
        save_path_nib = os.path.join(self.test_dir.name, 'saved_nibabel_uint8.nii.gz')
        save_nifti(data, ref_img_nib, np.uint8, save_path_nib, engine='nibabel')
        loaded_img_nib = nib.load(save_path_nib)
        np.testing.assert_array_equal(loaded_img_nib.get_fdata(), data)

        # Test with sitk
        ref_img_sitk = sitk.GetImageFromArray(data)
        save_path_sitk = os.path.join(self.test_dir.name, 'saved_sitk_uint8.nii.gz')
        save_nifti(data, ref_img_sitk, np.uint8, save_path_sitk, engine='sitk')
        loaded_img_sitk = sitk.ReadImage(save_path_sitk)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded_img_sitk), data)

    def test_invalid_engine(self):
        # Test with an invalid engine for load_nifti
        with self.assertRaises(AssertionError):
            load_nifti("fake_path.nii.gz", engine="invalid_engine")

        # Test with an invalid engine for save_nifti
        data = np.random.rand(5, 5, 5).astype(np.float32)
        ref_img = nib.Nifti1Image(data, np.eye(4))
        with self.assertRaises(AssertionError):
            save_nifti(data, ref_img, np.float32, "fake_path.nii.gz", engine="invalid_engine")

if __name__ == '__main__':
    unittest.main()
