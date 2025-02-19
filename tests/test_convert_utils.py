import os
import unittest
import tempfile
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import nrrd
import slicerio

# Import the conversion functions from your module.
# Adjust the import path if needed.
from optimed.processes.convert import (
    convert_sitk_to_nibabel,
    convert_nibabel_to_sitk,
    convert_dcm_to_nifti,
    convert_nrrd_to_nifti,
    conver_mha_to_nifti,
    convert_nifti_to_nrrd
)

# Dummy reader for simulating DICOM series conversion.
class DummyImageSeriesReader:
    def __init__(self):
        self.file_names = None

    def GetGDCMSeriesFileNames(self, dicom_path):
        # Return a dummy list of file names so that the existence check passes.
        return ["dummy1", "dummy2"]

    def SetFileNames(self, file_names):
        self.file_names = file_names

    def Execute(self):
        # Create a simple 5x5x5 image (with ones) for testing.
        data = np.ones((5, 5, 5), dtype=np.int16)
        image = sitk.GetImageFromArray(data)
        image.SetOrigin((0, 0, 0))
        image.SetSpacing((1, 1, 1))
        return image

class TestConvert(unittest.TestCase):

    def test_convert_sitk_to_nibabel(self):
        # Create a dummy SimpleITK image.
        data = np.random.rand(10, 10, 10)
        sitk_image = sitk.GetImageFromArray(data)
        sitk_image.SetOrigin((1.0, 2.0, 3.0))
        sitk_image.SetSpacing((1.0, 1.0, 1.0))
        
        # Convert to NIfTI using the provided function.
        nib_image = convert_sitk_to_nibabel(sitk_image)
        
        # Check that the output is a nibabel NIfTI image and that the data is preserved.
        self.assertIsInstance(nib_image, nib.Nifti1Image)
        np.testing.assert_allclose(nib_image.get_fdata(), data)

    def test_convert_nibabel_to_sitk(self):
        # Create a dummy nibabel image.
        data = np.random.rand(8, 8, 8)
        affine = np.eye(4)
        affine[:3, 3] = [5, 5, 5]
        nib_image = nib.Nifti1Image(data, affine)
        
        # Convert to SimpleITK.
        sitk_image = convert_nibabel_to_sitk(nib_image)
        
        # Check that the result is a SimpleITK image and that the image data is preserved.
        self.assertTrue(isinstance(sitk_image, sitk.Image))
        np.testing.assert_allclose(sitk.GetArrayFromImage(sitk_image), data)

    def test_convert_dcm_to_nifti(self):
        # Temporarily override SimpleITK's ImageSeriesReader to simulate DICOM reading.
        original_reader = sitk.ImageSeriesReader
        sitk.ImageSeriesReader = lambda: DummyImageSeriesReader()

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a dummy DICOM series directory.
            dicom_path = os.path.join(tmpdirname, "dicom_series")
            os.mkdir(dicom_path)
            # Create a dummy file inside the directory so that the exists() check passes.
            dummy_file = os.path.join(dicom_path, "dummy.dcm")
            with open(dummy_file, "w") as f:
                f.write("dummy content")
            
            nifti_file = os.path.join(tmpdirname, "output.nii.gz")
            # Call the conversion function.
            result = convert_dcm_to_nifti(
                dicom_path, nifti_file,
                permute_axes=False, return_object=True, return_type='sitk'
            )
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(nifti_file))
        
        # Restore the original ImageSeriesReader.
        sitk.ImageSeriesReader = original_reader

    def test_convert_nrrd_to_nifti(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a dummy NRRD file.
            nrrd_file = os.path.join(tmpdirname, "dummy.nrrd")
            nifti_file = os.path.join(tmpdirname, "output.nii.gz")
            data = np.random.rand(5, 5, 5)
            header = {
                'space directions': np.eye(3).tolist(),
                'space origin': [0, 0, 0],
                'space': 'left-posterior-superior'
            }
            nrrd.write(nrrd_file, data, header)
            
            # Convert the NRRD file to NIfTI.
            result = convert_nrrd_to_nifti(nrrd_file, nifti_file, return_object=True, return_type='nibabel')
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(nifti_file))

    def test_conver_mha_to_nifti(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a dummy MHA file using SimpleITK.
            mha_file = os.path.join(tmpdirname, "dummy.mha")
            nifti_file = os.path.join(tmpdirname, "output.nii.gz")
            data = np.random.rand(10, 10, 10)
            image = sitk.GetImageFromArray(data)
            sitk.WriteImage(image, mha_file)
            
            # Convert the MHA file to NIfTI.
            result = conver_mha_to_nifti(mha_file, nifti_file, return_object=True, return_type='nibabel')
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(nifti_file))

    def test_convert_nifti_to_nrrd(self):
        # Monkey-patch slicerio functions to bypass actual segmentation handling.
        original_read = slicerio.read_segmentation
        original_write = slicerio.write_segmentation

        def dummy_read_segmentation(filename):
            # Return an empty segmentation dictionary.
            return {}

        def dummy_write_segmentation(filename, segmentation):
            # Write a dummy NRRD header to the file.
            with open(filename, "w") as f:
                f.write("NRRD_dummy_header\n\ndummy content")

        slicerio.read_segmentation = dummy_read_segmentation
        slicerio.write_segmentation = dummy_write_segmentation

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a dummy NIfTI segmentation file.
            nifti_file = os.path.join(tmpdirname, "seg.nii.gz")
            data = np.zeros((5, 5, 5), dtype=np.int16)
            # Introduce a segmentation label.
            data[2, 2, 2] = 1
            affine = np.eye(4)
            nifti_image = nib.Nifti1Image(data, affine)
            nib.save(nifti_image, nifti_file)

            output_seg_filename = os.path.join(tmpdirname, "seg.seg.nrrd")
            # Convert the NIfTI segmentation to a Slicer segmentation (NRRD).
            convert_nifti_to_nrrd(nifti_file, output_seg_filename, segment_metadata=None, verbose=False)
            self.assertTrue(os.path.exists(output_seg_filename))
            with open(output_seg_filename, "r") as f:
                content = f.read()
            self.assertIn("NRRD_dummy_header", content)

        # Restore the original slicerio functions.
        slicerio.read_segmentation = original_read
        slicerio.write_segmentation = original_write

if __name__ == "__main__":
    unittest.main()
