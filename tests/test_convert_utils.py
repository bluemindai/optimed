import unittest
from optimed.processes.convert import (
    convert_dcm_to_nifti,
    convert_nrrd_to_nifti,
    conver_mha_to_nifti
)

class TestConvert(unittest.TestCase):
    # def test_convert_dcm_to_nifti(self):
    #     result = convert_dcm_to_nifti("local/dicom", "local/ct.nii.gz", return_object=True)
    #     self.assertIsNotNone(result)

    def test_convert_nrrd_to_nifti(self):
        result = convert_nrrd_to_nifti("local/ct.nrrd", "local/ct.nii.gz", return_object=True)
        self.assertIsNotNone(result)

    def test_convert_mha_to_nifti(self):
        result = conver_mha_to_nifti("local/ct.mha", "local/ct.nii.gz", return_object=True)
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
