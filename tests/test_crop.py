import unittest
import numpy as np
import nibabel as nib
from optimed.processes.crop import (
    get_bbox_from_mask,
    crop_to_bbox,
    crop_by_xyz_boudaries,
    crop_to_mask,
    undo_crop
)

class TestCrop(unittest.TestCase):
    def test_get_bbox_from_mask(self):
        data = np.zeros((10, 10, 10))
        data[2:5, 3:6, 4:7] = 1
        bbox = get_bbox_from_mask(data, outside_value=0, addon=1, verbose=False)
        self.assertEqual(bbox, [[1, 6], [2, 7], [3, 8]])

    def test_crop_to_bbox(self):
        data = np.random.rand(5, 5, 5)
        bbox = [[1, 4], [1, 4], [1, 4]]
        cropped = crop_to_bbox(data, bbox)
        self.assertEqual(cropped.shape, (3, 3, 3))

    def test_crop_by_xyz_boundaries(self):
        data = nib.Nifti1Image(np.random.rand(6, 6, 6), np.eye(4))
        cropped = crop_by_xyz_boudaries(data, x_start=2, x_end=5, y_start=1, y_end=4, z_start=0, z_end=3)
        self.assertEqual(cropped.shape, (3, 3, 3))

    def test_crop_to_mask(self):
        img_data = nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4))
        mask_data = nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4))
        cropped_img, bbox = crop_to_mask(img_data, mask_data, addon=[0, 0, 0])
        self.assertEqual(cropped_img.shape, (4, 4, 4))
        self.assertEqual(bbox, [[0, 4], [0, 4], [0, 4]])

    def test_undo_crop(self):
        ref_img = nib.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
        cropped_img = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
        bbox = [[1, 4], [1, 4], [1, 4]]
        undone = undo_crop(cropped_img, ref_img, bbox)
        self.assertEqual(undone.shape, (5, 5, 5))

if __name__ == "__main__":
    unittest.main()
