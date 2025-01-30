import unittest
import os
import json
import yaml
import shutil
import sys
from tempfile import TemporaryDirectory
from optimed.wrappers.operations import *

class TestFileOperations(unittest.TestCase):

    def setUp(self):
        self.test_dir = TemporaryDirectory()
        self.test_file = os.path.join(self.test_dir.name, "test_file.txt")
        self.test_json_file = os.path.join(self.test_dir.name, "test_file.json")
        self.test_yaml_file = os.path.join(self.test_dir.name, "test_file.yaml")
        self.test_content = "Hello, World!"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_maybe_mkdir(self):
        dir_path = os.path.join(self.test_dir.name, "new_dir")
        maybe_mkdir(dir_path)
        self.assertTrue(os.path.isdir(dir_path))

    def test_maybe_remove_dir(self):
        dir_path = os.path.join(self.test_dir.name, "new_dir")
        maybe_mkdir(dir_path)
        maybe_remove_dir(dir_path)
        self.assertTrue(~os.path.exists(dir_path))

    def test_maybe_remove_file(self):
        write_to_file(self.test_content, self.test_file)
        self.assertTrue(os.path.isfile(self.test_file))
        maybe_remove_file(self.test_file)
        self.assertFalse(os.path.isfile(self.test_file))

    def test_copy_file(self):
        write_to_file(self.test_content, self.test_file)
        dest_file = os.path.join(self.test_dir.name, "copied_file.txt")
        copy_file(self.test_file, dest_file)
        self.assertTrue(os.path.isfile(dest_file))
        self.assertEqual(read_file(dest_file), self.test_content)

    def test_move_file(self):
        write_to_file(self.test_content, self.test_file)
        dest_file = os.path.join(self.test_dir.name, "moved_file.txt")
        move_file(self.test_file, dest_file)
        self.assertTrue(os.path.isfile(dest_file))
        self.assertFalse(os.path.isfile(self.test_file))

    def test_rename_file(self):
        write_to_file(self.test_content, self.test_file)
        new_file = os.path.join(self.test_dir.name, "renamed_file.txt")
        rename_file(self.test_file, new_file)
        self.assertTrue(os.path.isfile(new_file))
        self.assertFalse(os.path.isfile(self.test_file))

    def test_get_file_size(self):
        write_to_file(self.test_content, self.test_file)
        self.assertEqual(get_file_size(self.test_file), len(self.test_content))

    def test_if_file_exists(self):
        write_to_file(self.test_content, self.test_file)
        self.assertTrue(if_file_exists(self.test_file))
        os.remove(self.test_file)
        self.assertFalse(if_file_exists(self.test_file))

    def test_if_directory_exists(self):
        self.assertTrue(if_directory_exists(self.test_dir.name))
        self.assertFalse(if_directory_exists("fake_directory"))

    def test_read_file(self):
        write_to_file(self.test_content, self.test_file)
        self.assertEqual(read_file(self.test_file), self.test_content)

    def test_write_to_file(self):
        write_to_file(self.test_content, self.test_file)
        self.assertTrue(os.path.isfile(self.test_file))
        self.assertEqual(read_file(self.test_file), self.test_content)

    def test_append_to_file(self):
        write_to_file(self.test_content, self.test_file)
        append_to_file(self.test_content, self.test_file)
        expected_content = self.test_content + self.test_content
        self.assertEqual(read_file(self.test_file), expected_content)

    def test_get_file_extension(self):
        self.assertEqual(get_file_extension(self.test_file), ".txt")

    def test_get_absolute_path(self):
        relative_path = "test.txt"
        absolute_path = get_absolute_path(relative_path)
        self.assertTrue(os.path.isabs(absolute_path))

    def test_count_files(self):
        write_to_file(self.test_content, self.test_file)
        write_to_file(self.test_content, os.path.join(self.test_dir.name, "file2.txt"))
        self.assertEqual(count_files(self.test_dir.name), 2)

    def test_list_dir(self):
        write_to_file(self.test_content, self.test_file)
        write_to_file(self.test_content, os.path.join(self.test_dir.name, "file2.txt"))
        directory_contents = list_dir(self.test_dir.name)
        self.assertIn("test_file.txt", directory_contents)
        self.assertIn("file2.txt", directory_contents)

    def test_list_subdirs(self):
        subdir1 = os.path.join(self.test_dir.name, "subdir1")
        subdir2 = os.path.join(self.test_dir.name, "subdir2")
        maybe_mkdir(subdir1)
        maybe_mkdir(subdir2)
        subdirs = list_subdirs(self.test_dir.name)
        self.assertIn("subdir1", subdirs)
        self.assertIn("subdir2", subdirs)

    def test_load_json(self):
        data = {"key": "value"}
        save_json(data, self.test_json_file)
        loaded_data = load_json(self.test_json_file)
        self.assertEqual(loaded_data, data)

    def test_save_json(self):
        data = {"key": "value"}
        save_json(data, self.test_json_file)
        self.assertTrue(os.path.isfile(self.test_json_file))
        with open(self.test_json_file, 'r') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, data)

    def test_load_yaml(self):
        data = {"key": "value"}
        save_yaml(data, self.test_yaml_file)
        loaded_data = load_yaml(self.test_yaml_file)
        self.assertEqual(loaded_data, data)

    def test_save_yaml(self):
        data = {"key": "value"}
        save_yaml(data, self.test_yaml_file)
        self.assertTrue(os.path.isfile(self.test_yaml_file))
        with open(self.test_yaml_file, 'r') as f:
            loaded_data = yaml.safe_load(f)
        self.assertEqual(loaded_data, data)

if __name__ == "__main__":
    unittest.main()
