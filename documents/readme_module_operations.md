# Operations Utilities

## Overview
This module provides a collection of convenient utility functions for file and directory operations. In addition to basic tasks such as copying, moving, and renaming files, it offers enhanced support for JSON/YAML handling and robust functions for managing file system paths and directories. These utilities aim to simplify common file system operations, making your code more intuitive and reliable.

## Functions

### 1. `maybe_mkdir(directory: str) -> None`
Creates the specified directory if it does not already exist. It also creates any necessary parent directories.
- **Returns**: None.

### 2. `maybe_remove_dir(directory: str) -> None`
Removes the specified directory if it exists. If the file does not exist, no action is taken.
- **Returns**: None.

### 3. `maybe_remove_file(file_path: str) -> None`
Removes the specified file if it exists.
- **Returns**: None.

### 4. `copy_file(src: str, dest: str) -> None`
Copies a file from the source path to the destination path.
- **Returns**: None.

### 5. `move_file(src: str, dest: str) -> None`
Moves a file from the source path to the destination path.
- **Returns**: None.

### 6. `rename_file(src: str, new_name: str) -> None`
Renames the specified file to the new name.
- **Returns**: None.

### 7. `get_file_size(file_path: str) -> int`
Returns the size of the specified file in bytes.
- **Returns**: Size of the file in bytes.

### 8. `if_file_exists(file_path: str) -> bool`
Checks whether the specified file exists.
- **Returns**: True if the file exists, False otherwise.

### 9. `if_directory_exists(directory: str) -> bool`
Checks whether the specified directory exists.
- **Returns**: True if the directory exists, False otherwise.

### 10. `read_file(file_path: str) -> str`
Reads the contents of the specified file and returns it as a string.
- **Returns**: Contents of the file.

### 11. `write_to_file(content: str, file_path: str) -> None`
Writes the specified content to a file. Overwrites the file if it already exists.
- **Returns**: None.

### 12. `append_to_file(content: str, file_path: str) -> None`
Appends the specified content to a file. Creates the file if it does not exist.
- **Returns**: None.

### 13. `get_file_extension(file_path: str) -> str`
Returns the file extension of the specified file.
- **Returns**: The file extension, including the dot (e.g., '.txt').

### 14. `get_absolute_path(path: str) -> str`
Returns the absolute path of the specified file or directory.
- **Returns**: The absolute path.

### 15. `count_files(directory: str) -> int`
Counts the number of files in the specified directory.
- **Returns**: The number of files in the directory.

### 16. `list_dir(directory: str) -> list`
Lists the contents of the specified directory.
- **Returns**: A list of the names of the entries in the directory.

### 17. `list_subdirs(directory: str) -> list`
Lists all subdirectories within the specified directory.
- **Returns**: A list of the names of the subdirectories.

### 18. `load_json(file: str)`
Loads and returns the contents of a JSON file as a Python object.
- **Returns**: The parsed contents of the JSON file.

### 19. `save_json(obj, file: str, indent: int = 4, ensure_ascii: bool = False, cls=None) -> None`
Saves a Python object to a JSON file with optional custom formatting and encoding.
- **Returns**: None.

### 20. `load_yaml(file: str)`
Loads and returns the contents of a YAML file as a Python object.
- **Returns**: The parsed contents of the YAML file.

### 21. `save_yaml(obj, file: str, default_flow_style: bool = False, indent: int = 2) -> None`
Saves a Python object to a YAML file.
- **Returns**: None.

### 22. `recursively_find_file(filename: str, start_path: str = ".") -> List[str]`
Recursively searches for files named `filename` starting from `start_path`.
- **Returns**: A list of matching file paths.

### 23. `recursive_find_python_class(folder: str, class_name: str, current_module: str) -> object`
Recursively searches for a Python class within modules in the specified folder.
- **Returns**: The class object if found, otherwise None.

## Usage Examples

```python
from optimed.wrappers.operations import *

# Create a directory if it doesn't exist
maybe_mkdir(join('path/to/new/dir'))

# Copy a file
copy_file("source.txt", "destination.txt")

# Load a JSON file
data = load_json("data.json")
print(data)

# Save JSON
save_json({"name": "John", "age": 30}, "output.json")
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please feel free to submit pull requests or raise issues.