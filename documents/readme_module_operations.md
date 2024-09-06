# Operations Utilities

This module provides a collection of convenient utility functions for working with files and directories. It simplifies common tasks such as file manipulation, directory management, JSON handling, and working with file system paths. The goal of this library is to make file and directory operations more intuitive, concise, and reliable.

## Functions

### 1. `maybe_mkdir(directory: str) -> None`
Creates the specified directory if it does not already exist. It also creates any necessary parent directories.

- **Parameters**:
  - `directory` (str): The path of the directory to create.
- **Returns**: None.

### 2. `maybe_remove_file(file_path: str) -> None`
Removes the specified file if it exists.

- **Parameters**:
  - `file_path` (str): The path of the file to remove.
- **Returns**: None.

### 3. `copy_file(src: str, dest: str) -> None`
Copies a file from the source path to the destination path.

- **Parameters**:
  - `src` (str): The path of the file to copy.
  - `dest` (str): The path where the file should be copied.
- **Returns**: None.

### 4. `move_file(src: str, dest: str) -> None`
Moves a file from the source path to the destination path.

- **Parameters**:
  - `src` (str): The path of the file to move.
  - `dest` (str): The path where the file should be moved.
- **Returns**: None.

### 5. `rename_file(src: str, new_name: str) -> None`
Renames the specified file to the new name.

- **Parameters**:
  - `src` (str): The current path of the file.
  - `new_name` (str): The new name for the file.
- **Returns**: None.

### 6. `get_file_size(file_path: str) -> int`
Returns the size of the specified file in bytes.

- **Parameters**:
  - `file_path` (str): The path of the file.
- **Returns**: Size of the file in bytes.

### 7. `if_file_exists(file_path: str) -> bool`
Checks whether the specified file exists.

- **Parameters**:
  - `file_path` (str): The path of the file to check.
- **Returns**: True if the file exists, False otherwise.

### 8. `if_directory_exists(directory: str) -> bool`
Checks whether the specified directory exists.

- **Parameters**:
  - `directory` (str): The path of the directory to check.
- **Returns**: True if the directory exists, False otherwise.

### 9. `read_file(file_path: str) -> str`
Reads the contents of the specified file and returns it as a string.

- **Parameters**:
  - `file_path` (str): The path of the file to be read.
- **Returns**: Contents of the file.

### 10. `write_to_file(content: str, file_path: str) -> None`
Writes the specified content to a file. Overwrites the file if it already exists.

- **Parameters**:
  - `content` (str): The content to write to the file.
  - `file_path` (str): The path of the file.
- **Returns**: None.

### 11. `append_to_file(content: str, file_path: str) -> None`
Appends the specified content to a file. Creates the file if it does not exist.

- **Parameters**:
  - `content` (str): The content to append to the file.
  - `file_path` (str): The path of the file.
- **Returns**: None.

### 12. `get_file_extension(file_path: str) -> str`
Returns the file extension of the specified file.

- **Parameters**:
  - `file_path` (str): The path of the file.
- **Returns**: The file extension, including the dot (e.g., '.txt').

### 13. `get_absolute_path(path: str) -> str`
Returns the absolute path of the specified file or directory.

- **Parameters**:
  - `path` (str): The relative or absolute path.
- **Returns**: The absolute path.

### 14. `count_files(directory: str) -> int`
Counts the number of files in the specified directory.

- **Parameters**:
  - `directory` (str): The path of the directory to count files in.
- **Returns**: The number of files in the directory.

### 15. `list_dir(directory: str) -> list`
Lists the contents of the specified directory.

- **Parameters**:
  - `directory` (str): The path of the directory.
- **Returns**: A list of the names of the entries in the directory.

### 16. `list_subdirs(directory: str) -> list`
Lists all subdirectories within the specified directory.

- **Parameters**:
  - `directory` (str): The path of the directory.
- **Returns**: A list of the names of the subdirectories.

### 17. `load_json(file: str)`
Loads and returns the contents of a JSON file as a Python object.

- **Parameters**:
  - `file` (str): The path of the JSON file to load.
- **Returns**: The parsed contents of the JSON file.

### 18. `save_json(obj, file: str, indent: int = 4, ensure_ascii: bool = False, cls=None) -> None`
Saves a Python object to a JSON file with optional custom formatting and encoding.

- **Parameters**:
  - `obj`: The Python object to serialize and save.
  - `file` (str): The path to the JSON file.
  - `indent` (int, optional): Number of spaces to use for indentation.
  - `ensure_ascii` (bool, optional): Whether to escape non-ASCII characters.
  - `cls` (optional): A custom encoder class for non-serializable objects.
- **Returns**: None.

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