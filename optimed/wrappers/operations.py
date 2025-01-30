import json
import yaml
import shutil
import os
import importlib
import pkgutil
from typing import List

join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
exists = os.path.exists
listdir = os.listdir


def maybe_mkdir(directory: str) -> None:
    """
    Creates the specified directory if it does not already exist. If the directory 
    or any of its parent directories do not exist, they will be created.

    Parameters:
        directory (str): The path of the directory to create.

    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)


def maybe_remove_dir(directory: str) -> None:
    """
    Removes the specified directory if it exists. If the file does not exist, no action is taken.

    Parameters:
        directory (str): The path of the directory to create.

    Returns:
        None
    """
    if exists(directory):
        shutil.rmtree(directory)


def maybe_remove_file(file_path: str) -> None:
    """
    Removes the specified file if it exists. If the file does not exist, no action is taken.

    Parameters:
        file_path (str): The path of the file to be removed.

    Returns:
        None
    """
    if exists(file_path):
        os.remove(file_path)


def copy_file(src: str, dest: str) -> None:
    """
    Copies a file from the source path to the destination path.

    Parameters:
        src (str): The path of the file to copy.
        dest (str): The path where the file should be copied to.

    Returns:
        None
    """
    shutil.copy(src, dest)


def move_file(src: str, dest: str) -> None:
    """
    Moves a file from the source path to the destination path.

    Parameters:
        src (str): The path of the file to move.
        dest (str): The path where the file should be moved to.

    Returns:
        None
    """
    shutil.move(src, dest)


def rename_file(src: str, new_name: str) -> None:
    """
    Renames the specified file to the new name.

    Parameters:
        src (str): The current path of the file.
        new_name (str): The new name for the file.

    Returns:
        None
    """
    os.rename(src, new_name)


def get_file_size(file_path: str) -> int:
    """
    Returns the size of the specified file in bytes.

    Parameters:
        file_path (str): The path of the file to check the size of.

    Returns:
        int: The size of the file in bytes.
    """
    return os.path.getsize(file_path)


def if_file_exists(file_path: str) -> bool:
    """
    Checks whether the specified file exists.

    Parameters:
        file_path (str): The path of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return isfile(file_path)


def if_directory_exists(directory: str) -> bool:
    """
    Checks whether the specified directory exists.

    Parameters:
        directory (str): The path of the directory to check.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return isdir(directory)


def read_file(file_path: str) -> str:
    """
    Reads the contents of the specified file and returns it as a string.

    Parameters:
        file_path (str): The path of the file to be read.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, 'r') as f:
        return f.read()


def write_to_file(content: str, file_path: str) -> None:
    """
    Writes the specified content to a file. If the file already exists, it will be overwritten.

    Parameters:
        content (str): The content to write to the file.
        file_path (str): The path of the file to write to.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        f.write(content)


def append_to_file(content: str, file_path: str) -> None:
    """
    Appends the specified content to a file. If the file does not exist, it will be created.

    Parameters:
        content (str): The content to append to the file.
        file_path (str): The path of the file to append to.

    Returns:
        None
    """
    with open(file_path, 'a') as f:
        f.write(content)


def get_file_extension(file_path: str) -> str:
    """
    Returns the file extension of the specified file.

    Parameters:
        file_path (str): The path of the file to check.

    Returns:
        str: The file extension, including the dot (e.g., '.txt').
    """
    return os.path.splitext(file_path)[1]


def get_absolute_path(path: str) -> str:
    """
    Returns the absolute path of the specified file or directory.

    Parameters:
        path (str): The relative or absolute path to convert.

    Returns:
        str: The absolute path.
    """
    return os.path.abspath(path)


def count_files(directory: str) -> int:
    """
    Counts the number of files in the specified directory.

    Parameters:
        directory (str): The path of the directory to count files in.

    Returns:
        int: The number of files in the directory.
    """
    return len([entry for entry in listdir(directory) if isfile(join(directory, entry))])


def list_dir(directory: str) -> list:
    """
    Lists the contents of the specified directory and returns them as a list.

    Parameters:
        directory (str): The path of the directory to list contents of.

    Returns:
        list: A list of the names of the entries in the directory.
    """
    return listdir(directory)


def list_subdirs(directory: str) -> list:
    """
    Lists all subdirectories within the specified directory.

    Parameters:
        directory (str): The path of the directory to list subdirectories of.

    Returns:
        list: A list of the names of the subdirectories in the directory.
    """
    return [entry for entry in listdir(directory) if isdir(join(directory, entry))]


def load_json(file: str):
    """
    Loads and returns the contents of a JSON file.

    This function reads the specified JSON file, parses its contents, and returns the resulting 
    Python object (usually a dictionary or a list).

    Parameters:
        file (str): The path to the JSON file to be loaded.

    Returns:
        The parsed contents of the JSON file, typically a dictionary or a list.
    """
    with open(file, 'r', encoding='utf8') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, ensure_ascii: bool = False, cls=None) -> None:
    """
    Saves a Python object to a JSON file with optional custom encoding.

    This function serializes the given Python object (usually a dictionary or a list) 
    into a JSON formatted string and writes it to the specified file. If a custom encoder 
    class is provided, it will be used to serialize objects that aren't natively serializable 
    by the default `json` module.

    Parameters:
        obj: The Python object to be serialized and saved to a JSON file.
        file (str): The path to the file where the JSON data will be saved.
        indent (int, optional): The number of spaces to use as indentation in the JSON file. 
                                Defaults to 4 for pretty-printing.
        ensure_ascii (bool, optional): If False, the JSON file will be saved with Unicode characters 
                                       instead of escaping them. Defaults to False.
        cls (type, optional): A custom JSON encoder class to serialize objects that aren't 
                              natively serializable by the default `json` module.

    Returns:
        None
    """
    with open(file, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii, cls=cls)


def load_yaml(file: str):
    """
    Loads and returns the contents of a YAML file.

    This function reads the specified YAML file, parses its contents, and returns the resulting 
    Python object (usually a dictionary or a list).

    Parameters:
        file (str): The path to the YAML file to be loaded.

    Returns:
        The parsed contents of the YAML file, typically a dictionary or a list.
    """
    with open(file, 'r', encoding='utf8') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(obj, file: str, default_flow_style: bool = False, indent: int = 2) -> None:
    """
    Saves a Python object to a YAML file.

    This function serializes the given Python object (usually a dictionary or a list) 
    into a YAML formatted string and writes it to the specified file.

    Parameters:
        obj: The Python object to be serialized and saved to a YAML file.
        file (str): The path to the file where the YAML data will be saved.
        default_flow_style (bool, optional): If True, the YAML will use the block style. 
                                             Defaults to False for a more human-readable format.
        indent (int, optional): The number of spaces to use as indentation in the YAML file. 
                                Defaults to 2.

    Returns:
        None
    """
    with open(file, 'w', encoding='utf8') as f:
        yaml.dump(obj, f, default_flow_style=default_flow_style, indent=indent)


def recursively_find_file(filename: str, start_path: str = ".") -> List[str]:
    """
    Recursively search for files named `filename` starting from `start_path`.

    Parameters
        filename : str
            Name of the file to look for (e.g. "myfile.txt").
        start_path : str, optional
            The directory path in which to begin the search.
            Defaults to the current directory ".".

    Returns
        List[str]
            A list of full paths to files that match the given filename.
            Returns an empty list if no matches are found.
    """
    matches = []
    for root, dirs, files in os.walk(start_path):
        if filename in files:
            full_path = os.path.join(root, filename)
            matches.append(full_path)
    return matches


def recursive_find_python_class(folder: str, class_name: str, current_module: str) -> object:
    """
    Recursively searches for a Python class within a given folder and its subfolders.

    This function searches for a specified class name within Python modules located in the given folder.
    It will recursively traverse subdirectories to find the class.

    Parameters:
        folder (str): The path of the folder to start the search in.
        class_name (str): The name of the class to search for.
        current_module (str): The current module path to use for importing.

    Returns:
        object: The class object if found, otherwise None.
    """
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        if not ispkg:
            m = importlib.import_module(
                current_module + "." + modname
            )
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    join(folder, modname),
                    class_name,
                    current_module=next_current_module
                )
            if tr is not None:
                break
    return tr
