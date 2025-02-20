# Optimed - CV Medical Tool
[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](LICENSE)
[![PyPI version](https://badge.fury.io/py/optimed.svg)](https://pypi.org/project/optimed/)
[![Downloads](https://pepy.tech/badge/optimed)](https://pepy.tech/project/optimed)

## About the Project  
Optimed is a tool for working with 2D/3D medical images.  
It simplifies the tasks of loading, transforming, analyzing, and visualizing data for medical research and clinical practice.

Read more about available modules [here](documents/readme_modules.md)

## Installation

Optimed is available via [PyPI](https://pypi.org/project/optimed/), which makes installation quick and easy. Follow the steps below to install the package for your desired setup.

### 1. Standard Installation (CPU Only)

For most users who do not require GPU acceleration, install the core package with:

```bash
pip install optimed
```

### 2. GPU-Accelerated Installation

To leverage GPU capabilities for faster processing, install Optimed with GPU support:

```bash
pip install optimed[gpu]
```

> **Note:** Ensure you have a compatible GPU and have installed the necessary drivers and libraries (such as CUDA) to enable GPU acceleration.

# Contributing
Optimed welcomes your contributions!<br>
Thanks so much to all of our amazing contributors!

<a href="https://github.com/bluemindai/optimed/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bluemindai/optimed&r="  width="100px"/>
</a>

## Contributor License Agreement
This project welcomes contributions and suggestions. Most contributions require you to
agree to a [Contributor License Agreement](CONTRIBUTING.md) (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. 

# License

This project is licensed under the Apache 2.0 License.

By the way, if by chance you ever run into me on the street, in a café, hotel, airport, or bar, I’d be happy if you bought me a beer!

# Citation
If you use ```optimed``` in your research, please cite it as follows:
```
@misc{optimed2025,
    title={optimed - CV Medical Library}, 
    author={BlueMind AI Inc. and Roman Fitzjalen, Gleb Sakhnov, Georgy Nanyan},
    year={2025},
    note={Version 1.0.0}
}
```
## Contact
If you have any questions, please raise an issue or contact us at [info@bluemind.co](info@bluemind.co).
