# NRRD Utilities

## Overview
This module provides convenient functions to work with NRRD files, including loading, saving, and processing segmentation data. It utilizes the `nrrd` and `slicerio` libraries, making it easy to integrate into various image processing pipelines.

## Functions

### 1. `load_nrrd(file: str, data_only: bool=False, header_only: bool=False) -> Tuple[np.ndarray, dict]`
Loads an NRRD file from the specified path.  
- **Returns**: A NumPy array or a tuple (array, header) depending on the parameters.

---

### 2. `save_nrrd(ndarray: np.ndarray, header: dict=None, compression_level: int=9, save_to: str='image.nrrd') -> None`
Saves a data array in NRRD format with the specified parameters.  
- **Returns**: None

---

### 3. `save_multilabel_nrrd_seg(ndarray: np.ndarray, segment_metadata: list=None, save_to: str='segmentation.seg.nrrd') -> None`
Saves a multi-label segmentation in NRRD format with optional metadata for each segment.  
- **Returns**: None

---

## Example Usage

```python
import numpy as np
from optimed.wrappers.nrrd import load_nrrd, save_nrrd

data, header = load_nrrd('input_image.nrrd')
print(data.shape)

save_nrrd(data, header, compression_level=5, save_to='output_image.nrrd')
```

## License
This project is licensed under the Apache License 2.0.

## Contributions
Contributions are welcome! Please submit pull requests or open issues.
