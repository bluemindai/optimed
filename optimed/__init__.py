__version__ = "1.0.0a4"


def _check_availability(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


_cupy_available = _check_availability("cupy")
_cucim_available = _check_availability("cucim")
_build_with_gpu = _cupy_available and _cucim_available
