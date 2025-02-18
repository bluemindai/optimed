import importlib.util

_cupy_available = importlib.util.find_spec("cupy") is not None
_cucim_available = importlib.util.find_spec("cucim") is not None