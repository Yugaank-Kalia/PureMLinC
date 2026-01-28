"""Core C library bindings and shared utilities."""
import ctypes
import pathlib
import numpy as np
import sys
from ctypes import c_double, c_int, POINTER


_here = pathlib.Path(__file__).resolve()

# Try multiple locations for the library:
# 1. First, check in the package directory (installed case)
# 2. Then, check in the development directory
_lib_locations = [
    _here.parent / "libml.dylib",  # Installed in same package
    _here.parent / "libml.so",      # Linux installed
    _here.parent / "libml.dll",     # Windows installed
    (_here.parent.parent.parent / "csrc" / "build" / "libml.dylib"),  # Development
]

_ml = None
_lib_path = None

for loc in _lib_locations:
    if loc.exists():
        _lib_path = loc.resolve()
        try:
            _ml = ctypes.CDLL(str(_lib_path))
            break
        except (OSError, TypeError):
            continue

if _ml is None:
    raise RuntimeError(
        f"Could not load C library. Searched: {[str(p) for p in _lib_locations]}\n"
        "Make sure to build the C library: cd csrc && make"
    )

# Core functions

_ml.ml_add_double.argtypes = (c_double, c_double)
_ml.ml_add_double.restype = c_double

def ml_add_double(a: float, b: float) -> float:
    return float(_ml.ml_add_double(a, b))

_ml.ml_dot.argtypes = (POINTER(c_double), POINTER(c_double), c_int)
_ml.ml_dot.restype = c_double


def ml_dot(x: np.ndarray, y: np.ndarray) -> float:
    """
    Dot product of two 1D float64 arrays using C.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    n = x.size

    x_ptr = x.ctypes.data_as(POINTER(c_double))
    y_ptr = y.ctypes.data_as(POINTER(c_double))

    return float(_ml.ml_dot(x_ptr, y_ptr, n))

_ml.ml_axpy.argtypes = (c_int, c_double, POINTER(c_double), POINTER(c_double))
_ml.ml_axpy.restype = None

def ml_axpy(a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    In-place: y <- a * x + y  (like BLAS axpy).
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    n = x.size

    x_ptr = x.ctypes.data_as(POINTER(c_double))
    y_ptr = y.ctypes.data_as(POINTER(c_double))

    _ml.ml_axpy(n, a, x_ptr, y_ptr)
    return y

_ml.ml_matvec.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # w
    POINTER(c_double),  # y
    c_int,              # n_rows
    c_int,              # n_cols
)
_ml.ml_matvec.restype = None

def ml_matvec(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    y = X @ w  where X is (n_rows, n_cols), w is (n_cols,)
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if w.ndim != 1:
        raise ValueError("w must be 1D")
    n_rows, n_cols = X.shape
    if w.shape[0] != n_cols:
        raise ValueError("w length must equal X.shape[1]")

    y = np.empty(n_rows, dtype=np.float64)

    X_ptr = X.ctypes.data_as(POINTER(c_double))
    w_ptr = w.ctypes.data_as(POINTER(c_double))
    y_ptr = y.ctypes.data_as(POINTER(c_double))

    _ml.ml_matvec(X_ptr, w_ptr, y_ptr, n_rows, n_cols)
    return y

# Utility functions

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance."""
    mean = np.asarray(X.mean(axis=0), dtype=np.float64)
    std = np.asarray(X.std(axis=0), dtype=np.float64)
    std = np.where(std == 0.0, 1.0, std)
    normalized = (X - mean) / std
    return normalized, mean, std

def train_test_split(X, y, test_ratio=0.2, seed=0):
    """Split data into train and test sets."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Export C library handle for use in other modules
__all__ = ['_ml', 'ml_add_double', 'ml_dot', 'ml_axpy', 'ml_matvec', 'standardize', 'train_test_split']
