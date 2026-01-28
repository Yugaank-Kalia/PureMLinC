import ctypes
import pathlib
import numpy as np
from ctypes import c_double, c_int, POINTER


_here = pathlib.Path(__file__).resolve()
_lib_path = (_here.parent.parent.parent / "csrc" / "build" / "libml.dylib").resolve()
_ml = ctypes.CDLL(str(_lib_path))

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

_ml.ml_mse.argtypes = (POINTER(c_double), POINTER(c_double), c_int)
_ml.ml_mse.restype = c_double

def ml_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean squared error between two 1D float64 arrays.
    """
    y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have same shape")
    if y_pred.ndim != 1:
        raise ValueError("y_pred and y_true must be 1D")

    n = y_pred.size
    yp_ptr = y_pred.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))

    return float(_ml.ml_mse(yp_ptr, yt_ptr, n))

_ml.ml_linreg_mse_grad_w.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # y_pred
    POINTER(c_double),  # grad_w
    c_int,              # n_rows
    c_int,              # n_cols
)
_ml.ml_linreg_mse_grad_w.restype = None


def ml_linreg_mse_grad_w(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute grad_w of MSE for linear regression: (2/n) X^T (y_pred - y_true).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64)
    y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    n_rows, n_cols = X.shape
    if y_true.shape[0] != n_rows or y_pred.shape[0] != n_rows:
        raise ValueError("y_true, y_pred length must equal X.shape[0]")

    grad_w = np.empty(n_cols, dtype=np.float64)

    X_ptr = X.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))
    yp_ptr = y_pred.ctypes.data_as(POINTER(c_double))
    gw_ptr = grad_w.ctypes.data_as(POINTER(c_double))

    _ml.ml_linreg_mse_grad_w(X_ptr, yt_ptr, yp_ptr, gw_ptr, n_rows, n_cols)
    return grad_w

_ml.ml_linreg_sgd_step.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # w (in-place)
    c_int,              # n_rows
    c_int,              # n_cols
    c_double,           # lr
)
_ml.ml_linreg_sgd_step.restype = c_double


def ml_linreg_sgd_step(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, lr: float) -> float:
    """
    One SGD step for linear regression, updates w in-place, returns loss.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if w.ndim != 1:
        raise ValueError("w must be 1D")

    n_rows, n_cols = X.shape
    if y_true.shape[0] != n_rows:
        raise ValueError("y_true length must equal X.shape[0]")
    if w.shape[0] != n_cols:
        raise ValueError("w length must equal X.shape[1]")

    X_ptr = X.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))
    w_ptr = w.ctypes.data_as(POINTER(c_double))

    loss = _ml.ml_linreg_sgd_step(X_ptr, yt_ptr, w_ptr, n_rows, n_cols, lr)
    return float(loss)

_ml.ml_linreg_train.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # w_init (nullable)
    POINTER(c_double),  # w_out
    c_int,              # n_rows
    c_int,              # n_cols
    c_int,              # num_iters
    c_double,           # lr
)
_ml.ml_linreg_train.restype = c_double

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(X.mean(axis=0), dtype=np.float64)
    std = np.asarray(X.std(axis=0), dtype=np.float64)
    std = np.where(std == 0.0, 1.0, std)
    normalized = (X - mean) / std
    return normalized, mean, std

def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def ml_linreg_train(X: np.ndarray, y_true: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """
    Run batch gradient descent for num_iters, return (final_w, final_loss).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")

    n_rows, n_cols = X.shape
    if y_true.shape[0] != n_rows:
        raise ValueError("y_true length must equal X.shape[0]")

    if w_init is not None:
        w_init = np.ascontiguousarray(w_init, dtype=np.float64).reshape(-1)
        if w_init.shape[0] != n_cols:
            raise ValueError("w_init length must equal X.shape[1]")
    else:
        w_init = None

    w_out = np.empty(n_cols, dtype=np.float64)

    X_ptr = X.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))
    w_init_ptr = (
        w_init.ctypes.data_as(POINTER(c_double)) if w_init is not None else None
    )
    w_out_ptr = w_out.ctypes.data_as(POINTER(c_double))

    final_loss = _ml.ml_linreg_train(
        X_ptr,
        yt_ptr,
        w_init_ptr,
        w_out_ptr,
        n_rows,
        n_cols,
        int(num_iters),
        float(lr),
    )
    return w_out, float(final_loss)

_ml.ml_logreg_grad_w.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # p_pred
    POINTER(c_double),  # grad_w
    c_int,              # n_rows
    c_int,              # n_cols
)
_ml.ml_logreg_grad_w.restype = None


def ml_logreg_grad_w(X: np.ndarray, y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64).reshape(-1)
    p_pred = np.ascontiguousarray(p_pred, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n_rows, n_cols = X.shape
    if y_true.shape[0] != n_rows or p_pred.shape[0] != n_rows:
        raise ValueError("y_true, p_pred length must equal X.shape[0]")

    grad_w = np.empty(n_cols, dtype=np.float64)

    X_ptr = X.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))
    pp_ptr = p_pred.ctypes.data_as(POINTER(c_double))
    gw_ptr = grad_w.ctypes.data_as(POINTER(c_double))

    _ml.ml_logreg_grad_w(X_ptr, yt_ptr, pp_ptr, gw_ptr, n_rows, n_cols)
    return grad_w

_ml.ml_logreg_sgd_step.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # w (in-place)
    c_int,              # n_rows
    c_int,              # n_cols
    c_double,           # lr
)
_ml.ml_logreg_sgd_step.restype = c_double


def ml_logreg_sgd_step(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, lr: float) -> float:
    """
    One batch SGD step for logistic regression; updates w in-place, returns loss.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64).reshape(-1)
    w = np.ascontiguousarray(w, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n_rows, n_cols = X.shape
    if y_true.shape[0] != n_rows:
        raise ValueError("y_true length must equal X.shape[0]")
    if w.shape[0] != n_cols:
        raise ValueError("w length must equal X.shape[1]")

    X_ptr = X.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))
    w_ptr = w.ctypes.data_as(POINTER(c_double))

    loss = _ml.ml_logreg_sgd_step(X_ptr, yt_ptr, w_ptr, n_rows, n_cols, float(lr))
    return float(loss)

_ml.ml_logreg_train.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # w_init (nullable)
    POINTER(c_double),  # w_out
    c_int,              # n_rows
    c_int,              # n_cols
    c_int,              # num_iters
    c_double,           # lr
)
_ml.ml_logreg_train.restype = c_double


def ml_logreg_train(X: np.ndarray, y_true: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """
    Batch GD for logistic regression. Returns (final_w, final_loss).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y_true = np.ascontiguousarray(y_true, dtype=np.float64).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D")
    n_rows, n_cols = X.shape
    if y_true.shape[0] != n_rows:
        raise ValueError("y_true length must equal X.shape[0]")

    if w_init is not None:
        w_init = np.ascontiguousarray(w_init, dtype=np.float64).reshape(-1)
        if w_init.shape[0] != n_cols:
            raise ValueError("w_init length must equal X.shape[1]")
        w_init_ptr = w_init.ctypes.data_as(POINTER(c_double))
    else:
        w_init_ptr = None

    w_out = np.empty(n_cols, dtype=np.float64)
    X_ptr = X.ctypes.data_as(POINTER(c_double))
    yt_ptr = y_true.ctypes.data_as(POINTER(c_double))
    w_out_ptr = w_out.ctypes.data_as(POINTER(c_double))

    final_loss = _ml.ml_logreg_train(
        X_ptr,
        yt_ptr,
        w_init_ptr,
        w_out_ptr,
        n_rows,
        n_cols,
        int(num_iters),
        float(lr),
    )
    return w_out, float(final_loss)

_ml.ml_sigmoid.argtypes = (
    POINTER(c_double),  # x
    POINTER(c_double),  # out
    c_int,              # n
)
_ml.ml_sigmoid.restype = None


def ml_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid elementwise: out = 1 / (1 + exp(-x)).
    x: 1D float64 array.
    Returns a new 1D float64 array.
    """
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    n = x.size

    out = np.empty(n, dtype=np.float64)

    x_ptr = x.ctypes.data_as(POINTER(c_double))
    out_ptr = out.ctypes.data_as(POINTER(c_double))

    _ml.ml_sigmoid(x_ptr, out_ptr, n)

    return out