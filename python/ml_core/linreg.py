import numpy as np
from ctypes import c_double, c_int, POINTER
from ._lib import _ml


# C function bindings
_ml.ml_mse.argtypes = (POINTER(c_double), POINTER(c_double), c_int)
_ml.ml_mse.restype = c_double

_ml.ml_linreg_mse_grad_w.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # y_pred
    POINTER(c_double),  # grad_w
    c_int,              # n_rows
    c_int,              # n_cols
)
_ml.ml_linreg_mse_grad_w.restype = None

_ml.ml_linreg_sgd_step.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # w (in-place)
    c_int,              # n_rows
    c_int,              # n_cols
    c_double,           # lr
)
_ml.ml_linreg_sgd_step.restype = c_double

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


# Python wrapper functions
def ml_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between two 1D float64 arrays."""
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


def ml_linreg_mse_grad_w(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute grad_w of MSE for linear regression: (2/n) X^T (y_pred - y_true)."""
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


def ml_linreg_sgd_step(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, lr: float) -> float:
    """One SGD step for linear regression, updates w in-place, returns loss."""
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


def ml_linreg_train(X: np.ndarray, y_true: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Run batch gradient descent for num_iters, return (final_w, final_loss)."""
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


def linreg_fit(X: np.ndarray, y: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """
    Train linear regression with bias using batch GD.
    Returns w_ext where w_ext[0] is bias, w_ext[1:] are weights.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    n_rows, n_cols = X.shape

    # Build extended design matrix with bias column
    X_ext = np.empty((n_rows, n_cols + 1), dtype=np.float64)
    X_ext[:, 0] = 1.0
    X_ext[:, 1:] = X

    # Initial weights
    if w_init is not None:
        w_init = np.asarray(w_init, dtype=np.float64).reshape(-1)
        if w_init.shape[0] != n_cols + 1:
            raise ValueError("w_init length must be n_features + 1 (for bias)")
    else:
        w_init = None

    w_out, final_loss = ml_linreg_train(
        X_ext,
        y,
        num_iters=num_iters,
        lr=lr,
        w_init=w_init,
    )

    return w_out, final_loss