import numpy as np
from ._lib import _ml
from ctypes import c_double, c_int, POINTER


# C function bindings
_ml.ml_sigmoid.argtypes = (
    POINTER(c_double),  # x
    POINTER(c_double),  # out
    c_int,              # n
)
_ml.ml_sigmoid.restype = None

_ml.ml_logreg_grad_w.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # p_pred
    POINTER(c_double),  # grad_w
    c_int,              # n_rows
    c_int,              # n_cols
)
_ml.ml_logreg_grad_w.restype = None

_ml.ml_logreg_sgd_step.argtypes = (
    POINTER(c_double),  # X
    POINTER(c_double),  # y_true
    POINTER(c_double),  # w (in-place)
    c_int,              # n_rows
    c_int,              # n_cols
    c_double,           # lr
)
_ml.ml_logreg_sgd_step.restype = c_double

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


# Python wrapper functions
def ml_sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid elementwise: out = 1 / (1 + exp(-x))."""
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    n = x.size

    out = np.empty(n, dtype=np.float64)

    x_ptr = x.ctypes.data_as(POINTER(c_double))
    out_ptr = out.ctypes.data_as(POINTER(c_double))

    _ml.ml_sigmoid(x_ptr, out_ptr, n)

    return out


def ml_logreg_grad_w(X: np.ndarray, y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    """Compute gradient of logistic loss w.r.t. weights."""
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


def ml_logreg_sgd_step(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, lr: float) -> float:
    """One batch SGD step for logistic regression; updates w in-place, returns loss."""
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


def ml_logreg_train(X: np.ndarray, y_true: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Batch GD for logistic regression. Returns (final_w, final_loss)."""
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


def logreg_fit(X: np.ndarray, y: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None) -> tuple[float, np.ndarray, float]:
    """
    Train logistic regression with bias using batch GD.
    Returns w_ext where w_ext[0] is bias, w_ext[1:] are weights.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Ensure 2D X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    n_rows, n_cols = X.shape

    # Build extended design matrix with bias column of ones
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

    # Call C training loop
    w_out, final_loss = ml_logreg_train(
        X_ext,
        y,
        num_iters=num_iters,
        lr=lr,
        w_init=w_init,
    )

    return w_out[0], w_out[1:], final_loss