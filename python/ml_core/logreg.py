import numpy as np
from ._lib import ml_logreg_train


def logreg_fit(X: np.ndarray, y: np.ndarray, num_iters: int, lr: float, w_init: np.ndarray | None = None,) -> tuple[float, np.ndarray, float]:
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