"""Machine Learning in C - Python bindings."""

__version__ = "0.2.0"
__author__ = "Yugaank"

from python.ml_core._lib import (
    ml_add_double,
    ml_dot,
    ml_axpy,
    ml_matvec,
    standardize,
    train_test_split,
)

from python.ml_core.linreg import (
    linreg_fit,
    ml_mse,
    ml_linreg_mse_grad_w,
    ml_linreg_sgd_step,
    ml_linreg_train,
)

from python.ml_core.logreg import (
    logreg_fit,
    ml_sigmoid,
    ml_logreg_grad_w,
    ml_logreg_sgd_step,
    ml_logreg_train,
)

__all__ = [
    # Core
    "ml_add_double",
    "ml_dot",
    "ml_axpy",
    "ml_matvec",
    "standardize",
    "train_test_split",

    # Linear regression
    "linreg_fit",
    "ml_mse",
    "ml_linreg_mse_grad_w",
    "ml_linreg_sgd_step",
    "ml_linreg_train",

    # Logistic regression
    "logreg_fit",
    "ml_sigmoid",
    "ml_logreg_grad_w",
    "ml_logreg_sgd_step",
    "ml_logreg_train",
]
