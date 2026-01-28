# Machine Learning in C

Version 0.2 — Linear and Logistic Regression

A high-performance machine learning library written in C with Python bindings. Version 0.2 includes both linear and logistic regression built on top of a small, optimized linear algebra core.

## Features (v0.2)

- Dense linear regression trained with SGD
- Logistic regression for binary classification with sigmoid activation
- Loss functions: Mean Squared Error (MSE) and Binary Cross-Entropy
- Gradient computation and SGD optimization routines
- Core BLAS-style helpers: dot product, AXPY, matrix-vector multiplication
- Data preprocessing utilities (standardization)
- Python bindings for straightforward integration in Python workflows
- Shared library build for native performance
- Performance benchmarks comparing pure Python vs C implementations
- Minimal dependency footprint (only `numpy` for core, `pandas` for examples)

## Project Structure

```
├── csrc/                       # C source code
│   ├── include/
│   │   └── ml.h            # C API header
│   ├── src/
│   │   ├── core.c          # Core linear algebra functions
│   │   ├── linreg.c        # Linear regression implementation
│   │   └── logreg.c        # Logistic regression implementation
│   └── Makefile            # Build configuration
├── python/                    # Python bindings and utilities
│   ├── ml_core/
│   │   ├── __init__.py
│   │   ├── _lib.py           # FFI binding to C library
│   │   ├── linreg.py        # Python API for linear regression
│   │   └── logreg.py        # Python API for logistic regression
│   └── requirements.txt  # Python dependencies
├── examples/                  # Example usage and benchmarks
│   ├── linear_regression.py    # Housing price prediction
│   ├── logistic_regression.py  # Rain prediction (binary classification)
│   └── data/                   # Sample datasets
└── test.py                        # Test suite
```

Key public interfaces live in [csrc/include/ml.h](csrc/include/ml.h) and the Python bindings in [python/ml_core](python/ml_core).

## Building

### Prerequisites

- C compiler (clang/gcc)
- Python 3.x with numpy
- Make

### Build the C Library

```bash
cd csrc
make
```

This generates `build/libml.dylib` on macOS (or `.so` on Linux, `.dll` on Windows).

### Install Python Dependencies

```bash
pip install -r python/requirements.txt
```

### Set Up a Virtual Environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

When finished, deactivate with `deactivate`. The `.venv` folder is ignored in git.

## Usage (v0.2)

### Linear Regression (see [examples/linear_regression.py](examples/linear_regression.py))

```python
import numpy as np
from python.ml_core import linreg_fit, standardize

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
true_w = np.array([1.0, 2.0, -1.5, 0.5, 3.0])
bias = 0.7
y = bias + X @ true_w + 0.1 * rng.normal(size=(200,))

# Standardize for stable training
X_std, X_mean, X_scale = standardize(X)
y_std, y_mean, y_scale = standardize(y)

# Train with bias term (returned as w_ext[0])
w_ext, final_loss = linreg_fit(X_std, y_std, num_iters=3000, lr=0.01)

# Predict back on the original scale
X_ext = np.c_[np.ones(X_std.shape[0]), X_std]
y_pred_std = X_ext @ w_ext
y_pred = y_mean + y_scale * y_pred_std
```

`linreg_fit` returns `(w_ext, final_loss)` where `w_ext` is an array containing the bias term at index 0 followed by feature weights, and `final_loss` is the mean squared error.

### Logistic Regression (see [examples/logistic_regression.py](examples/logistic_regression.py))

```python
import numpy as np
from python.ml_core import logreg_fit, standardize

# Binary classification data
X = np.random.randn(500, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)  # Binary labels

# Standardize features
X_std, X_mean, X_scale = standardize(X)

# Train logistic regression
bias, weights, final_loss = logreg_fit(X_std, y, num_iters=2000, lr=0.1)

# Make predictions
logits = bias + X_std @ weights
probs = 1.0 / (1.0 + np.exp(-logits))
y_pred = (probs >= 0.5).astype(int)
```

`logreg_fit` returns `(bias, weights, final_loss)` where `bias` is a scalar, `weights` is the feature weight vector, and `final_loss` is the binary cross-entropy loss.

### Core C API (v0.2)

Available symbols (see [csrc/include/ml.h](csrc/include/ml.h)):

**Core Linear Algebra:**
- `ml_add_double` — Add two doubles (test function)
- `ml_dot` — Dot product of two vectors
- `ml_axpy` — Vector operation: Y = a*X + Y
- `ml_matvec` — Matrix-vector multiplication
- `ml_sigmoid` — Sigmoid activation function

**Linear Regression:**
- `ml_mse` — Mean squared error loss
- `ml_linreg_mse_grad_w` — Gradient of MSE w.r.t. weights
- `ml_linreg_sgd_step` — Single SGD step for linear regression
- `ml_linreg_train` — Full linear regression training loop

**Logistic Regression:**
- `ml_logistic_loss` — Binary cross-entropy loss
- `ml_logreg_grad_w` — Gradient of logistic loss w.r.t. weights
- `ml_logreg_sgd_step` — Single SGD step for logistic regression
- `ml_logreg_train` — Full logistic regression training loop

## Examples

The `examples/` directory contains real-world demonstrations:

### Linear Regression - Housing Price Prediction
```bash
python examples/linear_regression.py
```

Trains on the California housing dataset and compares performance between pure Python and C implementations. Includes:
- Performance timing comparison
- Loss improvement tracking (initial vs final)
- Test set metrics: MSE, RMSE, and R² score

### Logistic Regression - Rain Prediction
```bash
python examples/logistic_regression.py
```

Binary classification on the Australian weather dataset with performance benchmarking. Includes:
- Pure Python vs C implementation comparison
- Training time and speedup metrics
- Loss convergence tracking
- Test set accuracy evaluation

## Testing

Run the test suite:

```bash
python test.py
```

## Performance

The C implementation provides significant speedup over pure Python for large-scale operations through:

- Optimized BLAS-like operations tailored to regression pipelines
- Memory-efficient algorithms
- Compiled native code execution
- Vectorized math operations (sigmoid, loss functions)

Both example scripts include built-in performance benchmarks comparing pure Python implementations against the C library, typically showing speedups of 10-100x depending on dataset size. Each script reports:

- **Execution time** for both implementations
- **Speedup factor** (C vs pure Python)
- **Loss convergence** (initial loss → final loss with percentage reduction)
- **Model accuracy** on held-out test data (MSE/RMSE/R² for regression, accuracy for classification)

## Release Notes

### v0.2 (Current)
- Added logistic regression for binary classification
- Implemented sigmoid activation and binary cross-entropy loss
- Added `standardize` utility for feature normalization
- Performance benchmark examples for both linear and logistic regression
- Expanded C API with logistic regression functions

### v0.1
- Initial linear regression implementation with SGD and MSE
- Core math helpers: dot product, AXPY, matrix-vector multiplication
- Python bindings with numpy integration
----

## License

MIT