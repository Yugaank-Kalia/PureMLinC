# Machine Learning in C

Version 0.2 — Linear and Logistic Regression

A high-performance machine learning library written in C with Python bindings. Version 0.2 includes both linear and logistic regression built on top of a small, optimized linear algebra core.

## Features (v0.2)

- Dense linear regression trained with batch gradient descent (GD)
- Logistic regression for binary classification with sigmoid activation
- Loss functions: Mean Squared Error (MSE) and Binary Cross-Entropy
- Gradient computation and batch GD optimization routines
- Core BLAS-style helpers: dot product, AXPY, matrix-vector multiplication
- Data preprocessing utilities (standardization)
- Python bindings for straightforward integration in Python workflows
- Shared library build for native performance
- Performance benchmarks comparing pure Python vs C implementations
- Minimal dependency footprint (only `numpy` for core, `pandas` for examples)

## Project Structure

```
├── csrc/                          # C source code
│   ├── include/
│   │   ├── ml.h                   # Main C API header
│   │   ├── core.h                 # Core linear algebra functions
│   │   ├── linear.h               # Linear regression
│   │   └── logistic.h             # Logistic regression
│   ├── src/
│   │   ├── core.c                 # Core linear algebra implementation
│   │   ├── linreg.c               # Linear regression implementation
│   │   └── logreg.c               # Logistic regression implementation
│   ├── build/                     # Compiled library output
│   └── Makefile                   # Build configuration
├── python/                        # Python bindings and utilities
│   ├── ml_core/
│   │   ├── __init__.py            # Package exports
│   │   ├── _lib.py                # FFI binding to C library
│   │   ├── linreg.py              # Python API for linear regression
│   │   └── logreg.py              # Python API for logistic regression
│   └── requirements.txt           # Python dependencies
├── examples/                      # Example usage and benchmarks
│   ├── linear_regression.py       # Housing price prediction
│   ├── logistic_regression.py     # Rain prediction (binary classification)
│   └── data/
│       ├── housing.csv            # California housing dataset
│       └── weatherAUS.csv         # Australian weather dataset
├── .github/workflows/
│   └── publish.yml                # PyPI publishing workflow
├── pyproject.toml                 # Package configuration
├── MANIFEST.in                    # Package manifest
└── test.py                        # Test suite
```

Key public interfaces live in [csrc/include/ml.h](csrc/include/ml.h) and the Python bindings in [python/ml_core](python/ml_core).

## Building

### Prerequisites

- C compiler (clang/gcc)
- Python 3.9+ with numpy
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

For running examples:
```bash
pip install pandas  # Required for example scripts
```

### Set Up a Virtual Environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r python/requirements.txt
```

When finished, deactivate with `deactivate`. The `.venv` folder is ignored in git.

### Install as Package

To install the package for development:
```bash
pip install -e .
```

## Usage (v0.2)

### Linear Regression

The library uses batch gradient descent for training. Here's how to use it:

```python
import numpy as np
from ml_core import linreg_fit, standardize

# Generate or load your data
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
true_w = np.array([1.0, 2.0, -1.5, 0.5, 3.0])
bias = 0.7
y = bias + X @ true_w + 0.1 * rng.normal(size=(200,))

# Standardize for stable training (important!)
X_std, X_mean, X_scale = standardize(X)
y_std, y_mean, y_scale = standardize(y)

# Train with batch gradient descent
# Returns: w_ext[0] is bias, w_ext[1:] are feature weights
w_ext, final_loss = linreg_fit(X_std, y_std, num_iters=3000, lr=0.01)

# Make predictions on original scale
X_ext = np.c_[np.ones(X_std.shape[0]), X_std]  # Add bias column
y_pred_std = X_ext @ w_ext
y_pred = y_mean + y_scale * y_pred_std  # Denormalize
```

**Returns:** `linreg_fit` returns `(w_ext, final_loss)` where:
- `w_ext`: Array with bias at index 0, feature weights at indices 1+
- `final_loss`: Final mean squared error on standardized data

See [examples/linear_regression.py](examples/linear_regression.py) for a complete example with the housing dataset.

### Logistic Regression

Binary classification with sigmoid activation:

```python
import numpy as np
from ml_core import logreg_fit, standardize

# Binary classification data
X = np.random.randn(500, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)  # Binary labels (0 or 1)

# Standardize features (important for convergence)
X_std, X_mean, X_scale = standardize(X)

# Train logistic regression
bias, weights, final_loss = logreg_fit(X_std, y, num_iters=2000, lr=0.1)

# Make predictions
logits = bias + X_std @ weights
probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid activation
y_pred = (probs >= 0.5).astype(int)    # Binary predictions

# Evaluate
accuracy = (y_pred == y).mean()
print(f"Accuracy: {accuracy:.3f}")
```

**Returns:** `logreg_fit` returns `(bias, weights, final_loss)` where:
- `bias`: Scalar bias term
- `weights`: Feature weight vector (1D array)
- `final_loss`: Final binary cross-entropy loss

See [examples/logistic_regression.py](examples/logistic_regression.py) for a complete example with the weather dataset.

### Data Preprocessing

The `standardize` utility normalizes features for stable training:

```python
from ml_core import standardize

# Standardize features
X_std, X_mean, X_scale = standardize(X)

# X_std has mean 0 and std 1
# Use X_mean and X_scale to transform new data or denormalize predictions
```

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
- `ml_linreg_sgd_step` — Single gradient descent step
- `ml_linreg_train` — Full linear regression training loop

**Logistic Regression:**
- `ml_logistic_loss` — Binary cross-entropy loss
- `ml_logreg_grad_w` — Gradient of logistic loss w.r.t. weights
- `ml_logreg_sgd_step` — Single gradient descent step
- `ml_logreg_train` — Full logistic regression training loop

## Examples

The `examples/` directory contains real-world demonstrations with performance benchmarks:

### Linear Regression - Housing Price Prediction
```bash
python examples/linear_regression.py
```

Trains on the California housing dataset ([examples/data/housing.csv](examples/data/housing.csv)) and compares performance between pure Python and C implementations. The script reports:

- **Performance timing comparison** (pure Python vs C library)
- **Speedup factor** (typically 10-100x)
- **Loss convergence** (initial → final loss with % reduction)
- **Test set metrics:** MSE, RMSE, and R² score
- **Model coefficients** for interpretation

### Logistic Regression - Rain Prediction
```bash
python examples/logistic_regression.py
```

Binary classification on the Australian weather dataset ([examples/data/weatherAUS.csv](examples/data/weatherAUS.csv)) with performance benchmarking. The script reports:

- **Pure Python vs C implementation comparison**
- **Training time and speedup metrics**
- **Loss convergence tracking**
- **Test set accuracy evaluation**
- **Weight analysis**

Both examples include built-in train/test splits and comprehensive performance metrics.

## Testing

Run the test suite:

```bash
python test.py
```

The test suite validates:
- Core linear algebra operations (dot product, AXPY, matrix-vector multiplication)
- Linear regression training and predictions
- Logistic regression training and predictions
- Loss function computations
- Python-C interface consistency

## Performance

The C implementation provides significant speedup over pure Python for large-scale operations through:

- **Optimized BLAS-like operations** tailored to regression pipelines
- **Memory-efficient algorithms** with minimal allocations
- **Compiled native code execution** (no interpreter overhead)
- **Vectorized math operations** (sigmoid, loss functions)
- **Batch gradient descent** for stable convergence

Both example scripts include built-in performance benchmarks comparing pure Python implementations against the C library, typically showing **speedups of 10-100x** depending on dataset size and number of iterations.

### Benchmark Results

Each example script reports:
- **Execution time** for both implementations
- **Speedup factor** (C vs pure Python)
- **Loss convergence** (initial loss → final loss with percentage reduction)
- **Model accuracy** on held-out test data (MSE/RMSE/R² for regression, accuracy for classification)

The pure Python implementations are included in the examples for direct comparison and educational purposes.

## Installation

### From PyPI (when published)
```bash
pip install PureMLinC
```

### From Source
```bash
git clone https://github.com/yugaank/machine-learning-in-c.git
cd machine-learning-in-c
cd csrc && make && cd ..
pip install -e .
```

## Release Notes

### v0.2 (Current)
- Added logistic regression for binary classification
- Implemented sigmoid activation and binary cross-entropy loss
- Added `standardize` utility for feature normalization
- Performance benchmark examples for both linear and logistic regression
- Expanded C API with logistic regression functions
- Added train/test split utilities in examples
- Comprehensive documentation and usage examples
- PyPI publishing workflow

### v0.1
- Initial linear regression implementation with batch GD and MSE
- Core math helpers: dot product, AXPY, matrix-vector multiplication
- Python bindings with numpy integration
- Basic example scripts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Homepage:** https://github.com/yugaank/PureMLinC
- **Documentation:** https://github.com/yugaank/PureMLinC#readme