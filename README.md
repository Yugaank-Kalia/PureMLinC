# Machine Learning in C

Version 0.1 — Linear Regression

A high-performance machine learning library written in C with Python bindings. Version 0.1 focuses on linear regression built on top of a small, optimized linear algebra core.

## Features (v0.1)

- Dense linear regression (single and multi-output) trained with SGD
- Mean Squared Error (MSE) loss and gradient routines
- Core BLAS-style helpers used by the model: dot product, AXPY, 2D matmul
- Python bindings for straightforward integration in Python workflows
- Shared library build for native performance
- Minimal dependency footprint (only `numpy` for Python examples)

## Project Structure

```
├── csrc/                    # C source code
│   ├── include/
│   │   └── ml.h            # C API header
│   ├── src/
│   │   ├── core.c          # Core linear algebra functions
│   │   └── linreg.c        # Linear regression implementation
│   └── Makefile            # Build configuration
├── python/                  # Python bindings and utilities
│   ├── ml_core/
│   │   ├── __init__.py
│   │   ├── _lib.py         # FFI binding to C library
│   │   └── linreg.py       # Python API for linear regression
│   └── requirements.txt    # Python dependencies
├── examples/               # Example usage
└── test.py                 # Test suite
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

## Usage (v0.1)

### Python API

```python
from ml_core.linreg import LinearRegression
import numpy as np

# Create and train a linear regression model
model = LinearRegression()
X = np.random.randn(100, 5)
y = np.random.randn(100)

model.fit(X, y, num_iters=100, lr=0.01)
predictions = model.predict(X)
```

For multi-output regression, pass `Y` with shape `(n_samples, n_outputs)`; the API handles the multi-target case.

### Core C API (v0.1)

Available symbols (see [csrc/include/ml.h](csrc/include/ml.h)):

- `ml_dot` — Dot product
- `ml_axpy` — Vector operation: Y = a*X + Y
- `ml_matmul2d` — 2D matrix multiplication
- `ml_mse_multi` — Multi-output MSE loss
- `ml_linreg_mse_grad_W_multi` — Gradient of MSE w.r.t. weights
- `ml_linreg_sgd_step_multi` — Single SGD step for multi-output regression
- `ml_linreg_train` — Full linear regression training loop

## Testing

Run the test suite:

```bash
python test.py
```

## Performance

The C implementation provides significant speedup over pure Python for large-scale operations through:

- Optimized BLAS-like operations tailored to the linear regression pipeline
- Memory-efficient algorithms
- Compiled native code execution

## Release Notes (v0.1)

- Initial linear regression implementation with SGD and MSE
- Multi-output training path exposed in both C and Python APIs
- Core math helpers stabilized for regression workloads

## License

TBD
