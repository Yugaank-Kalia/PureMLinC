import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path to import from python module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.ml_core import linreg_fit, standardize


def load_housing(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(path)
    df = df.drop(columns=["Address"])

    X = df.iloc[:, :5].to_numpy(dtype=np.float64)
    y = df.iloc[:, 5].to_numpy(dtype=np.float64)

    X_std, X_mean, X_scale = standardize(X)
    y_std, y_mean, y_scale = standardize(y)

    meta = {
        "X_mean": X_mean,
        "X_scale": X_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }
    return X_std, y_std, meta


def pure_python_linreg_fit(X: list[list[float]], y: list[float], num_iters: int, lr: float) -> tuple[list[float], float]:
    """Pure Python linear regression with bias using batch GD (no numpy operations)."""
    n_rows = len(X)
    n_cols = len(X[0])
    
    # Initialize weights (bias + features)
    w = [0.0] * (n_cols + 1)
    
    for _ in range(num_iters):
        # Compute predictions: y_pred = X_ext @ w
        y_pred = []
        for i in range(n_rows):
            pred = w[0]  # bias
            for j in range(n_cols):
                pred += X[i][j] * w[j + 1]
            y_pred.append(pred)
        
        # Compute residuals
        residuals = [y_pred[i] - y[i] for i in range(n_rows)]
        
        # Compute MSE
        mse = sum(r * r for r in residuals) / n_rows
        
        # Compute gradients
        grad = [0.0] * (n_cols + 1)
        
        # Gradient for bias
        grad[0] = (2.0 / n_rows) * sum(residuals)
        
        # Gradients for features
        for j in range(n_cols):
            grad[j + 1] = (2.0 / n_rows) * sum(residuals[i] * X[i][j] for i in range(n_rows))
        
        # Update weights
        for j in range(n_cols + 1):
            w[j] -= lr * grad[j]
    
    # Final loss
    y_pred = []
    for i in range(n_rows):
        pred = w[0]
        for j in range(n_cols):
            pred += X[i][j] * w[j + 1]
        y_pred.append(pred)
    
    residuals = [y_pred[i] - y[i] for i in range(n_rows)]
    final_loss = sum(r * r for r in residuals) / n_rows
    
    return w, final_loss


def main() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "housing.csv"
    X_std, y_std, meta = load_housing(data_path)

    print("=" * 70)
    print("Performance Comparison: Pure Python vs C Library")
    print("=" * 70)
    print(f"Dataset: {X_std.shape[0]} samples, {X_std.shape[1]} features")
    print(f"Training: 3000 iterations, learning rate 0.01")
    print()

    # Convert to Python lists for pure Python implementation
    X_list = X_std.tolist()
    y_list = y_std.tolist()

    # Time pure Python implementation
    print("Running pure Python implementation...")
    start_time = time.perf_counter()
    w_python, loss_python = pure_python_linreg_fit(X_list, y_list, num_iters=3000, lr=0.01)
    python_time = time.perf_counter() - start_time
    print(f"  Time: {python_time:.4f} seconds")
    print(f"  Final loss: {loss_python:.6f}")
    print()

    # Time C library implementation
    print("Running C library implementation...")
    start_time = time.perf_counter()
    w_ext, final_loss = linreg_fit(X_std, y_std, num_iters=3000, lr=0.01)
    c_time = time.perf_counter() - start_time
    print(f"  Time: {c_time:.4f} seconds")
    print(f"  Final loss: {final_loss:.6f}")
    print()

    # Compute speedup
    speedup = python_time / c_time
    print("=" * 70)
    print(f"Speedup: {speedup:.2f}x faster with C library")
    print("=" * 70)
    print()

    # Verify results match
    w_diff = sum(abs(w_python[i] - w_ext[i]) for i in range(len(w_python))) / len(w_python)
    print(f"Average weight difference: {w_diff:.6e} (should be small)")
    print()

    # Show final results on original scale
    X_ext_np = np.c_[np.ones(X_std.shape[0]), X_std]
    y_pred_std = X_ext_np @ w_ext
    y_pred = meta["y_mean"] + meta["y_scale"] * y_pred_std

    y_raw = meta["y_mean"] + meta["y_scale"] * y_std
    mse = float(((y_pred - y_raw) ** 2).mean())

    print("Final model performance (C library):")
    print(f"  Unscaled MSE on dataset: {mse:.2f}")
    print(f"  Weights (bias first) on standardized features:")
    print(f"  {w_ext}")


if __name__ == "__main__":
    main()