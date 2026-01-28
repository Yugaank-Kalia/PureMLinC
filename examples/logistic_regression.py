import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import from python module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.ml_core import standardize, logreg_fit, ml_matvec, train_test_split


def load_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["RainTomorrow"])

    y = (df["RainTomorrow"] == "Yes").astype(np.float64).to_numpy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [c for c in numeric_cols if c != "RainTomorrow"]

    X = df[numeric_cols].to_numpy(dtype=np.float64)

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y

import math
from typing import List, Tuple


def py_sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def py_logreg_fit(X: List[List[float]], y: List[float], num_iters: int = 2000, lr: float = 0.1) -> Tuple[float, List[float], float]:
    """
    Pure Python logistic regression with bias.
    
    X: list of samples, each sample is a list of features.
    y: list of 0/1 labels (same length as X).
    
    Returns (w_ext, final_loss) where:
        - w_ext[0] is the bias term
        - w_ext[1:] are the feature weights
    """
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Empty dataset")
    n_features = len(X[0])

    # Initialize weights (bias + feature weights)
    w_ext = [0.0] * (n_features + 1)

    loss = 0.0
    for _ in range(num_iters):
        grad = [0.0] * (n_features + 1)
        loss = 0.0

        for i in range(n_samples):
            xi = X[i]
            yi = y[i]

            # z = b + w^T x
            z = w_ext[0]
            for j in range(n_features):
                z += w_ext[j + 1] * xi[j]

            p = py_sigmoid(z)

            # Binary cross-entropy
            eps = 1e-12
            loss += -(yi * math.log(max(p, eps)) +
                      (1.0 - yi) * math.log(max(1.0 - p, eps)))

            diff = p - yi  # dL/dz
            grad[0] += diff
            for j in range(n_features):
                grad[j + 1] += diff * xi[j]

        # Average loss and gradients
        inv_n = 1.0 / n_samples
        loss *= inv_n
        for j in range(n_features + 1):
            grad[j] *= inv_n

        # Gradient step
        for j in range(n_features + 1):
            w_ext[j] -= lr * grad[j]

    return w_ext[0], w_ext[1:], loss

def main():
    csv_path = "examples/data/weatherAUS.csv"

    X, y = load_data(csv_path)
    X_std, _, _ = standardize(X)

    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_ratio=0.2, seed=42)

    lr = 0.3
    num_iters = 2000

    print("Running pure Python implementation...")
    start_time = time.perf_counter()
    b_python, w_python, final_loss_python = py_logreg_fit(X_train.tolist(), y_train.tolist(), num_iters, lr)
    python_time = time.perf_counter() - start_time
    print(f"  Time: {python_time:.4f} seconds")
    print(f"  Final loss: {final_loss_python:.6f}")
    print()

    # Time C library implementation
    print("Running C library implementation...")
    start_time = time.perf_counter()
    b_c, w_c, final_loss_c = logreg_fit(X_train, y_train, num_iters, lr)
    c_time = time.perf_counter() - start_time
    print(f"  Time: {c_time:.4f} seconds")
    print(f"  Final loss: {final_loss_c:.6f}")
    print()

    # Compute speedup
    speedup = python_time / c_time
    print("=" * 70)
    print(f"Speedup: {speedup:.2f}x faster with C library")
    print("=" * 70)
    print()

    # Verify results match
    w_diff = sum(abs(w_python[i] - w_c[i]) for i in range(len(w_python))) / len(w_python)
    print(f"Average weight difference: {w_diff:.6e} (should be small)")
    print()

    logits = b_c + ml_matvec(X_test, w_c)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(np.float64)

    acc = (y_pred == y_test).mean()
    print('\n' + '='*60)
    print(f"Bias: {b_c}")
    print(f"Final weights: {w_c[:5]}, {w_c.shape}")
    print(f"Final loss: {final_loss_c}")
    print(f"Test accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()