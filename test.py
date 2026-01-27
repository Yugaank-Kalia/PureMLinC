import numpy as np
from python.ml_core import (
    ml_add_double,
    ml_dot,
    ml_axpy,
    ml_matvec,
    ml_mse,
    ml_linreg_mse_grad_w,
    ml_linreg_sgd_step,
    ml_linreg_train
)

from python.ml_core.linreg import linreg_fit

print('Adding')
print(ml_add_double(1.5, 2.25))
print('='*60 + '\n')

x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y = np.array([10.0, 20.0, 30.0], dtype=np.float64)

print('Matrix dot')
print(ml_dot(x, y), np.dot(x, y))
print('='*60 + '\n')

print('y <= ax + y')
print(ml_axpy(0.5, x, y))
print('='*60 + '\n')

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
w = np.array([0.5, -1.0], dtype=np.float64)

print("Matrix mult")
print(ml_matvec(X, w))
print(X @ w)
print('='*60 + '\n')

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.5, 1.0, 4.0])

print("Mean Squared Error")
print(ml_mse(y_pred, y_true))
print(((y_pred - y_true) ** 2).mean())
print('='*60 + '\n')

X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
w = np.array([0.5, -1.0], dtype=np.float64)
y_true = np.array([0.0, 1.0], dtype=np.float64)

y_pred = ml_matvec(X, w)
loss_c = ml_mse(y_pred, y_true)
grad_c = ml_linreg_mse_grad_w(X, y_true, y_pred)

res = y_pred - y_true
loss_np = (res ** 2).mean()
grad_np = (2.0 / X.shape[0]) * X.T @ res

print('Loss and Gradient calc')
print("loss_c, loss_np:", loss_c, loss_np)
print("grad_c:", grad_c)
print("grad_np:", grad_np)
print('='*60 + '\n')

rng = np.random.default_rng(0)
n, d = 100, 1
X = rng.normal(size=(n, d))
true_w = np.array([3.0])
y = (X @ true_w + 0.1 * rng.normal(size=(n,))).astype(np.float64)

w_final, loss_final = ml_linreg_train(
    X, y, num_iters=1000, lr=0.1, w_init=None
)

print("Training")
print("final loss:", loss_final)
print("final w:", w_final)
print("true w:", true_w)
print('='*60 + '\n')

rng = np.random.default_rng(0)
n = 100
X = rng.normal(size=(n, 1))
true_b = 1.0
true_w = np.array([3.0])
y = (true_b + X @ true_w + 0.1 * rng.normal(size=(n,))).astype(np.float64)

w_ext, final_loss = linreg_fit(X, y, num_iters=1000, lr=0.1)
b_hat = w_ext[0]
w_hat = w_ext[1:]

print("Training with bias")
print("b_hat:", b_hat, "true_b:", true_b)
print("w_hat:", w_hat, "true_w:", true_w)
print("final loss:", final_loss)