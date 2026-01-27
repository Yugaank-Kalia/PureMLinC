#include "ml.h"
#include <stdlib.h>

double ml_mse(const double *y_pred, const double *y_true, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = y_pred[i] - y_true[i];
        acc += diff * diff;
    }
    return acc / (double)n;
}

void ml_linreg_mse_grad_w(const double *X, const double *y_true, const double *y_pred, double *grad_w, int n_rows, int n_cols)
{
    // Initialize gradient to zero
    for (int j = 0; j < n_cols; ++j) {
        grad_w[j] = 0.0;
    }

    // For each sample, accumulate X[i, j] * (y_pred[i] - y_true[i])
    for (int i = 0; i < n_rows; ++i) {
        double diff = y_pred[i] - y_true[i];  // residual
        const double *row = X + (size_t)i * n_cols;
        for (int j = 0; j < n_cols; ++j) {
            grad_w[j] += row[j] * diff;
        }
    }

    // Scale by 2 / n_rows (MSE gradient factor)
    double scale = 2.0 / (double)n_rows;
    for (int j = 0; j < n_cols; ++j) {
        grad_w[j] *= scale;
    }
}


double ml_linreg_sgd_step(const double *X, const double *y_true, double *w, int n_rows, int n_cols, double lr)
{
    // Temporary buffers: y_pred (n_rows), grad_w (n_cols)
    double *y_pred = (double *)malloc((size_t)n_rows * sizeof(double));
    double *grad_w = (double *)malloc((size_t)n_cols * sizeof(double));

    if (!y_pred || !grad_w) {
        // crude error handling; in real code you might signal error
        if (y_pred) free(y_pred);
        if (grad_w) free(grad_w);
        return -1.0;
    }

    // 1) y_pred = X w
    ml_matvec(X, w, y_pred, n_rows, n_cols);

    // 2) grad_w = dL/dw
    ml_linreg_mse_grad_w(X, y_true, y_pred, grad_w, n_rows, n_cols);

    // 3) w <- w - lr * grad_w   (use axpy with a = -lr)
    // ml_axpy(n, a, x, y) does: y <- a*x + y
    ml_axpy(n_cols, -lr, grad_w, w);

    // 4) compute MSE loss for logging
    double loss = ml_mse(y_pred, y_true, n_rows);

    free(y_pred);
    free(grad_w);
    return loss;
}

double ml_linreg_train(const double *X, const double *y_true, const double *w_init, double *w_out, int n_rows, int n_cols, int num_iters, double lr)
{
    // Initialize w_out from w_init or zeros
    if (w_init) {
        for (int j = 0; j < n_cols; ++j) {
            w_out[j] = w_init[j];
        }
    } else {
        for (int j = 0; j < n_cols; ++j) {
            w_out[j] = 0.0;
        }
    }

    double loss = 0.0;
    for (int it = 0; it < num_iters; ++it) {
        loss = ml_linreg_sgd_step(X, y_true, w_out, n_rows, n_cols, lr);
        // Optionally, you could early-stop if loss stops changing much.
    }
    return loss;
}