#include "ml.h"
#include <math.h>
#include <stdlib.h>

void ml_sigmoid(const double *x, double *out, int n) {
    for (int i = 0; i < n; ++i) {
        double z = x[i];
        out[i] = 1.0 / (1.0 + exp(-z));
    }
}

double ml_logistic_loss(const double *p_pred, const double *y_true, int n)
{
    const double eps = 1e-15;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double p = p_pred[i];
        if (p < eps) p = eps;
        if (p > 1.0 - eps) p = 1.0 - eps;

        double y = y_true[i];
        acc += y * log(p) + (1.0 - y) * log(1.0 - p);
    }
    return -acc / (double)n;
}

void ml_logreg_grad_w(const double *X, const double *y_true, const double *p_pred, double *grad_w, int n_rows, int n_cols)
{
    for (int j = 0; j < n_cols; ++j) {
        grad_w[j] = 0.0;
    }

    for (int i = 0; i < n_rows; ++i) {
        double diff = p_pred[i] - y_true[i];  // (p - y)
        const double *row = X + (size_t)i * n_cols;
        for (int j = 0; j < n_cols; ++j) {
            grad_w[j] += row[j] * diff;
        }
    }

    double scale = 1.0 / (double)n_rows;
    for (int j = 0; j < n_cols; ++j) {
        grad_w[j] *= scale;
    }
}

double ml_logreg_sgd_step(const double *X, const double *y_true, double *w, int n_rows, int n_cols, double lr)
{
    double *z = (double *)malloc((size_t)n_rows * sizeof(double));
    double *p = (double *)malloc((size_t)n_rows * sizeof(double));
    double *grad_w = (double *)malloc((size_t)n_cols * sizeof(double));

    if (!z || !p || !grad_w) {
        if (z) free(z);
        if (p) free(p);
        if (grad_w) free(grad_w);
        return -1.0;  // crude error indicator
    }

    // 1) z = X w
    ml_matvec(X, w, z, n_rows, n_cols);

    // 2) p = sigmoid(z)
    ml_sigmoid(z, p, n_rows);

    // 3) grad_w = (1/n) * X^T (p - y)
    ml_logreg_grad_w(X, y_true, p, grad_w, n_rows, n_cols);

    // 4) w <- w - lr * grad_w   (use axpy with a = -lr)
    ml_axpy(n_cols, -lr, grad_w, w);

    // 5) compute logistic loss for logging
    double loss = ml_logistic_loss(p, y_true, n_rows);

    free(z);
    free(p);
    free(grad_w);
    return loss;
}

double ml_logreg_train(const double *X, const double *y_true, const double *w_init, double *w_out, int n_rows, int n_cols, int num_iters, double lr)
{
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
        loss = ml_logreg_sgd_step(X, y_true, w_out, n_rows, n_cols, lr);
    }
    return loss;
}