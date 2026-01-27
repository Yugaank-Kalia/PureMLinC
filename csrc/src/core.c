// core.c
#include "ml.h"
#include <stdlib.h>

double ml_add_double(double a, double b) {
    return a + b;
}

double ml_dot(const double *x, const double *y, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += x[i] * y[i];
    }
    return acc;
}

void ml_axpy(int n, double a, const double *x, double *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void ml_matvec(const double *X, const double *w, double *y, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; ++i) {
        double acc = 0.0;
        const double *row = X + (size_t)i * n_cols;
        for (int j = 0; j < n_cols; ++j) {
            acc += row[j] * w[j];
        }
        y[i] = acc;
    }
}