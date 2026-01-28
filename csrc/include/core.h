// core.h
// Core linear algebra operations
#ifndef CORE_H
#define CORE_H

// Basic operations
double ml_add_double(double a, double b);

// Vector operations
double ml_dot(const double *x, const double *y, int n);

void ml_axpy(int n, double a, const double *x, double *y);

// Matrix operations
void ml_matvec(const double *X, const double *w, double *y, int n_rows, int n_cols);

// Activation functions
void ml_sigmoid(const double *x, double *out, int n);

#endif // CORE_H
