// linear.h
// Linear regression functions
#ifndef LINEAR_H
#define LINEAR_H

// Loss function
double ml_mse(const double *y_pred, const double *y_true, int n);

// Gradient computation
void ml_linreg_mse_grad_w(const double *X, const double *y_true, const double *y_pred, double *grad_w, int n_rows, int n_cols);

// Training functions
double ml_linreg_sgd_step(const double *X, const double *y_true, double *w, int n_rows, int n_cols, double lr);

double ml_linreg_train(const double *X, const double *y_true, const double *w_init, double *w_out, int n_rows, int n_cols, int num_iters, double lr);

#endif // LINEAR_H
