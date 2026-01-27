// ml.h
#ifndef ML_H
#define ML_H

double ml_add_double(double a, double b);

double ml_dot(const double *x, const double *y, int n);

void ml_axpy(int n, double a, const double *x, double *y);

void ml_matvec(const double *X, const double *w, double *y, int n_rows, int n_cols);

double ml_mse(const double *y_pred, const double *y_true, int n);

void ml_linreg_mse_grad_w(const double *X, const double *y_true, const double *y_pred, double *grad_w, int n_rows, int n_cols);

double ml_linreg_sgd_step(const double *X, const double *y_true, double *w, int n_rows, int n_cols, double lr);

double ml_linreg_train(const double *X, const double *y_true, const double *w_init, double *w_out, int n_rows, int n_cols, int num_iters, double lr);

#endif // ML_H