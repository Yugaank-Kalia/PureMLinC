// logistic.h
// Logistic regression functions
#ifndef LOGISTIC_H
#define LOGISTIC_H

// Loss function
double ml_logistic_loss(const double *p_pred, const double *y_true, int n);

// Gradient computation
void ml_logreg_grad_w(const double *X, const double *y_true, const double *p_pred, double *grad_w, int n_rows, int n_cols);

// Training functions
double ml_logreg_sgd_step(const double *X, const double *y_true, double *w, int n_rows, int n_cols, double lr);

double ml_logreg_train(const double *X, const double *y_true, const double *w_init, double *w_out, int n_rows, int n_cols, int num_iters, double lr);

#endif // LOGISTIC_H
