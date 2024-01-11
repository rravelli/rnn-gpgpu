#ifndef BACKWARD_H
#define BACKWARD_H
#include "matrix.h"

__global__ void backwardInitGPU(
    double *a_l, double *y, double *delta_l, double *z_l, int numRows, int numColumns);

void backward_init(matrix_t *a_l, matrix_t *y, matrix_t *delta_l, matrix_t *z_l);

void backward_recursion(matrix_t *w_l, matrix_t *delta_l, matrix_t *delta_lminus1, matrix_t *z_lminus1);

void backward_assign(matrix_t *w_l, matrix_t *delta_l, matrix_t *a_lminus1, float alpha, int m);
#endif