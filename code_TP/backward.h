#ifndef BACKWARD_H
#define BACKWARD_H
#include "matrix.h"
#include "ann.h"

__global__ void backwardInitGPU(
    double *a_l, double *y, double *delta_l, double *z_l, int numRows, int numColumns);

void backward_init(matrix_t *a_l, matrix_t *y, matrix_t *delta_l, matrix_t *z_l);

void backward_recursion(ann_t *nn, matrix_t *y);

void backward_assign(matrix_t *w_l, matrix_t *delta_l, matrix_t *a_lminus1, matrix_t *b_l, float alpha, int m);
#endif