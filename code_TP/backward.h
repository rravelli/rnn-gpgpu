#ifndef BACKWARD_H
#define BACKWARD_H
#include "matrix.h"
#include "ann.h"

void backward_recursion(ann_t *nn, matrix_t *y);

void backward_assign(matrix_t *w_l, matrix_t *delta_l, matrix_t *a_lminus1, matrix_t *b_l, float alpha, int m);
#endif