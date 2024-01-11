#ifndef FORWARD_H
#define FORWARD_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "matrix.h"
#include "ann.h"

void forward_operations(ann_t *nn, int l);

__global__ void forwardGPU(double *A, double *B, double *C, double *D,
                           int numARows, int numAColumns,
                           int numBRows, int numBColumns,
                           int numCRows, int numCColumns,
                           int numDRows, int numDColumns);

#endif