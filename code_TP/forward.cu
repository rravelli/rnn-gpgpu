#include "forward.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

__global__ void forwardGPU(double *A, double *B, double *C, double *res1, double *res2,
                           int numARows, int numAColumns,
                           int numBRows, int numBColumns,
                           int numCRows, int numCColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns)
    {
        float sumAB = 0;
        float sumC = 0;
        int idx = row * numBColumns + col;
        for (int ii = 0; ii < numAColumns; ii++)
        {
            sumAB += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        for (int ii = 0; ii < numCColumns; ii++)
        {
            sumC += C[row * numCColumns + ii];
        }
        res1[idx] = sumAB + sumC;
        res2[idx] = 1 / (1 + exp(-res1[idx]));
        
    }
}

void forward_operations(ann_t *nn, int l)
{
    // check dimensions for w^l x a^(l-1)
    assert(nn->layers[l]->weights->columns == nn->layers[l - 1]->activations->rows);

    // check dimensions for z^l = w^l x a^(l-1) + b^l x 1
    assert((nn->layers[l]->weights->rows == nn->layers[l]->biases->rows) &&
           (nn->layers[l - 1]->activations->columns == nn->layers[l]->z->columns) &&
           (nn->layers[l]->weights->rows == nn->layers[l]->z->rows));

    // check dimensions for f(z^l)
    assert((nn->layers[l]->z->columns == nn->layers[l]->activations->columns) &&
           (nn->layers[l]->z->rows == nn->layers[l]->activations->rows));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)nn->layers[l]->activations->columns) / blockDim.x), ceil(((float)nn->layers[l]->activations->rows) / blockDim.y));
    forwardGPU<<<gridDim, blockDim>>>(nn->layers[l]->weights->m, nn->layers[l - 1]->activations->m,
                                      nn->layers[l]->biases->m,
                                      nn->layers[l]->z->m, nn->layers[l]->activations->m,
                                      nn->layers[l]->weights->rows, nn->layers[l]->weights->columns,
                                      nn->layers[l - 1]->activations->rows, nn->layers[l - 1]->activations->columns,
                                      nn->layers[l]->biases->rows, nn->layers[l]->biases->columns);

    cudaDeviceSynchronize();
}