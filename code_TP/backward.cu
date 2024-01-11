#include "backward.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* Init delta_L in backward algorithm */
__global__ void backwardInitGPU(
    double *a_l, double *y, double *delta_l, double *z_l, int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numColumns)
    {
        int idx = row * numColumns + col;
        delta_l[idx] = (a_l[idx] - y[idx]) * 1 / (1 + exp(-z_l[idx])) * (1 - 1 / (1 + exp(-z_l[idx])));
    }
}

void backward_init(matrix_t *a_l, matrix_t *y, matrix_t *delta_l, matrix_t *z_l)
{
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)delta_l->columns) / blockDim.x), ceil(((float)delta_l->rows) / blockDim.y));
    backwardInitGPU<<<gridDim, blockDim>>>(a_l->m, y->m, delta_l->m, z_l->m, delta_l->rows, delta_l->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}

__global__ void backwardRecursionGPU(
    double *w_l, double *delta_l, double *delta_lminus1, double *z_lminus1, int numWRows, int numWColumns, int numDeltaRows, int numDeltaColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numWRows && col < numDeltaColumns)
    {
        int idx = row * numDeltaColumns + col;
        float sum = 0;
        for (int ii = 0; ii < numWColumns; ii++)
        {
            sum += w_l[ii * numWRows + row] * delta_l[ii * numDeltaColumns + col];
        }
        delta_lminus1[idx] = sum * 1 / (1 + exp(-z_lminus1[idx])) * (1 - 1 / (1 + exp(-z_lminus1[idx])));
    }
}

void backward_recursion(matrix_t *w_l, matrix_t *delta_l, matrix_t *delta_lminus1, matrix_t *z_lminus1)
{
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)delta_lminus1->columns) / blockDim.x), ceil(((float)delta_lminus1->rows) / blockDim.y));
    backwardRecursionGPU<<<gridDim, blockDim>>>(w_l->m, delta_l->m, delta_lminus1->m, z_lminus1->m, delta_l->rows, delta_l->columns, delta_lminus1->rows, delta_lminus1->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}