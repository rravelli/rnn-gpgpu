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
        double sigmoid = 1 / (1 + exp(-z_l[idx]));
        delta_l[idx] = (a_l[idx] - y[idx]) * sigmoid * (1 - sigmoid);
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

    if (row < numWColumns && col < numDeltaColumns)
    {
        int idx = row * numDeltaColumns + col;
        float sum = 0;
        for (int ii = 0; ii < numDeltaRows; ii++)
        {
            sum += w_l[ii * numWColumns + row] * delta_l[ii * numDeltaColumns + col];
        }
        double sigmoid = 1 / (1 + exp(-z_lminus1[idx]));
        delta_lminus1[idx] = sum * sigmoid * (1 - sigmoid);
    }
}

void backward_recursion(matrix_t *w_l, matrix_t *delta_l, matrix_t *delta_lminus1, matrix_t *z_lminus1)
{
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)delta_lminus1->columns) / blockDim.x), ceil(((float)delta_lminus1->rows) / blockDim.y));
    backwardRecursionGPU<<<gridDim, blockDim>>>(w_l->m, delta_l->m, delta_lminus1->m, z_lminus1->m, w_l->rows, w_l->columns, delta_l->rows, delta_l->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}

__global__ void backwardAssignGPU(
    double *w_l, double *delta_l, double *a_lminus1, double *b_l, float alpha, int m, int numDeltaRows, int numDeltaColumns, int numARows, int numAColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numDeltaRows && col < numARows)
    {
        int idx = row * numARows + col;
        float sum = 0;
        float sum2 = 0;
        for (int ii = 0; ii < numAColumns; ii++)
        {
            sum += delta_l[row * numDeltaColumns + ii] * a_lminus1[col * numAColumns + ii];
            sum2 += b_l[row * numDeltaColumns + ii];
        }
        w_l[idx] -= alpha / m * sum;
        b_l[idx] -= alpha / m * sum2;
    }
}

void backward_assign(matrix_t *w_l, matrix_t *delta_l, matrix_t *a_lminus1, matrix_t *b_l, float alpha, int m)
{
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)w_l->columns) / blockDim.x), ceil(((float)w_l->rows) / blockDim.y));
    backwardAssignGPU<<<gridDim, blockDim>>>(w_l->m, delta_l->m, a_lminus1->m, b_l->m, alpha, m, delta_l->rows, delta_l->columns, a_lminus1->rows, a_lminus1->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}