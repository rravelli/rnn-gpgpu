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
    matrix_t **w, matrix_t **delta, matrix_t **z, int numLayers)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum;
    for (int l = numLayers - 1; l > 1; l--)
    {
        if (row < w[l]->columns && col < delta[l]->columns)
        {
            int idx = row * delta[l]->columns + col;
            sum = 0;
            for (int ii = 0; ii < delta[l]->rows; ii++)
            {
                sum += w[l]->m[ii * w[l]->columns + row] * delta[l]->m[ii * delta[l]->columns + col];
            }
            double sigmoid = 1 / (1 + exp(-z[l - 1]->m[idx]));
            delta[l - 1]->m[idx] = sum * sigmoid * (1 - sigmoid);
        }
        __syncthreads();
    }
}

void backward_recursion(ann_t *nn)
{
    matrix_t **w, **delta, **z;
    int numLayers = nn->number_of_layers;
    cudaMallocManaged(&w, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&delta, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&z, numLayers * sizeof(matrix_t));
    int maxRows = 0;
    int maxCol = 0;
    for (int i = 0; i < nn->number_of_layers; i++)
    {
        w[i] = nn->layers[i]->weights;
        delta[i] = nn->layers[i]->delta;
        z[i] = nn->layers[i]->z;

        if (nn->layers[i]->delta->rows > maxRows)
        {
            maxRows = nn->layers[i]->delta->rows;
        }
        if (nn->layers[i]->delta->columns > maxCol)
        {
            maxCol = nn->layers[i]->delta->columns;
        }
    }
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)maxCol) / blockDim.x), ceil(((float)maxRows) / blockDim.y));
    backwardRecursionGPU<<<gridDim, blockDim>>>(w, delta, z, numLayers);

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
        bool assignB = (idx < numDeltaRows);
        float sum = 0;
        float sum2 = 0;
        for (int ii = 0; ii < numAColumns; ii++)
        {
            sum += delta_l[row * numDeltaColumns + ii] * a_lminus1[col * numAColumns + ii];
            if (assignB)
            {
                sum2 += b_l[row * numDeltaColumns + ii];
            }
        }
        w_l[idx] -= alpha / m * sum;
        if (assignB)
        {
            b_l[idx] -= alpha / m * sum2;
        }
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