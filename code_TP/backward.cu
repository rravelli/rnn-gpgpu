#include "backward.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void backwardRecursionGPU(
    matrix_t **w, matrix_t **delta, matrix_t **z, matrix_t **a, matrix_t **b, matrix_t *y, int numLayers, float alpha, int m)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum, sum2;
    // init
    if (row < delta[numLayers - 1]->rows && col < delta[numLayers - 1]->columns)
    {
        int idx = row * delta[numLayers - 1]->columns + col;
        double sigmoid = 1 / (1 + exp(-z[numLayers - 1]->m[idx]));
        // δ(L) = (a(L) - y) ◦ f'(z(l))
        delta[numLayers - 1]->m[idx] = (a[numLayers - 1]->m[idx] - y->m[idx]) * sigmoid * (1 - sigmoid);
    }
    __syncthreads();
    // recursion on the layers
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
            // δ(l−1) = (w(l))T × δ(l) ◦ f′(z(l−1))
            delta[l - 1]->m[idx] = sum * sigmoid * (1 - sigmoid);
        }
        __syncthreads();
    }
}

void backward_recursion(ann_t *nn, matrix_t *y)
{
    matrix_t **w, **delta, **z, **a, **b;
    int numLayers = nn->number_of_layers;
    cudaMallocManaged(&w, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&delta, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&z, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&a, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&b, numLayers * sizeof(matrix_t));
    int maxRows = 0;
    int maxCol = 0;
    for (int i = 0; i < nn->number_of_layers; i++)
    {
        w[i] = nn->layers[i]->weights;
        delta[i] = nn->layers[i]->delta;
        a[i] = nn->layers[i]->activations;
        z[i] = nn->layers[i]->z;
        b[i] = nn->layers[i]->biases;

        if (nn->layers[i]->weights->rows > maxRows)
        {
            maxRows = nn->layers[i]->weights->rows;
        }
        if (nn->layers[i]->weights->columns > maxCol)
        {
            maxCol = nn->layers[i]->weights->columns;
        }
    }
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)maxCol) / blockDim.x), ceil(((float)maxRows) / blockDim.y));
    backwardRecursionGPU<<<gridDim, blockDim>>>(w, delta, z, a, b, y, numLayers, nn->alpha, nn->minibatch_size);

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