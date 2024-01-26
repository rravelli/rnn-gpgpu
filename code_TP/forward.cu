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

__global__ void forwardGPU(matrix_t **w, matrix_t **a, matrix_t **z, matrix_t **b, int numLayers)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int l = 1; l < numLayers; l++)
    {
        if (row < w[l]->rows && col < a[l - 1]->columns)
        {
            float sumAB = 0;
            int idx = row * a[l - 1]->columns + col;

            // A x B
            for (int ii = 0; ii < w[l]->columns; ii++)
            {
                sumAB += w[l]->m[row * w[l]->columns + ii] * a[l - 1]->m[ii * a[l - 1]->columns + col];
            }
            z[l]->m[idx] = sumAB + b[l]->m[row * b[l]->columns]; // res1 = A * B + C
            a[l]->m[idx] = 1 / (1 + exp(-z[l]->m[idx]));         // res2 = sigmoid(res1)
        }
        __syncthreads();
    }
}

void forward_operations(ann_t *nn)
/* Makes the following operations for forward :
    - z^l = w^l x a^(l-1) + b^l x 1
    - a^l = f(z^l)
*/
{
    matrix_t **w, **a, **z, **b;
    int numLayers = nn->number_of_layers;
    cudaMallocManaged(&w, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&a, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&z, numLayers * sizeof(matrix_t));
    cudaMallocManaged(&b, numLayers * sizeof(matrix_t));
    int maxRows, maxCol;

    for (int i = 0; i < nn->number_of_layers; i++)
    {
        w[i] = nn->layers[i]->weights;
        a[i] = nn->layers[i]->activations;
        z[i] = nn->layers[i]->z;
        b[i] = nn->layers[i]->biases;

        if (nn->layers[i]->activations->rows > maxRows)
        {
            maxRows = nn->layers[i]->activations->rows;
        }
        if (nn->layers[i]->activations->columns > maxCol)
        {
            maxCol = nn->layers[i]->activations->columns;
        }
    }

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)maxCol) / blockDim.x), ceil(((float)maxRows) / blockDim.y));
    forwardGPU<<<gridDim, blockDim>>>(w, a, z, b, numLayers);

    // Synchronize device
    cudaDeviceSynchronize();
}