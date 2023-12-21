#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define CHECK_ERROR(err)                                                                  \
    {                                                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

matrix_t *alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *res;
    double *m;
    cudaMallocManaged(&res, sizeof(matrix_t));
    cudaMallocManaged(&m, columns * rows * sizeof(double));
    res->m = m;
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    cudaFree(m);
}

void print_matrix(matrix_t *m, bool is_short)
{
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row++)
    {
        for (int col = 0; col < lim_col; col++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns)
            printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows)
        printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

__global__ void computeMatrixSubGPU(
    double *A, double *B, double *C,
    int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numColumns)
    {
        int idx = row * numColumns + col;
        C[idx] = A[idx] - B[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)m2->columns) / blockDim.x), ceil(((float)m1->rows) / blockDim.y));
    computeMatrixSubGPU<<<gridDim, blockDim>>>(m1->m, m2->m, res->m, m1->rows, m1->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}

__global__ void computeMatrixMulGPU(
    double *A, double *B, double *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns)
    {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++)
        {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)m2->columns) / blockDim.x), ceil(((float)m1->rows) / blockDim.y));
    computeMatrixMulGPU<<<gridDim, blockDim>>>(m1->m, m2->m, res->m, m1->rows, m1->columns, m2->rows, m2->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert((m1->columns == res->columns) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

__global__ void transposeMatrixGPU(
    double *A, double *C,
    int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numColumns)
    {
        C[col * numRows + row] = A[row * numColumns + col];
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert((m1->columns == res->rows) &&
           (m1->rows == res->columns));
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)m1->columns) / blockDim.x), ceil(((float)m1->rows) / blockDim.y));
    transposeMatrixGPU<<<gridDim, blockDim>>>(m1->m, res->m, m1->rows, m1->columns);

    // Synchronize Device
    cudaDeviceSynchronize();
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert((m1->rows == res->rows) &&
           (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns * m1->rows; idx++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
}