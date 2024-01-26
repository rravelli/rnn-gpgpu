#include "ann.h"
#include "matrix.h"
#include "forward.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include "backward.h"
#include "forward.h"

double normalRand(double mu, double sigma);
void init_weight(matrix_t *w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

double normalRand(double mu, double sigma)
{
    const double epsilon = DBL_MIN;
    const double two_pi = 2.0 * M_PI;
    bool generate;
    double z1;

    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do
    {
        u1 = (double)rand() / RAND_MAX;
        u2 = (double)rand() / RAND_MAX;
    } while (u1 <= epsilon);

    double z0;
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

void init_weight(matrix_t *w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t *create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned *nneurons_per_layer)
{
    ann_t *nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l - 1], minibatch_size);
    }

    return nn;
}

layer_t *create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t *input)
{
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);

    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn, double (*activation_function)(double))
{
    forward_operations(nn);
}

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{

    // backward_init(nn->layers[L]->activations, y, nn->layers[L]->delta, nn->layers[L]->z);

    backward_recursion(nn, y);

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        backward_assign(nn->layers[l]->weights, nn->layers[l]->delta, nn->layers[l - 1]->activations, nn->layers[l]->biases, nn->alpha, nn->minibatch_size);
    }
}