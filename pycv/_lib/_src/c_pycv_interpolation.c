#include "c_pycv_interpolation.h"

#include <stdlib.h>

// #####################################################################################################################

#define TOLERANCE 1e-15

// #####################################################################################################################

double pycv_interpolate_nn(double *values, double delta)
{
    return *values;
}

double pycv_interpolate_linear(double *values, double delta)
{
    return (1 - delta) * *values + delta * *(values + 1);
}

double pycv_interpolate_quadratic(double *values, double delta)
{
    return *(values + 1) + 0.5 * delta * (*(values + 2) - *values) +
            0.5 * delta * delta * (*(values + 2) - 2 * *(values + 1) + *values);
}

double pycv_interpolate_cubic(double *values, double delta)
{
    return *(values + 1) + 0.5 * delta * (-*values + *(values + 2) +
           delta * (2 * *values - 5 * *(values + 1) + 4 * *(values + 2) - *(values + 3) +
           delta * (-*values + 3 * *(values + 1) - 3 * *(values + 2) + *(values + 3))));
}

double pycv_interpolate(PYCV_InterpOrder order, double *values, double delta)
{
    switch (order) {
        case PYCV_INTERP_NN:
            return pycv_interpolate_nn(values, delta);
        case PYCV_INTERP_LINEAR:
            return pycv_interpolate_linear(values, delta);
        case PYCV_INTERP_QUADRATIC:
            return pycv_interpolate_quadratic(values, delta);
        case PYCV_INTERP_CUBIC:
            return pycv_interpolate_cubic(values, delta);
        default:
            return 0;
    }
}

// #####################################################################################################################

int pycv_InterpTree_init(PYCV_InterpTree *self, int rank, int order)
{
    int ii, jj, *counter = NULL, *shifts = NULL, *strides = NULL;

    self->rank = rank;
    self->order = order;
    self->tree_size = 0;

    counter = calloc(rank * 3, sizeof(int));
    if (!counter) {
        return 0;
    }
    shifts = counter + rank;
    strides = shifts + rank;

    self->tree_size = order + 1;
    *(strides + rank - 1) = 1;

    for (ii = rank - 2; ii >= 0; ii--) {
        self->tree_size *= (order + 1);
        *(strides + ii) = *(strides + ii + 1) * (order + 1);
    }

    self->offsets = malloc(self->tree_size * rank * sizeof(int));
    if (!self->offsets) {
        free(counter);
        self->tree_size = 0;
        return 0;
    }

    for (jj = 0; jj < self->tree_size; jj++) {
        for (ii = 0; ii < rank; ii++) {
            *(self->offsets + jj * rank + ii) = *(shifts + ii);
            *(counter + ii) += 1;
            if (*(counter + ii) == *(strides + ii)) {
                *(counter + ii) = 0;
                if (*(shifts + ii) == order) {
                    *(shifts + ii) = 0;
                } else {
                    *(shifts + ii) += 1;
                }
            }
        }
    }

    free(counter);
    return 1;
}

void pycv_InterpTree_free(PYCV_InterpTree *self)
{
    if (self->tree_size) {
        free(self->offsets);
        self->tree_size = 0;
    }
}

void pycv_InterpTree_interpolate(PYCV_InterpTree *self, double *nodes, double *delta)
{
    int ii, jj, cc, fc_size;
    double t;

    fc_size = self->tree_size / (self->order + 1);
    for (jj = self->rank - 1; jj >= 0; jj--) {
        cc = 0;
        for (ii = 0; ii < fc_size; ii++) {
            t = pycv_interpolate(self->order, nodes + ii * (self->order + 1), *(delta + jj));
            *(nodes + cc) = t;
            cc++;
        }
        fc_size = cc / (self->order + 1);
    }
}

// #####################################################################################################################















