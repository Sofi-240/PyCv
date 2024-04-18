#ifndef C_PYCV_INTERPOLATION_H
#define C_PYCV_INTERPOLATION_H

// #####################################################################################################################

typedef enum {
    PYCV_INTERP_NN = 0,
    PYCV_INTERP_LINEAR = 1,
    PYCV_INTERP_QUADRATIC = 2,
    PYCV_INTERP_CUBIC = 3,
} PYCV_InterpOrder;

// *********************************************************************************************************************

double pycv_interpolate_nn(double *values, double delta);

#define PYCV_INTERPOLATE_NN(_values, _delta, _out)                                                                     \
{                                                                                                                      \
    _out = *_values;                                                                                                   \
}

// *********************************************************************************************************************

double pycv_interpolate_linear(double *values, double delta);

#define PYCV_INTERPOLATE_LINEAR(_values, _delta, _out)                                                                 \
{                                                                                                                      \
    _out = (1 - _delta) * *_values + _delta * *(_values + 1);                                                          \
}

// *********************************************************************************************************************

double pycv_interpolate_quadratic(double *values, double delta);


#define PYCV_INTERPOLATE_QUADRATIC(_values, _delta, _out)                                                              \
{                                                                                                                      \
    _out = *(_values + 1) + 0.5 * _delta * (*(_values + 2) - *_values) +                                               \
            0.5 * _delta * _delta * (*(_values + 2) - 2 * *(_values + 1) + *_values);                                  \
}

// *********************************************************************************************************************

double pycv_interpolate_cubic(double *values, double delta);

#define PYCV_INTERPOLATE_CUBIC(_values, _delta, _out)                                                                  \
{                                                                                                                      \
    _out = *(_values + 1) + 0.5 * _delta * (-*_values + *(_values + 2) +                                               \
           _delta * (2 * *_values - 5 * *(_values + 1) + 4 * *(_values + 2) - *(_values + 3) +                         \
           _delta * (-*_values + 3 * *(_values + 1) - 3 * *(_values + 2) + *(_values + 3))));                          \
}

// *********************************************************************************************************************

double pycv_interpolate(PYCV_InterpOrder order, double *values, double delta);

#define PYCV_INTERPOLATE(_order, _values, _delta, _out)                                                                \
{                                                                                                                      \
    switch (_order) {                                                                                                  \
        case PYCV_INTERP_NN:                                                                                           \
            PYCV_INTERPOLATE_NN(_values, _delta, _out);                                                                \
            break;                                                                                                     \
        case PYCV_INTERP_LINEAR:                                                                                       \
            PYCV_INTERPOLATE_LINEAR(_values, _delta, _out);                                                            \
            break;                                                                                                     \
        case PYCV_INTERP_QUADRATIC:                                                                                    \
            PYCV_INTERPOLATE_QUADRATIC(_values, _delta, _out);                                                         \
            break;                                                                                                     \
        case PYCV_INTERP_CUBIC:                                                                                        \
            PYCV_INTERPOLATE_CUBIC(_values, _delta, _out);                                                             \
            break;                                                                                                     \
        default:                                                                                                       \
            _out = 0;                                                                                                  \
    }                                                                                                                  \
}

// #####################################################################################################################

typedef struct {
    int rank;
    int order;
    int tree_size;
    int *offsets;
} PYCV_InterpTree;

int pycv_InterpTree_init(PYCV_InterpTree *self, int rank, int order);

void pycv_InterpTree_free(PYCV_InterpTree *self);

void pycv_InterpTree_interpolate(PYCV_InterpTree *self, double *nodes, double *delta);

// #####################################################################################################################

#endif






















