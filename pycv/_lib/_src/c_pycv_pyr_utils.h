#ifndef C_PYCV_PYR_UTILS_H
#define C_PYCV_PYR_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "c_pycv_interpolation.h"

// ############################################ UTILS ##################################################################

#define UTILS_MALLOC(_size, _sizeof, _arr)                                                                             \
{                                                                                                                      \
    _arr = malloc(_size * _sizeof);                                                                                    \
    if (!_arr) {                                                                                                       \
        _size = 0;                                                                                                     \
    }                                                                                                                  \
}

#define UTILS_REALLOC(_size, _sizeof, _arr)                                                                            \
{                                                                                                                      \
    _arr = realloc(_arr, _sizeof * _size);                                                                             \
    if (!_arr) {                                                                                                       \
        _size = 0;                                                                                                     \
    }                                                                                                                  \
}

#define UTILS_PRINT_LIST(_list, _size, _is_double)                                                                     \
{                                                                                                                      \
    printf("[");                                                                                                       \
    for (int _ii = 0; _ii < _size; _ii++) {                                                                            \
        if (_is_double)                                                                                                \
            printf("%.2f, ", (double)*(_list + _ii));                                                                  \
        else                                                                                                           \
            printf("%d, ", (int)*(_list + _ii));                                                                       \
    }                                                                                                                  \
    printf("]\n");                                                                                                     \
}

#define UTILS_DEFAULT_SETTING -1

#define UTILS_NODE_GOTO_LAST(_node)                                                                                    \
{                                                                                                                      \
    while (_node->next != NULL)                                                                                        \
        _node = _node->next;                                                                                           \
}

#define UTILS_NODE_GOTO_LAST_M1(_node)                                                                                 \
{                                                                                                                      \
    while (_node->next != NULL && _node->next->next != NULL)                                                           \
        _node = _node->next;                                                                                           \
}

#define UTILS_NODE_REVERS(_head, _dtype)                                                                               \
{                                                                                                                      \
    if (*_head != NULL && (*_head)->next != NULL) {                                                                    \
        _dtype *_next = (*_head)->next, *_prev = *_head, *_tmp = NULL;                                                 \
        _prev->next = NULL;                                                                                            \
        while (_next != NULL) {                                                                                        \
            _tmp = _next->next;                                                                                        \
            _next->next = _prev;                                                                                       \
            _prev = _next;                                                                                             \
            _next = _tmp;                                                                                              \
        }                                                                                                              \
        *_head = _prev;                                                                                                \
    }                                                                                                                  \
}

// ######################################### GAUSSIAN ##################################################################

#define GAUSSIAN_EPSILON 2e-16

#define GAUSSIAN_DEFAULT_TRUNCATE 3.0

#define GAUSSIAN_DEFAULT_RADIUS -1

#define GAUSSIAN_RADIUS_FROM_SIGMA(_sigma, _truncate) (int)floor(_sigma * _truncate + 0.5)

#define GAUSSIAN_KERNEL_SIZE(_radius, _ndim) (int)pow((double)(2 * _radius + 1), (double)_ndim)

#define GAUSSIAN_SIGMA_FROM_SIZE(_size) (double)(0.3 * (((double)_size / 2) - 1) + 0.8)

#define GAUSSIAN_DEFAULT_SIGMA 1.0

int gaussian_kernel(double sigma, int ndim, int radius, double **kernel);

// ######################################### OFFSETS ###################################################################

typedef enum {
    EXTEND_REFLECT = 3,
    EXTEND_CONSTANT = 4,
    EXTEND_SYMMETRIC = 5,
    EXTEND_WRAP = 6,
    EXTEND_EDGE = 7,
} PyrExtend;

int offsets_fit_coordinate(int coordinate, int dim, int flag, int mode);

void offsets_init(int input_dim, int input_stride, int offset_dim, int mode, int *flag, int *offsets);

// *********************************************************************************************************************

typedef struct {
    int dim;
    int stride;
    int stride_back;
    int low;
    int high;
    int init_pos;
    int nn_stride;
    int flag;
    int *offsets;
} Offsets1D;

void Offsets1D_adapt_dim(Offsets1D *self, int input_dim);

int Offsets1D_new(Offsets1D **self, int input_dim, int input_stride, int dim, int mode);

void Offsets1D_free(Offsets1D **self);

int Offsets1D_update_offsets(Offsets1D **self, int input_dim, int input_stride, int dim, int mode);

void Offsets1D_update_input_stride(Offsets1D *self, int input_stride, int prev_stride);

void Offsets1D_print(Offsets1D **self);

#define OFFSETS_GOTO(_offsets, _ptr, _to)                                                                              \
{                                                                                                                      \
    _ptr = _offsets->offsets + _offsets->init_pos;                                                                     \
    if (_to < _offsets->low)                                                                                           \
        _ptr += _offsets->stride + _to;                                                                                \
    else if (_to == _offsets->high)                                                                                    \
        _ptr += (_offsets->stride * (_offsets->low - 2)) + 2 * _offsets->nn_stride;                                    \
    else if (_to > _offsets->high)                                                                                     \
        _ptr += (_offsets->stride * (_to + _offsets->low - 2 - _offsets->high)) + 2 * _offsets->nn_stride;             \
    else                                                                                                               \
        _ptr += (_offsets->stride * (_offsets->low - 1)) + _offsets->nn_stride;                                        \
}

// ######################################## Iterator ###################################################################

// **************************************** Iterator 1D ****************************************************************

typedef struct Iterator1D {
    int axis;
    int coordinate;
    int dim;
    int stride;
    int stride_back;
    Offsets1D *offsets;
    struct Iterator1D *next;
} Iterator1D;

int Iterator1D_new(Iterator1D **self, int axis, int dim, int stride);

void Iterator1D_free(Iterator1D **self);

void Iterator1D_update_stride(Iterator1D *self, int stride);

void Iterator1D_update_dim(Iterator1D *self, int dim, int stride, int mode);

int Iterator1D_update_offset_dim(Iterator1D *self, int dim, int mode);

void Iterator1D_print(Iterator1D **self);

// **************************************** Iterator ND ****************************************************************

int IteratorND_copy_to(Iterator1D **self, Iterator1D **to);

void IteratorND_next_axis(Iterator1D **head);

void IteratorND_reset_axis(Iterator1D **head);

#define IteratorND_offsets_next2(_node, _ptr1, _ptr2, _ofs)                                                            \
{                                                                                                                      \
    while (_node != NULL) {                                                                                            \
        if (_node->coordinate < _node->dim - 1) {                                                                      \
            if (_node->next == NULL) {                                                                                 \
                if (_node->coordinate == _node->offsets->low - 1 || _node->coordinate == _node->offsets->high) {       \
                    _ofs += _node->offsets->nn_stride;                                                                 \
                } else if (_node->coordinate < _node->offsets->low - 1 || _node->coordinate > _node->offsets->high) {  \
                    _ofs += _node->offsets->stride;                                                                    \
                }                                                                                                      \
            }                                                                                                          \
            _node->coordinate += 1;                                                                                    \
            _ptr1 += _node->stride;                                                                                    \
            _ptr2 += _node->stride;                                                                                    \
            break;                                                                                                     \
        } else {                                                                                                       \
            _node->coordinate = 0;                                                                                     \
            _ptr1 -= _node->stride_back;                                                                               \
            _ptr2 -= _node->stride_back;                                                                               \
            _node = _node->next;                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
}

// ####################################### Kernels Mem #################################################################

typedef struct GaussianMem {
    double sigma;
    int len;
    double *h;
    struct GaussianMem *next;
    int entries;
} GaussianMem;

int GaussianMem_new(GaussianMem **self, double sigma);

void GaussianMem_free(GaussianMem **self);

void GaussianMem_push(GaussianMem **head, double sigma, GaussianMem **out);

void GaussianMem_pop(GaussianMem **head, double sigma);

void GaussianMem_print(GaussianMem **head);

// ####################################### Gaussian 1D #################################################################

typedef struct Gaussian1D {
    int axis;
    GaussianMem *kernel;
    struct Gaussian1D *next;
} Gaussian1D;

int Gaussian1D_new(Gaussian1D **self, int axis, double sigma, GaussianMem **mem);

void Gaussian1D_free(Gaussian1D **self, GaussianMem **mem);

void Gaussian1D_print(Gaussian1D **head);

int Gaussian1D_build_scalespace(Gaussian1D **root, GaussianMem **mem, int ndim, int nscales, double *scalespace);

// ######################################## Rescale 1D #################################################################

typedef struct Rescale1D {
    int axis;
    double factor;
    GaussianMem *kernel;
    struct Rescale1D *next;
} Rescale1D;

int Rescale1D_new(Rescale1D **self, int axis, double factor, int order, GaussianMem **mem);

void Rescale1D_free(Rescale1D **self, GaussianMem **mem);

int Rescale1D_update_order(Rescale1D **self, int order, GaussianMem **mem);

// *********************************************************************************************************************

typedef struct {
    Iterator1D *input;
    Iterator1D *output;
    int extend_mode;
} RescaleIterator;

int RescaleIterator_new(RescaleIterator **self, Iterator1D **base, int extend_mode);

void RescaleIterator_free(RescaleIterator **self);

void RescaleIterator_update_output(RescaleIterator *self, int dim);

void RescaleIterator_update_input(RescaleIterator *self);

void RescaleIterator_update_base(RescaleIterator *self, Iterator1D *base);

// ######################################## Layer ######################################################################

typedef struct {
    int ndim;
    int nscales;
    int numtype;
    int order;
    int itemsize;
    int extend_mode;
    double cval;
    GaussianMem *mem;
    Gaussian1D *scalespace;
    Rescale1D *factors;
    Iterator1D *iterator;
} Layer;

void Layer_free(Layer **self);

int Layer_new(Layer **self, int ndim, int itemsize, int numtype);

// ********************************************* setter ****************************************************************

void Layer_set_itemsize(Layer *self, int itemsize, int numtype);

void Layer_set_extend_mode(Layer *self, int extend_mode);

int Layer_set_scalespace(Layer *self, int nscales, double *scalespace);

int Layer_set_factors(Layer *self, double *factors);

int Layer_set_order(Layer *self, int order);

int Layer_set_input_dim(Layer *self, int *dims);

int Layer_update_offsets(Layer *self);

void Layer_reduce(Layer *self);


// #####################################################################################################################


#endif