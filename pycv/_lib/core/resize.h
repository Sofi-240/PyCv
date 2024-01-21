#ifndef RESIZE_H
#define RESIZE_H

// #####################################################################################################################

typedef enum {
    RESIZE_BILINEAR = 1,
    RESIZE_NN = 2,
} ResizeMode;

int ops_resize_bilinear(PyArrayObject *input, PyArrayObject *output);

int ops_resize_nearest_neighbor(PyArrayObject *input, PyArrayObject *output);

// #####################################################################################################################

#endif