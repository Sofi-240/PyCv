#ifndef FILTERS_H
#define FILTERS_H

// #####################################################################################################################

int ops_convolve(PyArrayObject *input,
                 PyArrayObject *kernel,
                 PyArrayObject *output,
                 npy_intp *origins,
                 BordersMode mode,
                 double constant_value);

int ops_rank_filter(PyArrayObject *input,
                    PyArrayObject *footprint,
                    PyArrayObject *output,
                    int rank,
                    npy_intp *origins,
                    BordersMode mode,
                    double constant_value);

// #####################################################################################################################


#endif