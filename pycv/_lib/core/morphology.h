#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

int ops_binary_erosion(PyArrayObject *input,
                       PyArrayObject *strel,
                       PyArrayObject *output,
                       npy_intp *origins,
                       int iterations,
                       PyArrayObject *mask,
                       int invert);

int ops_erosion(PyArrayObject *input,
                PyArrayObject *flat_strel,
                PyArrayObject *non_flat_strel,
                PyArrayObject *output,
                npy_intp *origins,
                PyArrayObject *mask,
                double cast_value);

int ops_dilation(PyArrayObject *input,
                 PyArrayObject *flat_strel,
                 PyArrayObject *non_flat_strel,
                 PyArrayObject *output,
                 npy_intp *origins,
                 PyArrayObject *mask,
                 double cast_value);

int ops_binary_region_fill(PyArrayObject *output,
                           PyArrayObject *strel,
                           npy_intp *seed_point,
                           npy_intp *origins);

#endif