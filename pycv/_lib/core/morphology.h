#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

typedef enum {
    ERO = 0,
    DIL = 1,
} ERO_OR_DIL_OP;

typedef enum {
    LABEL_BY_NLABELS = 1,
    LABEL_BY_INDEX = 2,
} LabelMode;

// #####################################################################################################################

int ops_binary_erosion(PyArrayObject *input,
                       PyArrayObject *strel,
                       PyArrayObject *output,
                       npy_intp *origins,
                       int iterations,
                       PyArrayObject *mask,
                       int invert);

int ops_gray_ero_or_dil(PyArrayObject *input,
                        PyArrayObject *flat_strel,
                        PyArrayObject *non_flat_strel,
                        PyArrayObject *output,
                        npy_intp *origins,
                        PyArrayObject *mask,
                        double cast_value,
                        ERO_OR_DIL_OP op);

int ops_binary_region_fill(PyArrayObject *output,
                           PyArrayObject *strel,
                           npy_intp *seed_point,
                           npy_intp *origins);

int ops_labeling(PyArrayObject *input,
                 int connectivity,
                 PyArrayObject *values_map,
                 PyArrayObject *output,
                 LabelMode label_mode);

// #####################################################################################################################

int ops_skeletonize(PyArrayObject *input, PyArrayObject *output);

// #####################################################################################################################


#endif