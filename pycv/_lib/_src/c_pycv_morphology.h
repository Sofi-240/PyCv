#ifndef C_PYCV_MORPHOLOGY_H
#define C_PYCV_MORPHOLOGY_H

// #####################################################################################################################

typedef enum {
    PYCV_MORPH_OP_ERO = 0,
    PYCV_MORPH_OP_DIL = 1,
} PYCV_MorphOP;

// #####################################################################################################################

int PYCV_binary_erosion(PyArrayObject *input,
                        PyArrayObject *strel,
                        PyArrayObject *output,
                        npy_intp *center,
                        PyArrayObject *mask,
                        PYCV_MorphOP op,
                        int c_val);

int PYCV_binary_erosion_iter(PyArrayObject *input,
                             PyArrayObject *strel,
                             PyArrayObject *output,
                             npy_intp *center,
                             npy_intp iterations,
                             PyArrayObject *mask,
                             PYCV_MorphOP op,
                             int c_val);

// #####################################################################################################################

int PYCV_gray_erosion_dilation(PyArrayObject *input,
                               PyArrayObject *flat_strel,
                               PyArrayObject *non_flat_strel,
                               PyArrayObject *output,
                               npy_intp *center,
                               PyArrayObject *mask,
                               PYCV_MorphOP op,
                               npy_double c_val);

// #####################################################################################################################

int PYCV_binary_region_fill(PyArrayObject *output,
                            npy_intp *seed_point,
                            PyArrayObject *strel,
                            npy_intp *center);

// #####################################################################################################################

int PYCV_labeling(PyArrayObject *input,
                  npy_intp connectivity,
                  PyArrayObject *output);

// #####################################################################################################################

PyArrayObject *PYCV_skeletonize(PyArrayObject *input);

// #####################################################################################################################

#endif