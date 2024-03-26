#ifndef C_PYCV_MORPHOLOGY_H
#define C_PYCV_MORPHOLOGY_H

// #####################################################################################################################

int PYCV_binary_erosion(PyArrayObject *input,
                        PyArrayObject *strel,
                        PyArrayObject *output,
                        npy_intp *center,
                        int iterations,
                        PyArrayObject *mask,
                        int invert,
                        int c_val);

// #####################################################################################################################

int PYCV_gray_erosion(PyArrayObject *input,
                      PyArrayObject *strel,
                      PyArrayObject *output,
                      npy_intp *center,
                      PyArrayObject *mask,
                      int invert,
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