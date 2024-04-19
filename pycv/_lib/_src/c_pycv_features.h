#ifndef C_PYCV_FEATURES_H
#define C_PYCV_FEATURES_H

// #####################################################################################################################

int PYCV_glcm(PyArrayObject *gray, PyArrayObject *distances, PyArrayObject *angle, int levels, PyArrayObject **glcm);

int PYCV_corner_FAST(PyArrayObject *input, int ncon, double threshold, PyArrayObject **response);

// #####################################################################################################################


#endif
