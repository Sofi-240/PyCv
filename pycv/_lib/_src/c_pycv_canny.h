#ifndef C_PYCV_CANNY_H
#define C_PYCV_CANNY_H

// #####################################################################################################################

int PYCV_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                      PyArrayObject *grad_y,
                                      PyArrayObject *grad_x,
                                      npy_double threshold,
                                      PyArrayObject *mask,
                                      PyArrayObject *output);

// #####################################################################################################################


#endif