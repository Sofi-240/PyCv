#ifndef IMAGE_SUPPORT_H
#define IMAGE_SUPPORT_H

// #####################################################################################################################

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output);

int ops_canny_hysteresis_edge_tracking(PyArrayObject *strong_edge, PyArrayObject *week_edge, PyArrayObject *strel);

// #####################################################################################################################


#endif