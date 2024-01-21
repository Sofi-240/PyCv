#ifndef IMAGE_SUPPORT_H
#define IMAGE_SUPPORT_H

// #####################################################################################################################

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output);

int ops_build_max_tree(PyArrayObject *input, PyArrayObject *output);

// #####################################################################################################################


#endif