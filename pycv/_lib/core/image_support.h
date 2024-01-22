#ifndef IMAGE_SUPPORT_H
#define IMAGE_SUPPORT_H

// #####################################################################################################################

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output);

int ops_build_max_tree(PyArrayObject *input,
                       PyArrayObject *traverser,
                       PyArrayObject *parent,
                       int connectivity,
                       PyArrayObject *values_map);


int ops_area_threshold(PyArrayObject *input,
                       int connectivity,
                       int threshold,
                       PyArrayObject *output,
                       PyArrayObject *traverser,
                       PyArrayObject *parent);

// #####################################################################################################################


#endif