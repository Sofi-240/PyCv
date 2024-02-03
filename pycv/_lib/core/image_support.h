#ifndef IMAGE_SUPPORT_H
#define IMAGE_SUPPORT_H

// #####################################################################################################################

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output);

// #####################################################################################################################

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

typedef enum {
    DRAW_LINE = 1,
    DRAW_CIRCLE = 2,
    DRAW_ELLIPSE = 3,
} DrawMode;

PyArrayObject *ops_draw_line(npy_intp *point1, npy_intp *point2);

PyArrayObject *ops_draw_circle(npy_intp *center_point, int radius);

PyArrayObject *ops_draw_ellipse(npy_intp *center_point, int a, int b);

// #####################################################################################################################

#endif