#ifndef C_PYCV_DRAW_H
#define C_PYCV_DRAW_H

typedef enum {
    PYCV_DRAW_LINE = 1,
    PYCV_DRAW_CIRCLE = 2,
    PYCV_DRAW_ELLIPSE = 3,
} PYCV_DrawMode;

// #####################################################################################################################

PyArrayObject *PYCV_draw_line(npy_intp *point1, npy_intp *point2);

PyArrayObject *PYCV_draw_circle(npy_intp *center_point, npy_intp radius);

PyArrayObject *PYCV_draw_ellipse(npy_intp *center_point, npy_intp a, npy_intp b);

// #####################################################################################################################


#endif