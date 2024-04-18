#ifndef C_PYCV_TRANSFORM_H
#define C_PYCV_TRANSFORM_H

// #####################################################################################################################

int PYCV_resize(PyArrayObject *input,
                PyArrayObject *output,
                npy_intp order,
                int grid_mode,
                PYCV_ExtendBorder mode,
                npy_double c_val);

int PYCV_map_coordinates_case_c(PyArrayObject *matrix, PyArrayObject *dst, PyArrayObject *src);

int PYCV_map_coordinates_case_a(PyArrayObject *matrix, PyArrayObject *dst, PyArrayObject *src,
                                int order, PYCV_ExtendBorder mode, double c_val);

int PYCV_geometric_transform(PyArrayObject *matrix,
                             PyArrayObject *input,
                             PyArrayObject *output,
                             PyArrayObject *dst,
                             PyArrayObject *src,
                             npy_intp order,
                             PYCV_ExtendBorder mode,
                             npy_double c_val);


// #####################################################################################################################

typedef enum {
    PYCV_HOUGH_LINE = 1,
    PYCV_HOUGH_CIRCLE = 2,
    PYCV_HOUGH_LINE_PROBABILISTIC = 3
} PYCV_HoughMode;

PyArrayObject *PYCV_hough_line_transform(PyArrayObject *input,
                                         PyArrayObject *theta,
                                         npy_intp offset);

PyArrayObject *PYCV_hough_circle_transform(PyArrayObject *input,
                                           PyArrayObject *radius,
                                           int normalize,
                                           int expend);

PyArrayObject *PYCV_hough_probabilistic_line(PyArrayObject *input,
                                             PyArrayObject *theta,
                                             npy_intp offset,
                                             npy_intp threshold,
                                             npy_intp line_length,
                                             npy_intp line_gap);

// #####################################################################################################################

int PYCV_integral_image(PyArrayObject *inputs, PyArrayObject **output);

// #####################################################################################################################

int PYCV_linear_interp1D(PyArrayObject *xn, PyArrayObject *xp, PyArrayObject *fp, double l, double h, PyArrayObject **fn);

// #####################################################################################################################

#endif