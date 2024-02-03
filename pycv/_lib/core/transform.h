#ifndef TRANSFORM_H
#define TRANSFORM_H

// #####################################################################################################################

typedef enum {
    HOUGH_LINE = 1,
    HOUGH_CIRCLE = 2,
} HoughMode;

PyArrayObject *ops_hough_line_transform(PyArrayObject *input,
                                        PyArrayObject *theta,
                                        npy_intp offset);

PyArrayObject *ops_hough_circle_transform(PyArrayObject *input,
                                          PyArrayObject *radius,
                                          int normalize,
                                          int expend);
// #####################################################################################################################

#endif