#ifndef C_PYCV_TRANSFORM_H
#define C_PYCV_TRANSFORM_H

// #####################################################################################################################

typedef struct {
    npy_intp a_dims[NPY_MAXDIMS];
    npy_intp a_strides[NPY_MAXDIMS];
    npy_intp flag;
    npy_intp rank;
    npy_intp order;
    npy_intp c_size;
    npy_intp c_strides[NPY_MAXDIMS];
    npy_intp c_counter[NPY_MAXDIMS];
    npy_intp c_shifts[NPY_MAXDIMS];
    npy_double *coefficients;
} InterpolationAuxObject;

int PYCV_InterpolationAuxObjectInit(npy_intp order, PyArrayObject *array, npy_intp ndim0, InterpolationAuxObject *object);

// #####################################################################################################################

int PYCV_resize(PyArrayObject *input,
                PyArrayObject *output,
                npy_intp order,
                int grid_mode,
                PYCV_ExtendBorder mode,
                npy_double c_val);

int PYCV_geometric_transform(PyArrayObject *matrix,
                             PyArrayObject *input,
                             PyArrayObject *output,
                             PyArrayObject *src,
                             PyArrayObject *dst,
                             npy_intp order,
                             PYCV_ExtendBorder mode,
                             npy_double c_val);


// #####################################################################################################################

typedef enum {
    PYCV_HOUGH_LINE = 1,
    PYCV_HOUGH_CIRCLE = 2,
} PYCV_HoughMode;

PyArrayObject *PYCV_hough_line_transform(PyArrayObject *input,
                                         PyArrayObject *theta,
                                         npy_intp offset);

PyArrayObject *PYCV_hough_circle_transform(PyArrayObject *input,
                                           PyArrayObject *radius,
                                           int normalize,
                                           int expend);

// #####################################################################################################################

#endif