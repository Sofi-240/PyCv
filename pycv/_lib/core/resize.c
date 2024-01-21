#include "ops_base.h"
#include "resize.h"

// #####################################################################################################################

#define RESIZE_BILINEAR_GET_POINT(_coord, _bound, _coord_low, _coord_high)                     \
{                                                                                              \
    _coord_low = (npy_intp)_coord;                                                             \
    if ((double)_coord_low == _coord) {                                                        \
        _coord_high = _coord_low;                                                              \
    } else {                                                                                   \
        _coord_high = (npy_intp)(_coord + 1);                                                  \
    }                                                                                          \
    _coord_high = _coord_high > _bound - 1 ? _bound - 1 : _coord_high;                         \
}

int ops_resize_bilinear(PyArrayObject *input, PyArrayObject *output)
{
    npy_intp nd, from_height, from_width, to_height, to_width, new_size, array_size, ii, count;
    int num_type_i, num_type_o;
    npy_intp i, j, position[NPY_MAXDIMS];
    double y_scale, x_scale, y, x;
    double val1 = 0.0, val2 = 0.0, val3 = 0.0, val4 = 0.0, out = 0.0, q1 = 0.0, q2 = 0.0;
    npy_intp y_low = 0, x_low = 0, y_high = 0, x_high = 0, shift_y = 0, shift_x = 0;
    char *pi_base = NULL, *pi = NULL, *po = NULL;
    ArrayIter iter_i, iter_o;

    NPY_BEGIN_THREADS_DEF;

    nd = PyArray_NDIM(input);
    array_size = PyArray_SIZE(output);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    from_height = PyArray_DIM(input, nd - 2);
    from_width = PyArray_DIM(input, nd - 1);

    to_height = PyArray_DIM(output, nd - 2);
    to_width = PyArray_DIM(output, nd - 1);

    new_size = to_height * to_width;

    y_scale = (double)from_height / (double)to_height;
    x_scale = (double)from_width / (double)to_width;

    count = array_size / new_size;

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);

    NPY_BEGIN_THREADS;

    pi_base = pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    while (count) {
        for (ii = 0; ii < nd - 2; ii++) {
            position[ii] = iter_o.coordinates[ii];
        }
        for (i = 0; i < to_height; i++) {
            y = (double)i * y_scale;
            RESIZE_BILINEAR_GET_POINT(y, from_height, y_low, y_high);
            position[nd - 2] = y_low;

            for (j = 0; j < to_width; j++) {
                x = (double)j * x_scale;
                RESIZE_BILINEAR_GET_POINT(x, from_width, x_low, x_high);
                position[nd - 1] = x_low;
                ARRAY_ITER_GOTO(iter_i, position, pi_base, pi);

                if (y_low == y_high && x_low == x_high) {

                    GET_VALUE_AS(num_type_i, double, pi, out);
                } else if (x_low == x_high) {

                    GET_VALUE_AS(num_type_i, double, pi, val1);

                    shift_y = (y_high - y_low) * iter_i.strides[nd - 2];
                    GET_VALUE_AS(num_type_i, double, (pi + shift_y), val2);

                    out = val1 * ((double)y_high - y) + val2 * (y - (double)y_low);
                } else if (y_low == y_high) {

                    GET_VALUE_AS(num_type_i, double, pi, val1);

                    shift_x = (x_high - x_low) * iter_i.strides[nd - 1];
                    GET_VALUE_AS(num_type_i, double, (pi + shift_x), val2);

                    out = val1 * ((double)x_high - x) + val2 * (x - (double)x_low);
                } else {

                    GET_VALUE_AS(num_type_i, double, pi, val1);

                    shift_y = (y_high - y_low) * iter_i.strides[nd - 2];
                    shift_x = (x_high - x_low) * iter_i.strides[nd - 1];

                    GET_VALUE_AS(num_type_i, double, (pi + shift_x), val2);
                    GET_VALUE_AS(num_type_i, double, (pi + shift_y), val3);

                    GET_VALUE_AS(num_type_i, double, (pi + shift_y + shift_x), val4);

                    q1 = val1 * ((double)x_high - x) + val2 * (x - (double)x_low);
                    q2 = val3 * ((double)x_high - x) + val4 * (x - (double)x_low);

                    out = q1 * ((double)y_high - y) + q2 * (y - (double)y_low);
                }
                SET_VALUE_TO(num_type_o, po, out);
                ARRAY_ITER_NEXT(iter_o, po);
            }
        }
        count--;
    }
    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

int ops_resize_nearest_neighbor(PyArrayObject *input, PyArrayObject *output)
{
    npy_intp nd, from_height, from_width, to_height, to_width, new_size, array_size, count, ii;
    int num_type_i, num_type_o;
    npy_intp i, j, position[NPY_MAXDIMS];
    double y_scale, x_scale, out = 0.0;
    npy_intp proj_y = 0, proj_x = 0, shift = 0;
    char *pi_base = NULL, *pi = NULL, *po = NULL;
    ArrayIter iter_i, iter_o;

    NPY_BEGIN_THREADS_DEF;

    nd = PyArray_NDIM(input);
    array_size = PyArray_SIZE(output);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    from_height = PyArray_DIM(input, nd - 2);
    from_width = PyArray_DIM(input, nd - 1);

    to_height = PyArray_DIM(output, nd - 2);
    to_width = PyArray_DIM(output, nd - 1);

    new_size = to_height * to_width;

    y_scale = (double)from_height / (double)to_height;
    x_scale = (double)from_width / (double)to_width;

    count = array_size / new_size;

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);

    NPY_BEGIN_THREADS;

    pi_base = pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    while (count) {
        for (ii = 0; ii < nd; ii++) {
            position[ii] = iter_o.coordinates[ii];
        }

        for (i = 0; i < to_height; i++) {
            proj_y = (npy_intp)((double)i * y_scale);

            position[nd - 2] = proj_y;
            ARRAY_ITER_GOTO(iter_i, position, pi_base, pi);

            for (j = 0; j < to_width; j++) {

                proj_x = (npy_intp)((double)j * x_scale);

                if (proj_x != position[nd - 1]) {
                    pi += (proj_x - position[nd - 1]) * iter_i.strides[nd - 1];
                    position[nd - 1] = proj_x;
                }

                GET_VALUE_AS(num_type_i, double, pi, out);

                SET_VALUE_TO(num_type_o, po, out);
                ARRAY_ITER_NEXT(iter_o, po);
            }
        }
        count--;
    }
    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################