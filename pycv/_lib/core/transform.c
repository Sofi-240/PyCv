#include "ops_base.h"
#include "transform.h"

// #####################################################################################################################

#define TYPE_CASE_HOUGH_ADD_VALUE(_NUM_TYPE, _dtype, _pointer, _val)                                                   \
case _NUM_TYPE:                                                                                                        \
    *(_dtype *)_pointer += (_dtype)_val;                                                                               \
    break

#define CASE_HOUGH_ADD_VALUE(_NUM_TYPE, _pointer, _val)                                                                \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_BOOL, npy_bool, _pointer, _val);                                                 \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_UBYTE, npy_ubyte, _pointer, _val);                                               \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_USHORT, npy_ushort, _pointer, _val);                                             \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_UINT, npy_uint, _pointer, _val);                                                 \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_ULONG, npy_ulong, _pointer, _val);                                               \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_ULONGLONG, npy_ulonglong, _pointer, _val);                                       \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_BYTE, npy_byte, _pointer, _val);                                                 \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_SHORT, npy_short, _pointer, _val);                                               \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_INT, npy_int, _pointer, _val);                                                   \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_LONG, npy_long, _pointer, _val);                                                 \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_LONGLONG, npy_longlong, _pointer, _val);                                         \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_FLOAT, npy_float, _pointer, _val);                                               \
        TYPE_CASE_HOUGH_ADD_VALUE(NPY_DOUBLE, npy_double, _pointer, _val);                                             \
    }                                                                                                                  \
}

PyArrayObject *ops_hough_line_transform(PyArrayObject *input,
                                        PyArrayObject *theta,
                                        npy_intp offset)
{
    int num_type_i, num_type_t, num_type_h = NPY_UINT64;
    ArrayIter iter_i, iter_t, iter_h;
    char *pi = NULL, *ph_base = NULL, *ph = NULL, *pt = NULL;
    npy_intp nd, channel_size = 1, input_shape[NPY_MAXDIMS], ndt_init;

    PyArrayObject *h_space;
    npy_intp *h_shape, h_size = 1, position[NPY_MAXDIMS], n_theta, n_rho;
    double *cosine, *sine, angle, i_val, y, x, proj;

    npy_intp ii, jj, hh;

    NPY_BEGIN_THREADS_DEF;

    nd = PyArray_NDIM(input);
    ndt_init = nd - 2;
    num_type_i = PyArray_TYPE(input);

    h_shape = malloc(nd * sizeof(double));

    if (!h_shape) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < nd; ii++) {
        position[ii] = 0;
        input_shape[ii] = PyArray_DIM(input, ii);
        if (ii < ndt_init) {
            channel_size *= input_shape[ii];
            h_shape[ii] = input_shape[ii];
        } else {
            h_size *= input_shape[ii];
        }
    }

    n_theta = PyArray_SIZE(theta);
    num_type_t = PyArray_TYPE(theta);

    cosine = malloc(n_theta * sizeof(double));
    sine = malloc(n_theta * sizeof(double));

    if (!cosine || !sine) {
        PyErr_NoMemory();
        goto exit;
    }

    pt = (void *)PyArray_DATA(theta);
    ArrayIterInit(theta, &iter_t);

    for (ii = 0; ii < n_theta; ii++) {
        GET_VALUE_AS(num_type_t, double, pt, angle);
        cosine[ii] = cos(angle);
        sine[ii] = sin(angle);
        ARRAY_ITER_NEXT(iter_t, pt);
    }

    n_rho = 2 * offset + 1;

    h_shape[ndt_init] = n_rho;
    h_shape[ndt_init + 1] = n_theta;

    h_space = (PyArrayObject *)PyArray_ZEROS(nd, h_shape, num_type_h, 0);

    ArrayIterInit(h_space, &iter_h);
    ArrayIterInit(input, &iter_i);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    ph_base = ph = (void *)PyArray_DATA(h_space);

    while (channel_size--) {
        for (ii = 0; ii < h_size; ii++) {
            y = (double)iter_i.coordinates[ndt_init];
            x = (double)iter_i.coordinates[ndt_init + 1];
            GET_VALUE_AS(num_type_i, double, pi, i_val);

            if (fabs(i_val) > DBL_EPSILON) {
                for (jj = 0; jj < n_theta; jj++) {
                    proj = cosine[jj] * x + sine[jj] * y;

                    hh = (npy_intp)floor(proj + 0.5) + offset;
                    hh = hh * iter_h.strides[ndt_init] + jj * iter_h.strides[ndt_init + 1];

                    CASE_HOUGH_ADD_VALUE(num_type_h, (ph + hh), 1);
                }
            }
            ARRAY_ITER_NEXT(iter_i, pi);
        }
        for (ii = 0; ii < ndt_init; ii++) {
            position[ii] = iter_i.coordinates[ii];
        }
        ARRAY_ITER_GOTO(iter_h, position, ph_base, ph);
    }

    NPY_END_THREADS;

    exit:
        free(h_shape);
        free(cosine);
        free(sine);
        return PyErr_Occurred() ? NULL : h_space;
}


#define HOUGH_GET_CIRCLE_POINTS(_radius, _circle, _size)                                                               \
{                                                                                                                      \
    npy_intp _rr = 0, _sh = 3 - 2 * _radius, _xx = _radius, _yy = 0;                                                   \
    _size = 0;                                                                                                         \
    while (_xx >= _yy) {                                                                                               \
        _circle[_rr++] = _yy;                                                                                          \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = _yy;                                                                                          \
        _circle[_rr++] = _yy;                                                                                          \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = _yy;                                                                                          \
        _size += 8;                                                                                                    \
        if (_sh < 0) {                                                                                                 \
            _sh += 4 * _yy + 6;                                                                                        \
        } else {                                                                                                       \
            _sh += 4 * (_yy - _xx) + 10;                                                                               \
            _xx -= 1;                                                                                                  \
        }                                                                                                              \
        _yy += 1;                                                                                                      \
    }                                                                                                                  \
}


PyArrayObject *ops_hough_circle_transform(PyArrayObject *input,
                                          PyArrayObject *radius,
                                          int normalize,
                                          int expend)
{
    int num_type_i, num_type_r, num_type_h;
    ArrayIter iter_i, iter_r, iter_h;
    char *pi = NULL, *ph_base = NULL, *ph = NULL, *pr = NULL, *pr_base = NULL;
    npy_intp nd, channel_size = 1, input_shape[NPY_MAXDIMS], ndt_init;

    PyArrayObject *h_space;
    npy_intp *h_shape, h_size = 1, position[NPY_MAXDIMS], n_radius, shift = 0;
    npy_intp **circle_points, circle_size, max_size, max_r = 0, r, y, x, proj_y, proj_x;
    double i_val, incr_val;

    npy_intp ii, jj, hh, hhp, rr, kk;

    NPY_BEGIN_THREADS_DEF;

    nd = PyArray_NDIM(input);
    ndt_init = nd - 2;
    num_type_i = PyArray_TYPE(input);

    h_shape = malloc((nd + 1) * sizeof(double));

    if (!h_shape) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < nd; ii++) {
        position[ii] = 0;
        input_shape[ii] = PyArray_DIM(input, ii);
        if (ii < ndt_init) {
            channel_size *= input_shape[ii];
            h_shape[ii] = input_shape[ii];
        } else {
            h_size *= input_shape[ii];
        }
    }
    position[nd] = 0;

    num_type_h = normalize ? NPY_DOUBLE : NPY_UINT64;

    n_radius = PyArray_SIZE(radius);
    num_type_r = PyArray_TYPE(radius);

    pr_base = pr = (void *)PyArray_DATA(radius);
    ArrayIterInit(radius, &iter_r);

    for (ii = 0; ii < n_radius; ii++) {
        GET_VALUE_AS(num_type_r, npy_intp, pr, r);
        max_r = r > max_r ? r : max_r;
        ARRAY_ITER_NEXT(iter_r, pr);
    }
    pr = pr_base;
    ARRAY_ITER_RESET(iter_r);

    if (expend) {
        shift = max_r;
    }

    h_shape[ndt_init] = n_radius;
    h_shape[ndt_init + 1] = input_shape[ndt_init] + 2 * shift;
    h_shape[ndt_init + 2] = input_shape[ndt_init + 1] + 2 * shift;

    h_space = (PyArrayObject *)PyArray_ZEROS(nd + 1, h_shape, num_type_h, 0);
    if (!h_space) {
        PyErr_NoMemory();
        goto exit;
    }

    circle_points = malloc(n_radius * sizeof(npy_intp*));

    if (!circle_points) {
        PyErr_NoMemory();
        goto exit;
    }
    for (ii = 0; ii < n_radius; ii++) {
        circle_points[ii] = NULL;
    }

    for (ii = 0; ii < n_radius; ii++) {
        GET_VALUE_AS(num_type_r, npy_intp, pr, r);
        max_size = 2 * ((r * 8) + 8);
        circle_points[ii] = malloc((max_size + 1) * sizeof(npy_intp));
        if (!circle_points[ii]) {
            PyErr_NoMemory();
            goto exit;
        }
        HOUGH_GET_CIRCLE_POINTS(r, circle_points[ii], circle_size);
        circle_points[ii][max_size] = circle_size;
        ARRAY_ITER_NEXT(iter_r, pr);
    }
    pr = pr_base;
    ARRAY_ITER_RESET(iter_r);

    ArrayIterInit(h_space, &iter_h);
    ArrayIterInit(input, &iter_i);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    ph_base = ph = (void *)PyArray_DATA(h_space);


    while (channel_size--) {

        for (ii = 0; ii < h_size; ii++) {
            y = iter_i.coordinates[ndt_init];
            x = iter_i.coordinates[ndt_init + 1];
            GET_VALUE_AS(num_type_i, double, pi, i_val);

            if (fabs(i_val) > DBL_EPSILON) {
                y += shift;
                x += shift;

                for (rr = 0; rr < n_radius; rr++) {
                    GET_VALUE_AS(num_type_r, npy_intp, pr, r);
                    max_size = 2 * ((r * 8) + 8);
                    circle_size = circle_points[rr][max_size];

                    incr_val = normalize ? 1 / (double)circle_size : 1;
                    hh = rr * iter_h.strides[ndt_init];
                    kk = 0;
                    for (jj = 0; jj < circle_size; jj++) {
                        proj_y = y + circle_points[rr][kk];
                        proj_x = x + circle_points[rr][kk + 1];
                        kk += 2;

                        if (shift || (proj_y >= 0 && proj_y < h_shape[ndt_init + 1] && proj_x >= 0 && proj_x < h_shape[ndt_init + 2])) {
                            hhp = hh + proj_y * iter_h.strides[ndt_init + 1] + proj_x * iter_h.strides[ndt_init + 2];
                            CASE_HOUGH_ADD_VALUE(num_type_h, (ph + hhp), incr_val);
                        }
                    }
                    ARRAY_ITER_NEXT(iter_r, pr);
                }
                pr = pr_base;
                ARRAY_ITER_RESET(iter_r);
            }

            ARRAY_ITER_NEXT(iter_i, pi);
        }

        for (ii = 0; ii < ndt_init; ii++) {
            position[ii] = iter_i.coordinates[ii];
        }
        ARRAY_ITER_GOTO(iter_h, position, ph_base, ph);
    }

    NPY_END_THREADS;
    exit:
        free(h_shape);
        for (ii = 0; ii < n_radius; ii++) {
            free(circle_points[ii]);
        }
        free(circle_points);
        return PyErr_Occurred() ? NULL : h_space;

}

// #####################################################################################################################

