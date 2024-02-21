#include "c_pycv_base.h"
#include "c_pycv_draw.h"

// #####################################################################################################################

#define PYCV_D_CASE_SET_POINT(_NTYPE, _dtype, _stride, _ndim, _pointer, _point)                                        \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii < _ndim; _ii++) {                                                                                \
        *(_dtype *)_pointer = (_dtype)_point[_ii];                                                                     \
        _pointer += _stride;                                                                                           \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_T_SET_POINT(_NTYPE, _stride, _ndim, _pointer, _point)                                                     \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_D_CASE_SET_POINT(BOOL, npy_bool, _stride, _ndim, _pointer, _point);                                       \
        PYCV_D_CASE_SET_POINT(UBYTE, npy_ubyte, _stride, _ndim, _pointer, _point);                                     \
        PYCV_D_CASE_SET_POINT(USHORT, npy_ushort, _stride, _ndim, _pointer, _point);                                   \
        PYCV_D_CASE_SET_POINT(UINT, npy_uint, _stride, _ndim, _pointer, _point);                                       \
        PYCV_D_CASE_SET_POINT(ULONG, npy_ulong, _stride, _ndim, _pointer, _point);                                     \
        PYCV_D_CASE_SET_POINT(ULONGLONG, npy_ulonglong, _stride, _ndim, _pointer, _point);                             \
        PYCV_D_CASE_SET_POINT(BYTE, npy_byte, _stride, _ndim, _pointer, _point);                                       \
        PYCV_D_CASE_SET_POINT(SHORT, npy_short, _stride, _ndim, _pointer, _point);                                     \
        PYCV_D_CASE_SET_POINT(INT, npy_int, _stride, _ndim, _pointer, _point);                                         \
        PYCV_D_CASE_SET_POINT(LONG, npy_long, _stride, _ndim, _pointer, _point);                                       \
        PYCV_D_CASE_SET_POINT(LONGLONG, npy_longlong, _stride, _ndim, _pointer, _point);                               \
        PYCV_D_CASE_SET_POINT(FLOAT, npy_float, _stride, _ndim, _pointer, _point);                                     \
        PYCV_D_CASE_SET_POINT(DOUBLE, npy_double, _stride, _ndim, _pointer, _point);                                   \
    }                                                                                                                  \
}

#define PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(_object, _yy, _xx)                                                       \
{                                                                                                                      \
    (_object).coordinates[(_object).coordinates_size][0] = _yy;                                                        \
    (_object).coordinates[(_object).coordinates_size][1] = _xx;                                                        \
    (_object).coordinates_size++;                                                                                      \
}

#define PYCV_D_SWAP_ARGS(_a1, _a2)                                                                                     \
{                                                                                                                      \
    npy_intp _tmp = _a1;                                                                                               \
    _a1 = _a2;                                                                                                         \
    _a2 = _tmp;                                                                                                        \
}

// #####################################################################################################################

PyArrayObject *PYCV_draw_line(npy_intp *point1, npy_intp *point2)
{
    npy_intp size, y1, y2, x1, x2, dy, dx, step_y, step_x, p, yy, xx, m, ii;
    npy_intp p_dims[2], stride, point[NPY_MAXDIMS];
    int flag, num_type_h;

    PyArrayObject *yx;
    char *pyx = NULL;

    NPY_BEGIN_THREADS_DEF;

    y1 = point1[0];
    x1 = point1[1];

    y2 = point2[0];
    x2 = point2[1];

    dy = y2 - y1 < 0 ? -(y2 - y1) : y2 - y1;
    dx = x2 - x1 < 0 ? -(x2 - x1) : x2 - x1;

    size = dx > dy ? dx + 1 : dy + 1;

    p_dims[0] = size;
    p_dims[1] = 2;

    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }
    num_type_h = PyArray_TYPE(yx);
    stride = PyArray_STRIDE(yx, 1);

    step_y = y2 - y1 > 0 ? 1 : -1;
    step_x = x2 - x1 > 0 ? 1 : -1;

    flag = dy > dx ? 1 : 0;

    if (flag) {
        PYCV_D_SWAP_ARGS(x1, y1);
        PYCV_D_SWAP_ARGS(dx, dy);
        PYCV_D_SWAP_ARGS(step_x, step_y);
    }

    p = 2 * dy - dx;
    yy = y1;
    xx = x1;

    NPY_BEGIN_THREADS;

    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii <= dx; ii++) {
        if (flag) {
            point[0] = xx;
            point[1] = yy;
        } else {
            point[0] = yy;
            point[1] = xx;
        }

        PYCV_T_SET_POINT(num_type_h, stride, 2, pyx, point);

        xx += step_x;
        m = p >= 0 ? 1 : 0;

        p += (2 * dy) - (2 * dx) * m;
        yy += step_y * m;
    }

    NPY_END_THREADS;

    exit:
        return PyErr_Occurred() ? NULL : yx;
}

// #####################################################################################################################


#define PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS4(_object, _yy, _xx)                                                  \
{                                                                                                                      \
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(_object, -_yy, _xx);                                                         \
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(_object, _yy, _xx);                                                          \
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(_object, -_yy, -_xx);                                                        \
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(_object, _yy, -_xx);                                                         \
}

#define PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS8(_object, _yy, _xx)                                                  \
{                                                                                                                      \
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS4(_object, _yy, _xx);                                                     \
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS4(_object, _xx, _yy);                                                     \
}

// *********************************************************************************************************************

PyArrayObject *PYCV_draw_circle(npy_intp *center_point, npy_intp radius)
{
    npy_intp y0, x0, xx = 0, yy = radius, max_size, ii, err = 0;
    PYCV_CoordinatesList circle;
    npy_intp p_dims[2] = {0, 2}, stride;
    int num_type_p;
    PyArrayObject *yx;
    char *pyx = NULL;

    NPY_BEGIN_THREADS_DEF;

    y0 = center_point[0];
    x0 = center_point[1];

    err = 3 - 2 * radius;
    max_size = (radius * 8) + 8;

    if (!PYCV_CoordinatesListInit(2, max_size, &circle)) {
        PyErr_NoMemory();
        goto exit;
    }

    NPY_BEGIN_THREADS;

    while (yy > xx) {
        if (xx == 0) {
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(circle, yy, xx);
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(circle, -yy, xx);
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(circle, xx, yy);
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(circle, xx, -yy);
        } else {
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS8(circle, yy, xx);
        }
        if (err < 0) {
            err += 4 * xx + 6;
        } else {
            err += 4 * (xx - yy) + 10;
            yy -= 1;
        }
        xx += 1;
    }
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS4(circle, yy, xx);

    NPY_END_THREADS;

    p_dims[0] = circle.coordinates_size;
    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }
    num_type_p = PyArray_TYPE(yx);
    stride = PyArray_STRIDE(yx, 1);
    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii < circle.coordinates_size; ii++) {
        circle.coordinates[ii][0] += y0;
        circle.coordinates[ii][1] += x0;
        PYCV_T_SET_POINT(num_type_p, stride, 2, pyx, circle.coordinates[ii]);
    }
    exit:
        if (circle.coordinates_size >= 0) {
            PYCV_CoordinatesListFree(&circle);
        }
        return PyErr_Occurred() ? NULL : yx;
}

// #####################################################################################################################

PyArrayObject *PYCV_draw_ellipse(npy_intp *center_point, npy_intp a, npy_intp b)
{
    npy_intp y0, x0, ry, rx, tr_y, tr_x, yy, xx = 0, py, px = 0, ii, p;
    npy_intp max_size;
    PYCV_CoordinatesList ellipse;
    npy_intp p_dims[2] = {0, 2}, stride;
    int num_type_p;
    PyArrayObject *yx;
    char *pyx = NULL;

    NPY_BEGIN_THREADS_DEF;

    y0 = center_point[0];
    x0 = center_point[1];

    rx = a * a;
    ry = b * b;

    tr_x = rx + rx;
    tr_y = ry + ry;

    yy = b;
    py = tr_x * yy;

    max_size = a > b ? a : b;
    max_size = max_size * 8 + 8;

    if (!PYCV_CoordinatesListInit(2, max_size, &ellipse)) {
        PyErr_NoMemory();
        goto exit;
    }

    NPY_BEGIN_THREADS;

    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(ellipse, yy, xx);
    PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(ellipse, -yy, xx);

    p = (npy_intp)((float)ry - (float)(rx * b) + (0.25 * (float)rx));

    while (px < py) {
        xx += 1;
        px += tr_y;
        if (p < 0) {
            p += ry + px;
        } else {
            yy -= 1;
            py -= tr_x;
            p += ry + px - py;
        }
        PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS4(ellipse, yy, xx);
    }

    p = (npy_intp)((float)ry * ((float)xx + 0.5) * ((float)xx + 0.5) + (float)(rx * (yy - 1) * (yy - 1)) - (float)(rx * ry));

    while (yy > 0) {
        yy -= 1;
        py -= tr_x;
        if (p > 0) {
            p += rx - py;
        } else {
            xx += 1;
            px += tr_y;
            p += rx - py + px;
        }
        if (yy == 0) {
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(ellipse, yy, xx);
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_YX(ellipse, yy, -xx);
        } else {
            PYCV_D_CIRCLE_COORDINATES_LIST_ADD_POINTS4(ellipse, yy, xx);
        }
    }

    NPY_END_THREADS;

    p_dims[0] = ellipse.coordinates_size;
    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }
    num_type_p = PyArray_TYPE(yx);
    stride = PyArray_STRIDE(yx, 1);
    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii < ellipse.coordinates_size; ii++) {
        ellipse.coordinates[ii][0] += y0;
        ellipse.coordinates[ii][1] += x0;
        PYCV_T_SET_POINT(num_type_p, stride, 2, pyx, ellipse.coordinates[ii]);
    }
    exit:
        if (ellipse.coordinates_size >= 0) {
            PYCV_CoordinatesListFree(&ellipse);
        }
        return PyErr_Occurred() ? NULL : yx;
}
