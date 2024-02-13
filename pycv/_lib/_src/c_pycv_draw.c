#include "c_pycv_base.h"
#include "c_pycv_draw.h"

// #####################################################################################################################

#define PYCV_D_SET_YX(_y, _x, _pointer, _item_size)                                                                    \
{                                                                                                                      \
    *(npy_longlong *)_pointer = (npy_longlong)_y;                                                                      \
    _pointer += _item_size;                                                                                            \
    *(npy_longlong *)_pointer = (npy_longlong)_x;                                                                      \
    _pointer += _item_size;                                                                                            \
}

#define PYCV_D_SWAP_ARGS(_a1, _a2)                                                                                     \
{                                                                                                                      \
    npy_intp _tmp = _a1;                                                                                               \
    _a1 = _a2;                                                                                                         \
    _a2 = _tmp;                                                                                                        \
}

// #####################################################################################################################

#define PYCV_D_ELLIPSE_PUSH_POINTS(_y0, _x0, _yy, _xx, _buffer, _buffer_end)                                           \
{                                                                                                                      \
    _buffer[_buffer_end] = _y0 + _yy;                                                                                  \
    _buffer[_buffer_end + 1] = _x0 + _xx;                                                                              \
    _buffer[_buffer_end + 2] = _y0 + _yy;                                                                              \
    _buffer[_buffer_end + 3] = _x0 - _xx;                                                                              \
    _buffer[_buffer_end + 4] = _y0 - _yy;                                                                              \
    _buffer[_buffer_end + 5] = _x0 + _xx;                                                                              \
    _buffer[_buffer_end + 6] = _y0 - _yy;                                                                              \
    _buffer[_buffer_end + 7] = _x0 - _xx;                                                                              \
    _buffer_end += 8;                                                                                                  \
}

// #####################################################################################################################

PyArrayObject *PYCV_draw_line(npy_intp *point1, npy_intp *point2)
{
    npy_intp size, y1, y2, x1, x2, dy, dx, step_y, step_x, p, yy, xx, m, ii;
    npy_intp p_dims[2], p_itemsize;
    int flag;

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
    p_itemsize = PyArray_ITEMSIZE(yx);

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
            PYCV_D_SET_YX(xx, yy, pyx, p_itemsize);
        } else {
            PYCV_D_SET_YX(yy, xx, pyx, p_itemsize);
        }

        xx += step_x;
        m = p >= 0 ? 1 : 0;

        p += (2 * dy) - (2 * dx) * m;
        yy += step_y * m;
    }

    NPY_END_THREADS;

    exit:
        return PyErr_Occurred() ? NULL : yx;
}

PyArrayObject *PYCV_draw_circle(npy_intp *center_point, npy_intp radius)
{
    npy_intp y0, x0, xx, yy = 0, size, ii, err = 0;

    npy_intp p_dims[2], p_itemsize;
    PyArrayObject *yx;
    char *pyx = NULL;

    NPY_BEGIN_THREADS_DEF;

    y0 = center_point[0];
    x0 = center_point[1];

    xx = radius;

    size = (radius * 8) + 8;

    p_dims[0] = size;
    p_dims[1] = 2;

    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }
    p_itemsize = PyArray_ITEMSIZE(yx);

    NPY_BEGIN_THREADS;

    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii <= radius; ii++) {

        PYCV_D_SET_YX(y0 + yy, x0 + xx, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 + xx, x0 + yy, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 + xx, x0 - yy, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 + yy, x0 - xx, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 - yy, x0 - xx, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 - xx, x0 - yy, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 - xx, x0 + yy, pyx, p_itemsize);
        PYCV_D_SET_YX(y0 - yy, x0 + xx, pyx, p_itemsize);

        if (err + yy + 1 > xx) {
            err += 1 - 2 * xx;
            xx -= 1;
        } else {
            err += 1 + 2 * yy;
            yy += 1;
        }
    }

    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? NULL : yx;
}

PyArrayObject *PYCV_draw_ellipse(npy_intp *center_point, npy_intp a, npy_intp b)
{
    npy_intp y0, x0, ry, rx, tr_y, tr_x, yy, xx = 0, py, px = 0, ii, p;
    npy_intp *buffer, max_size, buffer_end = 0, size;

    npy_intp p_dims[2], p_itemsize;
    PyArrayObject *yx;
    char *pyx = NULL;

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

    buffer = (npy_intp *)malloc(max_size * 2 * sizeof(npy_intp));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    PYCV_D_ELLIPSE_PUSH_POINTS(y0, x0, yy, xx, buffer, buffer_end);

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
        PYCV_D_ELLIPSE_PUSH_POINTS(y0, x0, yy, xx, buffer, buffer_end);
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
        PYCV_D_ELLIPSE_PUSH_POINTS(y0, x0, yy, xx, buffer, buffer_end);
    }

    size = buffer_end / 2;
    p_dims[0] = size;
    p_dims[1] = 2;

    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    pyx = (void *)PyArray_DATA(yx);
    p_itemsize = PyArray_ITEMSIZE(yx);
    for (ii = 0; ii < size; ii++) {
        PYCV_D_SET_YX(buffer[ii * 2], buffer[ii * 2 + 1], pyx, p_itemsize);
    }

    exit:
        free(buffer);
        return PyErr_Occurred() ? NULL : yx;
}

// #####################################################################################################################