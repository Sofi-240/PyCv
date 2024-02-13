#include "c_pycv_base.h"
#include "c_pycv_canny.h"

// #####################################################################################################################

#define PYCV_C_CASE_CANNY_INTERPOLATE(_NTYPE, _dtype, _mag_p, _gy, _gx, _th, _mask, _offsets, _out)                    \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    unsigned int _grad_up, _grad_down, _grad_left, _grad_right, _diag_pos, _diag_neg;                                  \
    npy_double _abs_gy, _abs_gx, _d, _mag, _p11, _p12, _p21, _p22;                                                     \
    _mag = (npy_double)(*((_dtype *)_mag_p));                                                                          \
    _out = 0;                                                                                                          \
    if (_mask && _mag >= _th) {                                                                                        \
        _grad_up = _gy >= 0 ? 1 : 0;                                                                                   \
        _grad_down = _gy <= 0 ? 1 : 0;                                                                                 \
        _grad_right = _gx >= 0 ? 1 : 0;                                                                                \
        _grad_left = _gx <= 0 ? 1 : 0;                                                                                 \
        _diag_neg = (_grad_up && _grad_right) || (_grad_down && _grad_left) ? 1 : 0;                                   \
        _diag_pos = (_grad_down && _grad_right) || (_grad_up && _grad_left) ? 1 : 0;                                   \
        if (_diag_neg || _diag_pos) {                                                                                  \
            _abs_gy = _gy < 0 ? -_gy : _gy;                                                                            \
            _abs_gx = _gx < 0 ? -_gx : _gx;                                                                            \
            if (_diag_neg && _abs_gy > _abs_gx) {                                                                      \
                _d = _abs_gx / _abs_gy;                                                                                \
                _p11 = (npy_double)(*((_dtype *)(_mag_p + _offsets[7])));                                              \
                _p12 = (npy_double)(*((_dtype *)(_mag_p + _offsets[8])));                                              \
                _p21 = (npy_double)(*((_dtype *)(_mag_p + _offsets[1])));                                              \
                _p22 = (npy_double)(*((_dtype *)(_mag_p + _offsets[0])));                                              \
            } else if (_diag_neg) {                                                                                    \
                _d = _abs_gy / _abs_gx;                                                                                \
                _p11 = (npy_double)(*((_dtype *)(_mag_p + _offsets[5])));                                              \
                _p12 = (npy_double)(*((_dtype *)(_mag_p + _offsets[8])));                                              \
                _p21 = (npy_double)(*((_dtype *)(_mag_p + _offsets[3])));                                              \
                _p22 = (npy_double)(*((_dtype *)(_mag_p + _offsets[0])));                                              \
            } else if (_diag_pos && _abs_gy < _abs_gx) {                                                               \
                _d = _abs_gy / _abs_gx;                                                                                \
                _p11 = (npy_double)(*((_dtype *)(_mag_p + _offsets[5])));                                              \
                _p12 = (npy_double)(*((_dtype *)(_mag_p + _offsets[2])));                                              \
                _p21 = (npy_double)(*((_dtype *)(_mag_p + _offsets[3])));                                              \
                _p22 = (npy_double)(*((_dtype *)(_mag_p + _offsets[6])));                                              \
            } else {                                                                                                   \
                _d = _abs_gx / _abs_gy;                                                                                \
                _p11 = (npy_double)(*((_dtype *)(_mag_p + _offsets[1])));                                              \
                _p12 = (npy_double)(*((_dtype *)(_mag_p + _offsets[2])));                                              \
                _p21 = (npy_double)(*((_dtype *)(_mag_p + _offsets[7])));                                              \
                _p22 = (npy_double)(*((_dtype *)(_mag_p + _offsets[6])));                                              \
            }                                                                                                          \
            if ((_p12 * _d + _p11 * (1 - _d)) <= _mag && (_p22 * _d + _p21 * (1 - _d)) <= _mag) {                      \
                _out = 1;                                                                                              \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_C_CANNY_INTERPOLATE(_NTYPE, _mag_p, _gy, _gx, _th, _mask, _offsets, _out)                                 \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_C_CASE_CANNY_INTERPOLATE(BOOL, npy_bool, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                   \
        PYCV_C_CASE_CANNY_INTERPOLATE(UBYTE, npy_ubyte, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                 \
        PYCV_C_CASE_CANNY_INTERPOLATE(USHORT, npy_ushort, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);               \
        PYCV_C_CASE_CANNY_INTERPOLATE(UINT, npy_uint, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                   \
        PYCV_C_CASE_CANNY_INTERPOLATE(ULONG, npy_ulong, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                 \
        PYCV_C_CASE_CANNY_INTERPOLATE(ULONGLONG, npy_ulonglong, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);         \
        PYCV_C_CASE_CANNY_INTERPOLATE(BYTE, npy_byte, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                   \
        PYCV_C_CASE_CANNY_INTERPOLATE(SHORT, npy_short, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                 \
        PYCV_C_CASE_CANNY_INTERPOLATE(INT, npy_int, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                     \
        PYCV_C_CASE_CANNY_INTERPOLATE(LONG, npy_long, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                   \
        PYCV_C_CASE_CANNY_INTERPOLATE(LONGLONG, npy_longlong, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);           \
        PYCV_C_CASE_CANNY_INTERPOLATE(FLOAT, npy_float, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);                 \
        PYCV_C_CASE_CANNY_INTERPOLATE(DOUBLE, npy_double, _mag_p, _gy, _gx, _th, _mask, _offsets, _out);               \
    }                                                                                                                  \
}

int PYCV_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                      PyArrayObject *grad_y,
                                      PyArrayObject *grad_x,
                                      npy_double threshold,
                                      PyArrayObject *mask,
                                      PyArrayObject *output)
{
    npy_intp f_shape[2] = {3, 3}, f_center[2] = {1, 1}, f_size = 9;
    npy_intp array_size, *offsets, *ff, flag, ii, ma = 1;
    PYCV_ArrayIterator iter_gy, iter_gx, iter_ma, iter_o;
    NeighborhoodIterator iter_mag;
    char *p_mag = NULL, *p_gy = NULL, *p_gx = NULL, *p_ma = NULL, *p_o = NULL;
    int num_type_mag, num_type_gy, num_type_gx, num_type_ma, num_type_o;
    npy_double gy, gx, out;

    NPY_BEGIN_THREADS_DEF;

    if (PyArray_NDIM(magnitude) != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Error: magnitude ndim need to be 2");
        goto exit;
    }

    array_size = PyArray_SIZE(magnitude);
    PYCV_NeighborhoodIteratorInit(magnitude, f_shape, f_center, f_size, &iter_mag);

    if (!PYCV_InitNeighborhoodOffsets(magnitude, f_shape, f_center, NULL, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    PYCV_ArrayIteratorInit(grad_y, &iter_gy);
    PYCV_ArrayIteratorInit(grad_x, &iter_gx);
    if (mask) {
        PYCV_ArrayIteratorInit(mask, &iter_ma);
    }
    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_mag = PyArray_TYPE(magnitude);
    num_type_gy = PyArray_TYPE(grad_y);
    num_type_gx = PyArray_TYPE(grad_x);
    if (mask) {
        num_type_ma = PyArray_TYPE(mask);
    }
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    p_mag = (void *)PyArray_DATA(magnitude);
    p_gy = (void *)PyArray_DATA(grad_y);
    p_gx = (void *)PyArray_DATA(grad_x);
    if (mask) {
        p_ma = (void *)PyArray_DATA(mask);
    }
    p_o = (void *)PyArray_DATA(output);
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        if (ff[0] != flag && ff[f_size - 1] != flag) {
            if (mask) {
                PYCV_GET_VALUE(num_type_ma, npy_intp, p_ma, ma);
            }
            PYCV_GET_VALUE(num_type_gy, npy_double, p_gy, gy);
            PYCV_GET_VALUE(num_type_gx, npy_double, p_gx, gx);
            PYCV_C_CANNY_INTERPOLATE(num_type_mag, p_mag, gy, gx, threshold, ma, ff, out);
        } else {
            out = 0;
        }
        PYCV_SET_VALUE(num_type_o, p_o, out);
        PYCV_NEIGHBORHOOD_ITERATOR_NEXT3(iter_mag, p_mag, iter_gy, p_gy, iter_gx, p_gx, ff);
        if (mask) {
            PYCV_ARRAY_ITERATOR_NEXT2(iter_ma, p_ma, iter_o, p_o);
        } else {
            PYCV_ARRAY_ITERATOR_NEXT(iter_o, p_o);
        }
    }

    NPY_END_THREADS;
    exit:
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

