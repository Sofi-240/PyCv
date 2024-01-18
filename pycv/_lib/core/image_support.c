#include "ops_base.h"
#include "image_support.h"

// #####################################################################################################################

/*
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


is_up and is_right:
gy >= 0 and gx >= 0

  |
   __ ---->  / ---> \ gradient direction

is_down and is_left:
gy <= 0 and gx <= 0

  __     ---> / ---> \ gradient direction
     |

is_down and is_right:
gy <= 0 and gx >= 0

    __   ---> \---> / gradient direction
   |

is_up and is_left:
gy >= 0 and gx <= 0

    |  ---> \---> / gradient direction
  __

[0, 1, 2],
[3, 4, 5]
[6, 7, 8]

90 - 135                    135 - 180
[22, 21, 0],                [22,  0,  0],
[0,  1,  0],      or        [21,  1, 11],
[0, 11, 12]                 [ 0,  0, 12],

|gy| > |gx|                 |gy| <= |gx|
e = |gx| / |gy|             e = |gy| / |gx|

0 - 45                     45 - 90
[ 0,  0, 12],              [ 0, 11, 12],
[21,  1, 11],      or      [ 0,  1,  0],
[22,  0,  0]               [22, 21,  0],

|gx| > |gy|                |gx| <= |gy|
e = |gy| / |gx|            e = |gx| / |gy|

upper = (12 * w + 11 * (1.0 - w))
lower = (22 * w + 21 * (1.0 - w))
*/

#define CASE_TYPE_CANNY_INTERPOLATE(_NUM_TYPE, _type, _offsets, _th, _mag_po, _gy, _gx, _out_val)                      \
case _NUM_TYPE:                                                                                                        \
{                                                                                                                      \
    unsigned int _grad_up, _grad_down, _grad_left, _grad_right, _diag_pos, _diag_neg;                                  \
    double _abs_gy, _abs_gx, _est, _mag, _p11, _p12, _p21, _p22;                                                       \
    _mag = (double)(*((_type *)_mag_po));                                                                              \
    _out_val = 0.0;                                                                                                    \
    if (_mag >= _th) {                                                                                                 \
        _grad_up = _gy >= 0 ? 1 : 0;                                                                                   \
        _grad_down = _gy <= 0 ? 1 : 0;                                                                                 \
        _grad_right = _gx >= 0 ? 1 : 0;                                                                                \
        _grad_left = _gx <= 0 ? 1 : 0;                                                                                 \
        _diag_neg = (_grad_up && _grad_right) || (_grad_down && _grad_left) ? 1 : 0;                                   \
        _diag_pos = (_grad_down && _grad_right) || (_grad_up && _grad_left) ? 1 : 0;                                   \
        if (_diag_neg || _diag_pos) {                                                                                  \
            _abs_gy = fabs(_gy);                                                                                       \
            _abs_gx = fabs(_gx);                                                                                       \
            if (_diag_neg && _abs_gy > _abs_gx) {                                                                      \
                _est = _abs_gx / _abs_gy;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[7])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[8])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[1])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[0])));                                                  \
            } else if (_diag_neg) {                                                                                    \
                _est = _abs_gy / _abs_gx;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[5])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[8])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[3])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[0])));                                                  \
            } else if (_diag_pos && _abs_gy < _abs_gx) {                                                               \
                _est = _abs_gy / _abs_gx;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[5])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[2])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[3])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[6])));                                                  \
            } else {                                                                                                   \
                _est = _abs_gx / _abs_gy;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[1])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[2])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[7])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[6])));                                                  \
            }                                                                                                          \
            if ((_p12 * _est + _p11 * (1 - _est)) <= _mag && (_p22 * _est + _p21 * (1 - _est)) <= _mag) {              \
                _out_val = _mag;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define EX_CANNY_INTERPOLATE(_NUM_TYPE, _offsets, _th, _mag_po, _gy, _gx, _out_val)                                    \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_BOOL, npy_bool, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_UBYTE, npy_ubyte, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_USHORT, npy_ushort, _offsets, _th, _mag_po, _gy, _gx, _out_val);               \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_UINT, npy_uint, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_ULONG, npy_ulong, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_ULONGLONG, npy_ulonglong, _offsets, _th, _mag_po, _gy, _gx, _out_val);         \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_BYTE, npy_byte, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_SHORT, npy_short, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_INT, npy_int, _offsets, _th, _mag_po, _gy, _gx, _out_val);                     \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_LONG, npy_long, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_LONGLONG, npy_longlong, _offsets, _th, _mag_po, _gy, _gx, _out_val);           \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_FLOAT, npy_float, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_DOUBLE, npy_double, _offsets, _th, _mag_po, _gy, _gx, _out_val);               \
    }                                                                                                                  \
}

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output)
{
    const npy_intp shape[2] = {3, 3};

    npy_intp array_size;
    npy_intp *offsets;
    npy_bool *borders_lookup;
    npy_intp ii;

    ArrayIter iter_o, iter_ma, iter_m, iter_y, iter_x;
    char *po = NULL, *pma = NULL, *pm = NULL, *py = NULL, *px = NULL;
    int num_type_o, num_type_ma, num_type_m, num_type_y, num_type_x;

    double gy = 0.0, gx = 0.0, mag;
    int ma;

    NPY_BEGIN_THREADS_DEF;

    if (!init_offsets_ravel(magnitude, shape, NULL, NULL, &offsets)) {
        goto exit;
    }
    if (!init_borders_lut(PyArray_NDIM(magnitude), PyArray_DIMS(magnitude), shape, NULL, &borders_lookup)) {
        goto exit;
    }

    array_size = PyArray_SIZE(magnitude);

    num_type_o = PyArray_TYPE(output);
    if (mask) {
        num_type_ma = PyArray_TYPE(mask);
    }
    num_type_m = PyArray_TYPE(magnitude);
    num_type_y = PyArray_TYPE(grad_y);
    num_type_x = PyArray_TYPE(grad_x);

    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }
    ArrayIterInit(magnitude, &iter_m);
    ArrayIterInit(grad_y, &iter_y);
    ArrayIterInit(grad_x, &iter_x);

    NPY_BEGIN_THREADS;

    po = (void *)PyArray_DATA(output);
    if (mask) {
        pma = (void *)PyArray_DATA(mask);
    }
    pm = (void *)PyArray_DATA(magnitude);
    py = (void *)PyArray_DATA(grad_y);
    px = (void *)PyArray_DATA(grad_x);

    for (ii = 0; ii < array_size; ii++) {
        mag = 0.0;
        ma = 0;
        if (!borders_lookup[ii]) {
            ma = 1;
            if (mask) {
                GET_VALUE_AS(num_type_ma, int, pma, ma);
            }
        }
        if (ma) {
            GET_VALUE_AS(num_type_y, double, py, gy);
            GET_VALUE_AS(num_type_x, double, px, gx);
            EX_CANNY_INTERPOLATE(num_type_m, offsets, threshold, pm, gy, gx, mag);
        }
        SET_VALUE_TO(num_type_o, po, mag);

        ARRAY_ITER_NEXT3(iter_m, pm, iter_y, py, iter_x, px);
        if (mask) {
            ARRAY_ITER_NEXT2(iter_ma, pma, iter_o, po);
        } else {
            ARRAY_ITER_NEXT(iter_o, po);
        }
    }
    NPY_END_THREADS;
    exit:
        free(offsets);
        free(borders_lookup);
        return PyErr_Occurred() ? 0 : 1;
}

int ops_canny_hysteresis_edge_tracking(PyArrayObject *strong_edge, PyArrayObject *week_edge, PyArrayObject *strel)
{
    npy_intp array_size, nd, ii, jj, kk;

    ArrayIter iter_s, iter_w;
    char *ps_base = NULL, *pw_base = NULL, *ps = NULL, *pw = NULL;
    int num_type_s, num_type_w;

    npy_bool *footprint;
    int offsets_size;
    npy_intp *offsets_ravel, *offsets, *offsets_run;

    int tmp_val1 = 0, tmp_val2 = 0, is_valid = 1;

    npy_intp position[NPY_MAXDIMS], stack_start = 0, stack_end = 0, *stack, *stack_fill, *stack_go;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(strong_edge);
    nd = PyArray_NDIM(strong_edge);

    for (kk = 0; kk < nd; kk++) {
        position[kk] = 0;
    }

    num_type_s = PyArray_TYPE(strong_edge);
    num_type_w = PyArray_TYPE(week_edge);

    if (!array_to_footprint(strel, &footprint, &offsets_size)) {
        goto exit;
    }

    if (!init_offsets_ravel(week_edge, PyArray_DIMS(strel), NULL, footprint, &offsets_ravel)) {
        goto exit;
    }

    if (!init_offsets_coordinates(nd, PyArray_DIMS(strel), NULL, footprint, &offsets)) {
        goto exit;
    }

    stack = (npy_intp *)malloc(array_size * nd * sizeof(npy_intp));
    if (!stack) {
        PyErr_NoMemory();
        goto exit;
    }
    stack_fill = stack;
    stack_go = stack;

    ArrayIterInit(strong_edge, &iter_s);
    ArrayIterInit(week_edge, &iter_w);

    NPY_BEGIN_THREADS;

    ps_base = ps = (void *)PyArray_DATA(strong_edge);
    pw_base = pw = (void *)PyArray_DATA(week_edge);

    for (ii = 0; ii < array_size; ii++) {
        GET_VALUE_AS(num_type_s, int, ps, tmp_val1);
        if (tmp_val1) {
            offsets_run = offsets;
            for (jj = 0; jj < offsets_size; jj++) {
                is_valid = 1;
                for (kk = 0; kk < nd; kk++) {
                    position[kk] = iter_s.coordinates[kk] + offsets_run[kk];
                    if (position[kk] < 0 || position[kk] > iter_s.dims_m1[kk]) {
                        is_valid = 0;
                        break;
                    }
                }
                if (is_valid) {
                    GET_VALUE_AS(num_type_w, int, (pw + offsets_ravel[jj]), tmp_val2);
                    if (tmp_val2) {
                        for (kk = 0; kk < nd; kk++) {
                            *stack_fill++ = position[kk];
                        }
                        stack_end++;
                        SET_VALUE_TO(num_type_w, (pw + offsets_ravel[jj]), 0);
                    }
                }
                offsets_run += nd;
            }
        }
        ARRAY_ITER_NEXT2(iter_s, ps, iter_w, pw);
    }

    while (stack_start < stack_end) {
        ARRAY_ITER_GOTO(iter_s, stack_go, ps_base, ps);
        SET_VALUE_TO(num_type_s, ps, 1);

        offsets_run = offsets;

        for (jj = 0; jj < offsets_size; jj++) {
            is_valid = 1;
            for (kk = 0; kk < nd; kk++) {
                position[kk] = iter_s.coordinates[kk] - offsets_run[kk];
                if (position[kk] < 0 || position[kk] > iter_s.dims_m1[kk]) {
                    is_valid = 0;
                    break;
                }
            }
            if (is_valid) {
                ARRAY_ITER_GOTO(iter_w, position, pw_base, pw);
                GET_VALUE_AS(num_type_w, int, pw, tmp_val2);
                if (tmp_val2) {
                    for (kk = 0; kk < nd; kk++) {
                        *stack_fill++ = position[kk];
                    }
                    stack_end++;
                    SET_VALUE_TO(num_type_w, pw, 0);
                }
            }
            offsets_run += nd;
        }
        stack_start++;
        stack_go += nd;
    }

    NPY_END_THREADS;
    exit:
        free(offsets);
        free(offsets_ravel);
        free(footprint);
        free(stack);
        return PyErr_Occurred() ? 0 : 1;
}


// #####################################################################################################################