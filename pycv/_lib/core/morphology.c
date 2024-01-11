#include "ops_support.h"
#include "morphology.h"

// #####################################################################################################################

#define TYPE_CASE_BINARY_EROSION(_TYPE, _type, _pi, _buffer_size,                                \
                                _offsets, _offsets_flag, _buffer_val, _true_val, _false_val)     \
case _TYPE:                                                                                      \
{                                                                                                \
    npy_intp _ii;                                                                                \
    _buffer_val = _true_val;                                                                     \
    for (_ii = 0; _ii < _buffer_size; _ii++) {                                                   \
        if (_offsets[_ii] < _offsets_flag) {                                                     \
            _buffer_val = *(_type *)(_pi + _offsets[_ii]) ? _true_val : _false_val;              \
        }                                                                                        \
        if (!_buffer_val) {                                                                      \
            break;                                                                               \
        }                                                                                        \
    }                                                                                            \
}                                                                                                \
break

#define TYPE_CASE_GET_VALUE_BINARY(_TYPE, _type, _pi, _buffer_val, _true_val, _false_val)        \
case _TYPE:                                                                                      \
{                                                                                                \
    _buffer_val = *(_type *)_pi ? _true_val : _false_val;                                        \
}                                                                                                \
break

int ops_binary_erosion(PyArrayObject *input,
                       PyArrayObject *strel,
                       PyArrayObject *output,
                       npy_intp *origins,
                       int iterations,
                       PyArrayObject *mask,
                       int invert)
{
    Base_Iterator dptr_o, dptr_i, dptr_s, dptr_m;
    Neighborhood_Iterator dptr_of;
    char *pi = NULL, *po = NULL, *si = NULL, *ma = NULL;
    npy_bool *footprint;
    npy_intp nd, ii, jj, offsets_stride, offsets_flag, *of, *offsets;
    npy_bool buffer, _true, _false;

    NPY_BEGIN_THREADS_DEF;
    nd = PyArray_NDIM(input);
    if (!INIT_Base_Iterator(strel, &dptr_s)){
        goto exit;
    }

    si = (void *)PyArray_DATA(strel);
    footprint = (npy_bool *)malloc(PyArray_SIZE(strel) * sizeof(npy_bool));
    if (!footprint) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < PyArray_SIZE(strel); ii++) {
        footprint[ii] = *(npy_bool *)si ? NPY_TRUE : NPY_FALSE;
        BASE_ITERATOR_NEXT(dptr_s, si);
    }

    INIT_OFFSETS_ARRAY(input, PyArray_DIMS(strel), origins, footprint, &offsets, &offsets_flag, &offsets_stride);

    if (!INIT_Neighborhood_Iterator(offsets_stride, PyArray_SIZE(input), &dptr_of)){
        goto exit;
    }

    if (!INIT_Base_Iterator(input, &dptr_i)){
        goto exit;
    }

    if (!INIT_Base_Iterator(output, &dptr_o)){
        goto exit;
    }

    if (mask) {
        if (!INIT_Base_Iterator(mask, &dptr_m)){
            goto exit;
        }
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }

    of = offsets;
    if (invert) {
        _true = NPY_FALSE;
        _false = NPY_TRUE;
    } else {
        _true = NPY_TRUE;
        _false = NPY_FALSE;
    }

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        buffer = _true;
        if (!mask || *(npy_bool *)(ma)) {
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_BINARY_EROSION(NPY_BOOL, npy_bool,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_UBYTE, npy_ubyte,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_USHORT, npy_ushort,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_UINT, npy_uint,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_ULONG, npy_ulong,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_ULONGLONG, npy_ulonglong,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_BYTE, npy_byte,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_SHORT, npy_short,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_INT, npy_int,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_LONG, npy_long,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_LONGLONG, npy_longlong,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_FLOAT, npy_float,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                TYPE_CASE_BINARY_EROSION(NPY_DOUBLE, npy_double,
                                         pi, offsets_stride, of, offsets_flag, buffer, _true, _false);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
        } else {
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_GET_VALUE_BINARY(NPY_BOOL, npy_bool, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_UBYTE, npy_ubyte, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_USHORT, npy_ushort, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_UINT, npy_uint, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_ULONG, npy_ulong, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_ULONGLONG, npy_ulonglong, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_BYTE, npy_byte, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_SHORT, npy_short, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_INT, npy_int, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_LONG, npy_long, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_LONGLONG, npy_longlong, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_FLOAT, npy_float, pi, buffer, _true, _false);
                TYPE_CASE_GET_VALUE_BINARY(NPY_DOUBLE, npy_double, pi, buffer, _true, _false);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
        }
        switch (PyArray_TYPE(output)) {
            TYPE_CASE_VALUE_OUT(NPY_BOOL, npy_bool, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_UBYTE, npy_ubyte, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_USHORT, npy_ushort, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_UINT, npy_uint, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_ULONG, npy_ulong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_ULONGLONG, npy_ulonglong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, buffer);
            default:
                NPY_END_THREADS;
                PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
                goto exit;
        }

        if (mask) {
            BASE_ITERATOR_NEXT3(dptr_i, pi, dptr_o, po, dptr_m, ma);
        } else {
            BASE_ITERATOR_NEXT2(dptr_i, pi, dptr_o, po);
        }
        NEIGHBORHOOD_ITERATOR_NEXT(dptr_of, of);
    }

    NPY_END_THREADS;

    exit:
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define TYPE_CASE_GRAY_EROSION(_TYPE, _type, _pi, _buffer_size, _offsets, _offsets_flag, _weights, _buffer_val)       \
case _TYPE:                                                                                                           \
{                                                                                                                     \
    npy_intp _ii, _jj = 0;                                                                                            \
    double _tmp;                                                                                                      \
    _buffer_val = 0.0;                                                                                                \
    while (_jj < _buffer_size) {                                                                                      \
        if (_offsets[_jj] < _offsets_flag) {                                                                          \
            _buffer_val = (double)(*((_type *)(_pi + _offsets[_jj]))) - _weights[_jj];                                \
            break;                                                                                                    \
        }                                                                                                             \
        _jj++;                                                                                                        \
    }                                                                                                                 \
    for (_ii = _jj + 1; _ii < _buffer_size; _ii++) {                                                                  \
        if (_offsets[_ii] < _offsets_flag) {                                                                          \
            _tmp = (double)(*((_type *)(_pi + _offsets[_ii]))) - _weights[_jj];                                       \
            _buffer_val = _buffer_val < _tmp ? _buffer_val : _tmp;                                                    \
        }                                                                                                             \
    }                                                                                                                 \
}                                                                                                                     \
break

#define TYPE_CASE_GRAY_DILATION(_TYPE, _type, _pi, _buffer_size, _offsets, _offsets_flag, _weights, _buffer_val)      \
case _TYPE:                                                                                                           \
{                                                                                                                     \
    npy_intp _ii, _jj = 0;                                                                                            \
    double _tmp;                                                                                                      \
    _buffer_val = 0.0;                                                                                                \
    while (_jj < _buffer_size) {                                                                                      \
        if (_offsets[_jj] < _offsets_flag) {                                                                          \
            _buffer_val = (double)(*((_type *)(_pi + _offsets[_jj]))) + _weights[_jj];                                \
            break;                                                                                                    \
        }                                                                                                             \
        _jj++;                                                                                                        \
    }                                                                                                                 \
    for (_ii = _jj + 1; _ii < _buffer_size; _ii++) {                                                                  \
        if (_offsets[_ii] < _offsets_flag) {                                                                          \
            _tmp = (double)(*((_type *)(_pi + _offsets[_ii]))) + _weights[_ii];                                       \
            _buffer_val = _buffer_val > _tmp ? _buffer_val : _tmp;                                                    \
        }                                                                                                             \
    }                                                                                                                 \
}                                                                                                                     \
break

int ops_erosion(PyArrayObject *input,
                PyArrayObject *flat_strel,
                PyArrayObject *non_flat_strel,
                PyArrayObject *output,
                npy_intp *origins,
                PyArrayObject *mask,
                double cast_value)
{
    Base_Iterator dptr_o, dptr_i, dptr_s, dptr_m;
    Neighborhood_Iterator dptr_of;
    char *pi = NULL, *po = NULL, *fsi = NULL, *ma = NULL;
    npy_bool *footprint;
    npy_intp nd, ii, jj, offsets_stride, offsets_flag, *of, *offsets;
    double buffer, *weights;

    NPY_BEGIN_THREADS_DEF;
    nd = PyArray_NDIM(input);

    if (!INIT_Base_Iterator(flat_strel, &dptr_s)){
        goto exit;
    }

    fsi = (void *)PyArray_DATA(flat_strel);
    footprint = (npy_bool *)malloc(PyArray_SIZE(flat_strel) * sizeof(npy_bool));
    if (!footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    for (ii = 0; ii < PyArray_SIZE(flat_strel); ii++) {
        footprint[ii] = *(npy_bool *)fsi ? NPY_TRUE : NPY_FALSE;
        BASE_ITERATOR_NEXT(dptr_s, fsi);
    }

    INIT_OFFSETS_ARRAY(input, PyArray_DIMS(flat_strel), origins, footprint, &offsets, &offsets_flag, &offsets_stride);

    if (non_flat_strel) {
        if (!COPY_DATA_TO_DOUBLE(non_flat_strel, &weights, footprint)) {
            goto exit;
        }
    } else {
        weights = (double *)malloc(offsets_stride * sizeof(double));
        if (!weights) {
            PyErr_NoMemory();
            goto exit;
        }
        for (ii = 0; ii < offsets_stride; ii++) {
            weights[ii] = 0.0;
        }
    }

    if (!INIT_Neighborhood_Iterator(offsets_stride, PyArray_SIZE(input), &dptr_of)){
        goto exit;
    }

    if (!INIT_Base_Iterator(input, &dptr_i)){
        goto exit;
    }

    if (!INIT_Base_Iterator(output, &dptr_o)){
        goto exit;
    }

    if (mask) {
        if (!INIT_Base_Iterator(mask, &dptr_m)){
            goto exit;
        }
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }

    of = offsets;

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        buffer = 0.0;
        if (!mask || *(npy_bool *)(ma)) {
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_GRAY_EROSION(NPY_BOOL, npy_bool,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_UBYTE, npy_ubyte,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_USHORT, npy_ushort,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_UINT, npy_uint,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_ULONG, npy_ulong,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_ULONGLONG, npy_ulonglong,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_BYTE, npy_byte,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_SHORT, npy_short,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_INT, npy_int,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_LONG, npy_long,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_LONGLONG, npy_longlong,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_FLOAT, npy_float,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_EROSION(NPY_DOUBLE, npy_double,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
        } else {
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_BOOL, npy_bool, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_UBYTE, npy_ubyte, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_USHORT, npy_ushort, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_UINT, npy_uint, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_ULONG, npy_ulong, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_ULONGLONG, npy_ulonglong, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_BYTE, npy_byte, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_SHORT, npy_short, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_INT, npy_int, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_LONG, npy_long, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_LONGLONG, npy_longlong, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_FLOAT, npy_float, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_DOUBLE, npy_double, pi, buffer);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
        }
        buffer = cast_value > buffer ? cast_value : buffer;
        switch (PyArray_TYPE(output)) {
            TYPE_CASE_VALUE_OUT_F2U(NPY_BOOL, npy_bool, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_UBYTE, npy_ubyte, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_USHORT, npy_ushort, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_UINT, npy_uint, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_ULONG, npy_ulong, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_ULONGLONG, npy_ulonglong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, buffer);
            default:
                NPY_END_THREADS;
                PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
                goto exit;
        }
        if (mask) {
            BASE_ITERATOR_NEXT3(dptr_i, pi, dptr_o, po, dptr_m, ma);
        } else {
            BASE_ITERATOR_NEXT2(dptr_i, pi, dptr_o, po);
        }
        NEIGHBORHOOD_ITERATOR_NEXT(dptr_of, of);
    }

    NPY_END_THREADS;
    exit:
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;

}

int ops_dilation(PyArrayObject *input,
                 PyArrayObject *flat_strel,
                 PyArrayObject *non_flat_strel,
                 PyArrayObject *output,
                 npy_intp *origins,
                 PyArrayObject *mask,
                 double cast_value)
{
    Base_Iterator dptr_o, dptr_i, dptr_s, dptr_m;
    Neighborhood_Iterator dptr_of;
    char *pi = NULL, *po = NULL, *fsi = NULL, *ma = NULL;
    npy_bool *footprint;
    npy_intp nd, ii, jj, offsets_stride, offsets_flag, *of, *offsets;
    double buffer, *weights = NULL;

    NPY_BEGIN_THREADS_DEF;
    nd = PyArray_NDIM(input);

    if (!INIT_Base_Iterator(flat_strel, &dptr_s)){
        goto exit;
    }

    fsi = (void *)PyArray_DATA(flat_strel);
    footprint = (npy_bool *)malloc(PyArray_SIZE(flat_strel) * sizeof(npy_bool));
    if (!footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    for (ii = 0; ii < PyArray_SIZE(flat_strel); ii++) {
        footprint[ii] = *(npy_bool *)fsi ? NPY_TRUE : NPY_FALSE;
        BASE_ITERATOR_NEXT(dptr_s, fsi);
    }

    INIT_OFFSETS_ARRAY(input, PyArray_DIMS(flat_strel), origins, footprint, &offsets, &offsets_flag, &offsets_stride);

    if (non_flat_strel) {
        if (!COPY_DATA_TO_DOUBLE(non_flat_strel, &weights, footprint)) {
            goto exit;
        }
    } else {
        weights = (double *)malloc(offsets_stride * sizeof(double));
        if (!weights) {
            PyErr_NoMemory();
            goto exit;
        }
        for (ii = 0; ii < offsets_stride; ii++) {
            weights[ii] = 0.0;
        }
    }



    if (!INIT_Neighborhood_Iterator(offsets_stride, PyArray_SIZE(input), &dptr_of)){
        goto exit;
    }

    if (!INIT_Base_Iterator(input, &dptr_i)){
        goto exit;
    }

    if (!INIT_Base_Iterator(output, &dptr_o)){
        goto exit;
    }

    if (mask) {
        if (!INIT_Base_Iterator(mask, &dptr_m)){
            goto exit;
        }
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }

    of = offsets;

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        buffer = 0.0;
        if (!mask || *(npy_bool *)(ma)) {
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_GRAY_DILATION(NPY_BOOL, npy_bool,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_UBYTE, npy_ubyte,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_USHORT, npy_ushort,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_UINT, npy_uint,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_ULONG, npy_ulong,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_ULONGLONG, npy_ulonglong,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_BYTE, npy_byte,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_SHORT, npy_short,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_INT, npy_int,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_LONG, npy_long,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_LONGLONG, npy_longlong,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_FLOAT, npy_float,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                TYPE_CASE_GRAY_DILATION(NPY_DOUBLE, npy_double,
                                       pi, offsets_stride, of, offsets_flag, weights, buffer);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
        } else {
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_BOOL, npy_bool, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_UBYTE, npy_ubyte, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_USHORT, npy_ushort, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_UINT, npy_uint, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_ULONG, npy_ulong, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_ULONGLONG, npy_ulonglong, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_BYTE, npy_byte, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_SHORT, npy_short, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_INT, npy_int, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_LONG, npy_long, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_LONGLONG, npy_longlong, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_FLOAT, npy_float, pi, buffer);
                TYPE_CASE_GET_VALUE_DOUBLE(NPY_DOUBLE, npy_double, pi, buffer);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
        }
        buffer = cast_value < buffer ? cast_value : buffer;
        switch (PyArray_TYPE(output)) {
            TYPE_CASE_VALUE_OUT_F2U(NPY_BOOL, npy_bool, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_UBYTE, npy_ubyte, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_USHORT, npy_ushort, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_UINT, npy_uint, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_ULONG, npy_ulong, po, buffer);
            TYPE_CASE_VALUE_OUT_F2U(NPY_ULONGLONG, npy_ulonglong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, buffer);
            TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, buffer);
            default:
                NPY_END_THREADS;
                PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
                goto exit;
        }
        if (mask) {
            BASE_ITERATOR_NEXT3(dptr_i, pi, dptr_o, po, dptr_m, ma);
        } else {
            BASE_ITERATOR_NEXT2(dptr_i, pi, dptr_o, po);
        }
        NEIGHBORHOOD_ITERATOR_NEXT(dptr_of, of);
    }

    NPY_END_THREADS;
    exit:
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

int ops_binary_region_fill(PyArrayObject *output,
                           PyArrayObject *strel,
                           npy_intp *seed_point,
                           npy_intp *origins)
{
    npy_bool valid, val, *footprint = NULL;
    npy_intp ii, jj, nd, array_size, array_shape[NPY_MAXDIMS], position[NPY_MAXDIMS], footprint_size = 0, *offsets_coord, *of_c;
    npy_intp seed_ravel, stack_start = 0, stack_end = 0, *stack, *stack_fill, *stack_go;
    Base_Iterator dptr_o;
    char *po_base = NULL, *po = NULL;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(output);
    nd = PyArray_NDIM(output);

    footprint = (npy_bool *)PyArray_DATA(strel);

    for (ii = 0; ii < PyArray_SIZE(strel); ii++) {
        if (footprint[ii]) {
            footprint_size++;
        }
    }

    if (!INIT_OFFSETS_AS_COORDINATES(nd, PyArray_DIMS(strel), origins, footprint, &offsets_coord)) {
        goto exit;
    }

    stack = (npy_intp *)malloc(array_size * nd * sizeof(npy_intp));
    if (!stack) {
        PyErr_NoMemory();
        goto exit;
    }
    stack_fill = stack;
    stack_go = stack;

    for (ii = 0; ii < nd; ii++) {
        position[ii] = 0;
        *stack_fill++ = seed_point[ii];
        array_shape[ii] = PyArray_DIM(output, ii);
    }
    stack_end = 1;

    if (!INIT_Base_Iterator(output, &dptr_o)){
        goto exit;
    }

    NPY_BEGIN_THREADS;

    po_base = po = (void *)PyArray_DATA(output);

    BASE_ITERATOR_GOTO(dptr_o, stack_go, po_base, po);

    switch (PyArray_TYPE(output)) {
        TYPE_CASE_VALUE_OUT(NPY_BOOL, npy_bool, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_UBYTE, npy_ubyte, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_USHORT, npy_ushort, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_UINT, npy_uint, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_ULONG, npy_ulong, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_ULONGLONG, npy_ulonglong, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, NPY_TRUE);
        TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, NPY_TRUE);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
            goto exit;
    }

    while (stack_start < stack_end) {
        of_c = offsets_coord;

        for (ii = 0; ii < footprint_size; ii++) {
            valid = NPY_TRUE;
            for (jj = 0; jj < nd; jj++) {
                position[jj] = stack_go[jj] - of_c[jj];
                if (position[jj] < 0 || position[jj] >= array_shape[jj]) {
                    valid = NPY_FALSE;
                    break;
                }
            }
            if (valid) {
                val = NPY_FALSE;
                BASE_ITERATOR_GOTO(dptr_o, position, po_base, po);
                switch (PyArray_TYPE(output)) {
                    TYPE_CASE_GET_VALUE_BOOL(NPY_BOOL, npy_bool, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_UBYTE, npy_ubyte, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_USHORT, npy_ushort, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_UINT, npy_uint, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_ULONG, npy_ulong, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_ULONGLONG, npy_ulonglong, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_BYTE, npy_byte, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_SHORT, npy_short, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_INT, npy_int, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_LONG, npy_long, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_LONGLONG, npy_longlong, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_FLOAT, npy_float, po, val);
                    TYPE_CASE_GET_VALUE_BOOL(NPY_DOUBLE, npy_double, po, val);
                    default:
                        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                        goto exit;
                }
                if (!val) {
                    switch (PyArray_TYPE(output)) {
                        TYPE_CASE_VALUE_OUT(NPY_BOOL, npy_bool, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_UBYTE, npy_ubyte, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_USHORT, npy_ushort, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_UINT, npy_uint, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_ULONG, npy_ulong, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_ULONGLONG, npy_ulonglong, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, NPY_TRUE);
                        TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, NPY_TRUE);
                        default:
                            NPY_END_THREADS;
                            PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                            goto exit;
                    }
                    for (jj = 0; jj < nd; jj++) {
                        *stack_fill++ = position[jj];
                    }
                    stack_end++;
                }
            }
            of_c += nd;
        }
        stack_start++;
        stack_go += nd;
    }

    NPY_END_THREADS;
    exit:
        free(offsets_coord);
        free(stack);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################








