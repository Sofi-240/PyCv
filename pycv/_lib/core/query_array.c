#include "ops_support.h"
#include "query_array.h"


// #####################################################################################################################

#define CASE_ISLOCAL(_TYPE, _type, _pi, _buffer_size, _offsets, _mode, _offsets_flag, _out)   \
case _TYPE:    \
{    \
    npy_intp _ii;    \
    double _pivot = (double)(*((_type *)_pi));      \
    double _v;      \
    _out = 1;      \
    for (_ii = 0; _ii < _buffer_size; _ii++) {      \
        if (_offsets[_ii] < _offsets_flag) {      \
            _v = (double)(*((_type *)(_pi + _offsets[_ii])));      \
            if ((_mode == LOCAL_MIN && _v < _pivot) || (_v > _pivot)) {      \
                _out = 0;      \
            }      \
        }      \
        if (_out == NPY_FALSE) {      \
            break;      \
        }      \
    }    \
}    \
break


int is_local_q(PyArrayObject *input,
               PyArrayObject *strel,
               PyArrayObject *output,
               LOCAL_Mode mode,
               npy_intp *origins)
{
    Base_Iterator dptr_o, dptr_i, dptr_s;
    Neighborhood_Iterator dptr_of;
    char *pi = NULL, *po = NULL, *si = NULL;
    npy_bool *footprint, buffer;
    npy_intp nd, ii, offsets_stride, offsets_flag, *offsets, *of;

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

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    of = offsets;

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        buffer = NPY_TRUE;
        switch (PyArray_TYPE(input)) {
            CASE_ISLOCAL(NPY_BOOL, npy_bool,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_UBYTE, npy_ubyte,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_USHORT, npy_ushort,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_UINT, npy_uint,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_ULONG, npy_ulong,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_ULONGLONG, npy_ulonglong,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_BYTE, npy_byte,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_SHORT, npy_short,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_INT, npy_int,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_LONG, npy_long,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_LONGLONG, npy_longlong,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_FLOAT, npy_float,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            CASE_ISLOCAL(NPY_DOUBLE, npy_double,
                         pi, offsets_stride, of, mode, offsets_flag, buffer);
            default:
                NPY_END_THREADS;
                PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                goto exit;
            }
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

        BASE_ITERATOR_NEXT2(dptr_i, pi, dptr_o, po);
        NEIGHBORHOOD_ITERATOR_NEXT(dptr_of, of);
    }

    NPY_END_THREADS;

    exit:
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}


// #####################################################################################################################