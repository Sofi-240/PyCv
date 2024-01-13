#include "ops_support.h"
#include "filters.h"


// #####################################################################################################################

#define TYPE_CASE_CONVOLVE(_TYPE, _type, _pi, _weights, _buffer_size, _offsets, _buffer_val)   \
case _TYPE:                                                                                    \
{                                                                                              \
    npy_intp _ii;                                                                              \
    for (_ii = 0; _ii < _buffer_size; _ii++) {                                                 \
        _buffer_val += _weights[_ii] * (double)(*((_type *)(_pi + _offsets[_ii])));            \
    }                                                                                          \
}                                                                                              \
break

int ops_convolve(PyArrayObject *input, PyArrayObject *kernel, PyArrayObject *output, npy_intp *origins)
{
    Base_Iterator dptr_o, dptr_i;
    char *pi = NULL, *po = NULL;
    npy_bool *footprint, *borders_lookup;
    int footprint_size;
    npy_intp ii, *offsets;
    double buffer, *weights;

    NPY_BEGIN_THREADS_DEF;

    if (!check_dtype(PyArray_TYPE(input)) || !check_dtype(PyArray_TYPE(output)) || !check_dtype(PyArray_TYPE(kernel))) {
        goto exit;
    }

    if (!INIT_FOOTPRINT(kernel, &footprint, &footprint_size)) {
        goto exit;
    }

    if (!INIT_OFFSETS_WITH_BORDERS(input, PyArray_DIMS(kernel), origins, footprint, &offsets, &borders_lookup)) {
        goto exit;
    }

    if (!COPY_DATA_TO_DOUBLE(kernel, &weights, footprint)) {
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

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        if (!borders_lookup[ii]) {
            buffer = 0.0;
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_CONVOLVE(NPY_BOOL, npy_bool,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_UBYTE, npy_ubyte,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_USHORT, npy_ushort,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_UINT, npy_uint,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_ULONG, npy_ulong,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_ULONGLONG, npy_ulonglong,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_BYTE, npy_byte,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_SHORT, npy_short,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_INT, npy_int,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_LONG, npy_long,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_LONGLONG, npy_longlong,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_FLOAT, npy_float,
                                   pi, weights, footprint_size, offsets, buffer);
                TYPE_CASE_CONVOLVE(NPY_DOUBLE, npy_double,
                                   pi, weights, footprint_size, offsets, buffer);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
            SET_VALUE_FROM_DOUBLE(PyArray_TYPE(output), po, buffer);
//            switch (PyArray_TYPE(output)) {
//                TYPE_CASE_VALUE_OUT_F2U(NPY_BOOL, npy_bool, po, buffer);
//                TYPE_CASE_VALUE_OUT_F2U(NPY_UBYTE, npy_ubyte, po, buffer);
//                TYPE_CASE_VALUE_OUT_F2U(NPY_USHORT, npy_ushort, po, buffer);
//                TYPE_CASE_VALUE_OUT_F2U(NPY_UINT, npy_uint, po, buffer);
//                TYPE_CASE_VALUE_OUT_F2U(NPY_ULONG, npy_ulong, po, buffer);
//                TYPE_CASE_VALUE_OUT_F2U(NPY_ULONGLONG, npy_ulonglong, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, buffer);
//                TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, buffer);
//                default:
//                    NPY_END_THREADS;
//                    PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
//                    goto exit;
//            }
            BASE_ITERATOR_NEXT(dptr_o, po);
        }
        BASE_ITERATOR_NEXT(dptr_i, pi);
    }
    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

static void SWAP(double *i1, double *i2)
{
    double tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static npy_intp PARTITION(double *buffer, npy_intp low, npy_intp high)
{
    double pivot = buffer[high];
    npy_intp jj, ii = low;
    for (jj = low; jj < high; jj++) {
        if (buffer[jj] < pivot) {
            SWAP(&buffer[ii], &buffer[jj]);
            ii++;
        }
    }
    SWAP(&buffer[ii], &buffer[high]);
    return ii;
}

static double QUICK_SELECT(double *buffer, npy_intp low, npy_intp high, int rank)
{
    npy_intp pivot_index = PARTITION(buffer, low, high);
    if (rank == pivot_index) {
        return buffer[pivot_index];
    } else if (rank - 1 < pivot_index) {
        return QUICK_SELECT(buffer, low, pivot_index - 1, rank);
    } else {
        return QUICK_SELECT(buffer, pivot_index + 1, high, rank);
    }
}

#define TYPE_CASE_SELECT(_TYPE, _type, _pi, _buffer_size, _offsets, _buffer, _rank, _rank_val)   \
case _TYPE:    \
{    \
    npy_intp _ii;    \
    for (_ii = 0; _ii < _buffer_size; _ii++) {    \
        _buffer[_ii] = (double)(*((_type *)(_pi + _offsets[_ii])));    \
    }    \
    _rank_val = QUICK_SELECT(_buffer, 0, _buffer_size - 1, _rank);     \
}    \
break

int ops_rank_filter(PyArrayObject *input,
                    PyArrayObject *footprint,
                    PyArrayObject *output,
                    int rank,
                    npy_intp *origins)
{
    Base_Iterator dptr_o, dptr_i;
    char *pi = NULL, *po = NULL;
    npy_bool *borders_lookup, *fo = NULL;
    npy_intp footprint_size = 0;
    npy_intp ii, jj, kernel_size, *offsets;
    double rank_val, *buffer = NULL;

    NPY_BEGIN_THREADS_DEF;

    kernel_size = PyArray_SIZE(footprint);
    fo = (npy_bool *)PyArray_DATA(footprint);

    for (ii = 0; ii < kernel_size; ii++) {
        if (fo[ii]) {
            footprint_size++;
        }
    }

    if (!INIT_OFFSETS_WITH_BORDERS(input, PyArray_DIMS(footprint), origins, fo, &offsets, &borders_lookup)) {
        goto exit;
    }

    if (!INIT_Base_Iterator(input, &dptr_i)){
        goto exit;
    }

    if (!INIT_Base_Iterator(output, &dptr_o)){
        goto exit;
    }

    buffer = malloc(footprint_size * sizeof(double));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        if (!borders_lookup[ii]) {
            rank_val = 0.0;
            switch (PyArray_TYPE(input)) {
                TYPE_CASE_SELECT(NPY_BOOL, npy_bool,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_UBYTE, npy_ubyte,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_USHORT, npy_ushort,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_UINT, npy_uint,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_ULONG, npy_ulong,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_ULONGLONG, npy_ulonglong,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_BYTE, npy_byte,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_SHORT, npy_short,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_INT, npy_int,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_LONG, npy_long,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_LONGLONG, npy_longlong,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_FLOAT, npy_float,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                TYPE_CASE_SELECT(NPY_DOUBLE, npy_double,
                                 pi, footprint_size, offsets, buffer, rank, rank_val);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
                    goto exit;
            }
            switch (PyArray_TYPE(output)) {
                TYPE_CASE_VALUE_OUT_F2U(NPY_BOOL, npy_bool, po, rank_val);
                TYPE_CASE_VALUE_OUT_F2U(NPY_UBYTE, npy_ubyte, po, rank_val);
                TYPE_CASE_VALUE_OUT_F2U(NPY_USHORT, npy_ushort, po, rank_val);
                TYPE_CASE_VALUE_OUT_F2U(NPY_UINT, npy_uint, po, rank_val);
                TYPE_CASE_VALUE_OUT_F2U(NPY_ULONG, npy_ulong, po, rank_val);
                TYPE_CASE_VALUE_OUT_F2U(NPY_ULONGLONG, npy_ulonglong, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, po, rank_val);
                TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, po, rank_val);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
                    goto exit;
            }
            BASE_ITERATOR_NEXT(dptr_o, po);
        }
        BASE_ITERATOR_NEXT(dptr_i, pi);
    }
    NPY_END_THREADS;
    exit:
        free(offsets);
        free(borders_lookup);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################