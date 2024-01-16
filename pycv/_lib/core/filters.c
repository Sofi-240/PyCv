#include "ops_base.h"
#include "filters.h"

// #####################################################################################################################

#define TYPE_CASE_CONVOLVE(_NUM_TYPE, _type, _pi, _weights, _offsets_size, _offsets, _out)     \
case _NUM_TYPE:                                                                                \
{                                                                                              \
    npy_intp _ii;                                                                              \
    for (_ii = 0; _ii < _offsets_size; _ii++) {                                                \
        _out += _weights[_ii] * (double)(*((_type *)(_pi + _offsets[_ii])));                   \
    }                                                                                          \
}                                                                                              \
break

#define EX_CONVOLVE(_NUM_TYPE, _pi, _weights, _offsets_size, _offsets, _out)                                           \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        TYPE_CASE_CONVOLVE(NPY_BOOL, npy_bool, _pi, _weights, _offsets_size, _offsets, _out);                          \
        TYPE_CASE_CONVOLVE(NPY_UBYTE, npy_ubyte, _pi, _weights, _offsets_size, _offsets, _out);                        \
        TYPE_CASE_CONVOLVE(NPY_USHORT, npy_ushort, _pi, _weights, _offsets_size, _offsets, _out);                      \
        TYPE_CASE_CONVOLVE(NPY_UINT, npy_uint, _pi, _weights, _offsets_size, _offsets, _out);                          \
        TYPE_CASE_CONVOLVE(NPY_ULONG, npy_ulong, _pi, _weights, _offsets_size, _offsets, _out);                        \
        TYPE_CASE_CONVOLVE(NPY_ULONGLONG, npy_ulonglong, _pi, _weights, _offsets_size, _offsets, _out);                \
        TYPE_CASE_CONVOLVE(NPY_BYTE, npy_byte, _pi, _weights, _offsets_size, _offsets, _out);                          \
        TYPE_CASE_CONVOLVE(NPY_SHORT, npy_short, _pi, _weights, _offsets_size, _offsets, _out);                        \
        TYPE_CASE_CONVOLVE(NPY_INT, npy_int, _pi, _weights, _offsets_size, _offsets, _out);                            \
        TYPE_CASE_CONVOLVE(NPY_LONG, npy_long, _pi, _weights, _offsets_size, _offsets, _out);                          \
        TYPE_CASE_CONVOLVE(NPY_LONGLONG, npy_longlong, _pi, _weights, _offsets_size, _offsets, _out);                  \
        TYPE_CASE_CONVOLVE(NPY_FLOAT, npy_float, _pi, _weights, _offsets_size, _offsets, _out);                        \
        TYPE_CASE_CONVOLVE(NPY_DOUBLE, npy_double, _pi, _weights, _offsets_size, _offsets, _out);                      \
    }                                                                                                                  \
}

static int flip_kernel(PyArrayObject *kernel, double **kernel_out) {
    ArrayIter iter;
    npy_intp size, ii;
    char *pointer = NULL;
    double *k_run, tmp = 0.0;
    int num_type;

    size = PyArray_SIZE(kernel);
    num_type = PyArray_TYPE(kernel);

    *kernel_out = malloc(size * sizeof(double));;
    if (!*kernel_out) {
        PyErr_NoMemory();
        return 0;
    }
    k_run = *kernel_out;

    ArrayIterInit(kernel, &iter);
    pointer = (void *)PyArray_DATA(kernel);

    for (ii = size - 1; ii >= 0; ii--) {
        GET_VALUE_AS(num_type, double, pointer, tmp);
        *k_run++ = tmp;
        ARRAY_ITER_NEXT(iter, pointer);
    }
    return 1;
}

int ops_convolve(PyArrayObject *input, PyArrayObject *kernel, PyArrayObject *output, npy_intp *origins)
{
    ArrayIter iter_i, iter_o;
    char *po = NULL, *pi = NULL;
    npy_bool *footprint, *borders_lookup;
    int offsets_size, num_type_i, num_type_o;
    npy_intp ii, *offsets;
    double tmp, *kernel_flipted, *weights, *ww;

    NPY_BEGIN_THREADS_DEF;

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(kernel))) {
        PyErr_SetString(PyExc_RuntimeError, "kernel dtype not supported");
        goto exit;
    }

    if (!flip_kernel(kernel, &kernel_flipted)) {
        goto exit;
    }

    footprint = malloc(PyArray_SIZE(kernel) * sizeof(npy_bool));
    if (!footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    offsets_size = 0;
    for (ii = 0; ii < PyArray_SIZE(kernel); ii++) {
        if (kernel_flipted[ii]) {
            footprint[ii] = NPY_TRUE;
            offsets_size++;
        } else {
            footprint[ii] = NPY_FALSE;
        }
    }

    weights = malloc(offsets_size * sizeof(double));
    if (!weights) {
        PyErr_NoMemory();
        goto exit;
    }
    ww = weights;
    for (ii = 0; ii < PyArray_SIZE(kernel); ii++) {
        if (footprint[ii]) {
            *ww++ = kernel_flipted[ii];
        }
    }

    if (!init_offsets_ravel(input, PyArray_DIMS(kernel), origins, footprint, &offsets)) {
        goto exit;
    }

    if (!init_borders_lut(PyArray_NDIM(input), PyArray_DIMS(input), PyArray_DIMS(kernel), origins, &borders_lookup)) {
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        if (!borders_lookup[ii]) {
            tmp = 0.0;
            EX_CONVOLVE(num_type_i, pi, weights, offsets_size, offsets, tmp);
            SET_VALUE_TO(num_type_o, po, tmp);
            ARRAY_ITER_NEXT(iter_o, po);
        }
        ARRAY_ITER_NEXT(iter_i, pi);
    }
    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

// QUICK SELECT

static void qs_swap(double *i1, double *i2)
{
    double tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static npy_intp qs_partition(double *buffer, npy_intp low, npy_intp high)
{
    double pivot = buffer[high];
    npy_intp jj, ii = low;
    for (jj = low; jj < high; jj++) {
        if (buffer[jj] < pivot) {
            qs_swap(&buffer[ii], &buffer[jj]);
            ii++;
        }
    }
    qs_swap(&buffer[ii], &buffer[high]);
    return ii;
}

static double quick_select(double *buffer, npy_intp low, npy_intp high, int rank)
{
    npy_intp pivot_index = qs_partition(buffer, low, high);
    if (rank == pivot_index) {
        return buffer[pivot_index];
    } else if (rank - 1 < pivot_index) {
        return quick_select(buffer, low, pivot_index - 1, rank);
    } else {
        return quick_select(buffer, pivot_index + 1, high, rank);
    }
}

#define TYPE_CASE_SELECT(_NUM_TYPE, _type, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val)   \
case _NUM_TYPE:                                                                                       \
{                                                                                                     \
    npy_intp _ii;                                                                                     \
    for (_ii = 0; _ii < _offsets_size; _ii++) {                                                       \
        _buffer[_ii] = (double)(*((_type *)(_pi + _offsets[_ii])));                                   \
    }                                                                                                 \
    _rank_val = quick_select(_buffer, 0, _offsets_size - 1, _rank);                                   \
}                                                                                                     \
break

#define EX_SELECT(_NUM_TYPE, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val)                               \
{                                                                                                                   \
    switch (_NUM_TYPE) {                                                                                            \
        TYPE_CASE_SELECT(NPY_BOOL, npy_bool, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);              \
        TYPE_CASE_SELECT(NPY_UBYTE, npy_ubyte, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);            \
        TYPE_CASE_SELECT(NPY_USHORT, npy_ushort, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);          \
        TYPE_CASE_SELECT(NPY_UINT, npy_uint, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);              \
        TYPE_CASE_SELECT(NPY_ULONG, npy_ulong, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);            \
        TYPE_CASE_SELECT(NPY_ULONGLONG, npy_ulonglong, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);    \
        TYPE_CASE_SELECT(NPY_BYTE, npy_byte, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);              \
        TYPE_CASE_SELECT(NPY_SHORT, npy_short, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);            \
        TYPE_CASE_SELECT(NPY_INT, npy_int, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);                \
        TYPE_CASE_SELECT(NPY_LONG, npy_long, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);              \
        TYPE_CASE_SELECT(NPY_LONGLONG, npy_longlong, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);      \
        TYPE_CASE_SELECT(NPY_FLOAT, npy_float, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);            \
        TYPE_CASE_SELECT(NPY_DOUBLE, npy_double, _pi, _offsets_size, _offsets, _buffer, _rank, _rank_val);          \
    }                                                                                                               \
}

int ops_rank_filter(PyArrayObject *input,
                    PyArrayObject *footprint,
                    PyArrayObject *output,
                    int rank,
                    npy_intp *origins)
{
    ArrayIter iter_i, iter_o;
    char *po = NULL, *pi = NULL;
    npy_bool *footprint_arr, *borders_lookup;
    int offsets_size, num_type_i, num_type_o;
    npy_intp ii, *offsets;
    double rank_val, *buffer;

    NPY_BEGIN_THREADS_DEF;

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(footprint))) {
        PyErr_SetString(PyExc_RuntimeError, "footprint dtype not supported");
        goto exit;
    }

    if (!array_to_footprint(footprint, &footprint_arr, &offsets_size)) {
        goto exit;
    }

    if (!init_offsets_ravel(input, PyArray_DIMS(footprint), origins, footprint_arr, &offsets)) {
        goto exit;
    }

    if (!init_borders_lut(PyArray_NDIM(input), PyArray_DIMS(input), PyArray_DIMS(footprint), origins, &borders_lookup)) {
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    buffer = malloc(offsets_size * sizeof(double));
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
            EX_SELECT(num_type_i, pi, offsets_size, offsets, buffer, rank, rank_val);
            SET_VALUE_TO(num_type_o, po, rank_val);
            ARRAY_ITER_NEXT(iter_o, po);
        }
        ARRAY_ITER_NEXT(iter_i, pi);
    }
    NPY_END_THREADS;
    exit:
        free(buffer);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################


// #####################################################################################################################


