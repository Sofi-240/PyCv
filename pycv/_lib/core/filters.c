#include "ops_base.h"
#include "filters.h"

// #####################################################################################################################

static int flip_kernel(PyArrayObject *kernel, npy_double **kernel_out) {
    ArrayIter iter;
    npy_intp size, ii;
    char *pointer = NULL;
    npy_double *k_run, tmp = 0.0;
    int num_type;

    size = PyArray_SIZE(kernel);
    num_type = PyArray_TYPE(kernel);

    *kernel_out = malloc(size * sizeof(npy_double));
    if (!*kernel_out) {
        PyErr_NoMemory();
        return 0;
    }
    k_run = *kernel_out;

    ArrayIterInit(kernel, &iter);
    pointer = (void *)PyArray_DATA(kernel);

    for (ii = size - 1; ii >= 0; ii--) {
        GET_VALUE_AS(num_type, npy_double, pointer, tmp);
        k_run[ii] = tmp;
        ARRAY_ITER_NEXT(iter, pointer);
    }
    return 1;
}

#define TYPE_CASE_CONVOLVE(_NUM_TYPE, _type, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border)      \
case _NUM_TYPE:                                                                                                                                 \
{                                                                                                                                               \
    npy_intp _ii;                                                                                                                               \
    _out = 0;                                                                                                                                   \
    _is_border = 0;                                                                                                                             \
    for (_ii = 0; _ii < _offsets_size; _ii++) {                                                                                                 \
        if (_offsets[_ii] >= _offsets_flag) {                                                                                                   \
            if (!_use_border) {                                                                                                                 \
                _is_border = 1;                                                                                                                 \
                break;                                                                                                                          \
            }                                                                                                                                   \
            _out += _weights[_ii] * _c_val;                                                                                                     \
        } else {                                                                                                                                \
            _out += _weights[_ii] * (npy_double)(*((_type *)(_pi + _offsets[_ii])));                                                            \
        }                                                                                                                                       \
    }                                                                                                                                           \
}                                                                                                                                               \
break

#define EX_CONVOLVE(_NUM_TYPE, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border)                                           \
{                                                                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                                                                               \
        TYPE_CASE_CONVOLVE(NPY_BOOL, npy_bool, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                          \
        TYPE_CASE_CONVOLVE(NPY_UBYTE, npy_ubyte, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                        \
        TYPE_CASE_CONVOLVE(NPY_USHORT, npy_ushort, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                      \
        TYPE_CASE_CONVOLVE(NPY_UINT, npy_uint, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                          \
        TYPE_CASE_CONVOLVE(NPY_ULONG, npy_ulong, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                        \
        TYPE_CASE_CONVOLVE(NPY_ULONGLONG, npy_ulonglong, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                \
        TYPE_CASE_CONVOLVE(NPY_BYTE, npy_byte, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                          \
        TYPE_CASE_CONVOLVE(NPY_SHORT, npy_short, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                        \
        TYPE_CASE_CONVOLVE(NPY_INT, npy_int, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                            \
        TYPE_CASE_CONVOLVE(NPY_LONG, npy_long, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                          \
        TYPE_CASE_CONVOLVE(NPY_LONGLONG, npy_longlong, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                  \
        TYPE_CASE_CONVOLVE(NPY_FLOAT, npy_float, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                        \
        TYPE_CASE_CONVOLVE(NPY_DOUBLE, npy_double, _pi, _weights, _offsets_size, _offsets_flag, _offsets, _out, _c_val, _use_border, _is_border);                      \
    }                                                                                                                                                                  \
}

int ops_convolve(PyArrayObject *input,
                 PyArrayObject *kernel,
                 PyArrayObject *output,
                 npy_intp *origins,
                 BordersMode mode,
                 double constant_value)
{
    npy_intp array_size, kernel_size;
    int num_type_i, num_type_o, footprint_size;
    npy_intp *offsets, of_flag, *of_run;
    npy_bool *footprint;
    npy_intp ii, jj;

    ArrayIter iter_i, iter_o;
    FilterIter iter_f;
    char *po = NULL, *pi = NULL;
    npy_double tmp = 0.0, *kernel_flipped, *weights;
    int use_border, is_border = 0;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);
    kernel_size = PyArray_SIZE(kernel);
    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    if (!flip_kernel(kernel, &kernel_flipped)) {
        goto exit;
    }

    footprint = (npy_bool *)malloc(kernel_size * sizeof(npy_bool));
    if (!footprint) {
        PyErr_NoMemory();
        return 0;
    }

    footprint_size = 0;
    for (ii = 0; ii < kernel_size; ii++) {
        if (fabs(kernel_flipped[ii]) > DBL_EPSILON) {
            footprint_size++;
            footprint[ii] = 1;
        } else {
            footprint[ii] = 0;
        }
    }

    weights = (npy_double *)malloc(footprint_size * sizeof(npy_double));
    if (!weights) {
        PyErr_NoMemory();
        return 0;
    }
    jj = 0;
    for (ii = 0; ii < kernel_size; ii++) {
        if (footprint[ii]) {
            weights[jj] = kernel_flipped[ii];
            jj++;
        }
    }

    if (!init_filter_offsets(input, PyArray_DIMS(kernel), origins, footprint, &offsets, &of_flag, mode)) {
        goto exit;
    }

    use_border = mode == BORDER_VALID ? 0 : 1;

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);

    FilterIterInit(PyArray_NDIM(input), PyArray_DIMS(input), PyArray_DIMS(kernel), origins, footprint_size, &iter_f);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    of_run = offsets;

    for (ii = 0; ii < array_size; ii++) {
        EX_CONVOLVE(num_type_i, pi, weights, footprint_size, of_flag, of_run, tmp, constant_value, use_border, is_border);
        if (!is_border) {
            SET_VALUE_TO(num_type_o, po, tmp);
            ARRAY_ITER_NEXT(iter_o, po);
        }
        FILTER_ITER_NEXT(iter_f, of_run, iter_i, pi);
    }

    NPY_END_THREADS;

    exit:
        free(offsets);
        free(footprint);
        free(weights);
        free(kernel_flipped);
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

#define TYPE_CASE_SELECT(_NUM_TYPE, _type, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border)   \
case _NUM_TYPE:                                                                                                                                       \
{                                                                                                                                                     \
    npy_intp _ii;                                                                                                                                     \
    _is_border = 0;                                                                                                                                   \
    for (_ii = 0; _ii < _offsets_size; _ii++) {                                                                                                       \
        if (_offsets[_ii] >= _offsets_flag){                                                                                                          \
            if (_use_border) {                                                                                                                        \
                _buffer[_ii] = _c_val;                                                                                                                \
            } else {                                                                                                                                  \
                _is_border = 1;                                                                                                                       \
                break;                                                                                                                                \
            }                                                                                                                                         \
        } else {                                                                                                                                      \
            _buffer[_ii] = (double)(*((_type *)(_pi + _offsets[_ii])));                                                                               \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    if (!_is_border) {                                                                                                                                \
        _rank_val = quick_select(_buffer, 0, _offsets_size - 1, _rank);                                                                               \
    }                                                                                                                                                 \
}                                                                                                                                                     \
break

#define EX_SELECT(_NUM_TYPE, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border)                               \
{                                                                                                                                                                   \
    switch (_NUM_TYPE) {                                                                                                                                            \
        TYPE_CASE_SELECT(NPY_BOOL, npy_bool, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);              \
        TYPE_CASE_SELECT(NPY_UBYTE, npy_ubyte, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);            \
        TYPE_CASE_SELECT(NPY_USHORT, npy_ushort, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);          \
        TYPE_CASE_SELECT(NPY_UINT, npy_uint, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);              \
        TYPE_CASE_SELECT(NPY_ULONG, npy_ulong, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);            \
        TYPE_CASE_SELECT(NPY_ULONGLONG, npy_ulonglong, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);    \
        TYPE_CASE_SELECT(NPY_BYTE, npy_byte, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);              \
        TYPE_CASE_SELECT(NPY_SHORT, npy_short, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);            \
        TYPE_CASE_SELECT(NPY_INT, npy_int, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);                \
        TYPE_CASE_SELECT(NPY_LONG, npy_long, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);              \
        TYPE_CASE_SELECT(NPY_LONGLONG, npy_longlong, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);      \
        TYPE_CASE_SELECT(NPY_FLOAT, npy_float, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);            \
        TYPE_CASE_SELECT(NPY_DOUBLE, npy_double, _pi, _offsets_size, _offsets_flag, _offsets, _use_border, _c_val, _buffer, _rank, _rank_val, _is_border);          \
    }                                                                                                                                                               \
}

int ops_rank_filter(PyArrayObject *input,
                    PyArrayObject *footprint,
                    PyArrayObject *output,
                    int rank,
                    npy_intp *origins,
                    BordersMode mode,
                    double constant_value)
{
    ArrayIter iter_i, iter_o;
    FilterIter iter_f;
    char *po = NULL, *pi = NULL;
    npy_bool *footprint_arr;
    npy_intp *offsets, of_flag, *of_run;
    int num_type_i, num_type_o, footprint_size;

    npy_intp ii;
    double rank_val = 0.0, *buffer;
    int is_border = 0, use_border;

    NPY_BEGIN_THREADS_DEF;


    if (!array_to_footprint(footprint, &footprint_arr, &footprint_size)) {
        goto exit;
    }

    if (!init_filter_offsets(input, PyArray_DIMS(footprint), origins, footprint_arr, &offsets, &of_flag, mode)) {
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);
    FilterIterInit(PyArray_NDIM(input), PyArray_DIMS(input), PyArray_DIMS(footprint), origins, footprint_size, &iter_f);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    buffer = malloc(footprint_size * sizeof(double));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }
    use_border = mode == BORDER_VALID ? 0 : 1;

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    of_run = offsets;

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        EX_SELECT(num_type_i, pi, footprint_size, of_flag, of_run, use_border, constant_value, buffer, rank, rank_val, is_border);
        if (!is_border) {
            SET_VALUE_TO(num_type_o, po, rank_val);
            ARRAY_ITER_NEXT(iter_o, po);
        }
        FILTER_ITER_NEXT(iter_f, of_run, iter_i, pi);
    }
    NPY_END_THREADS;
    exit:
        free(buffer);
        free(offsets);
        free(footprint_arr);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################







