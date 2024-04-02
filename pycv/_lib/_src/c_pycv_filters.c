#include "c_pycv_base.h"
#include "c_pycv_filters.h"

// #####################################################################################################################

#define PYCV_F_CASE_CONVOLVE(_NTYPE, _dtype, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside)               \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _out = 0;                                                                                                          \
    for (_ii = 0; _ii < _n; _ii++) {                                                                                   \
        if (_offsets[_ii] == _flag) {                                                                                  \
            if (_mode == PYCV_EXTEND_VALID) {                                                                          \
                _outside = 1;                                                                                          \
                break;                                                                                                 \
            }                                                                                                          \
            _out += _h[_ii] * _c_val;                                                                                  \
        }                                                                                                              \
        else {                                                                                                         \
            _out += _h[_ii] * (npy_double)(*(_dtype *)(_x + _offsets[_ii]));                                           \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_F_CONVOLVE(_NTYPE, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside)                            \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_F_CASE_CONVOLVE(BOOL, npy_bool, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);              \
        PYCV_F_CASE_CONVOLVE(UBYTE, npy_ubyte, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);            \
        PYCV_F_CASE_CONVOLVE(USHORT, npy_ushort, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);          \
        PYCV_F_CASE_CONVOLVE(UINT, npy_uint, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);              \
        PYCV_F_CASE_CONVOLVE(ULONG, npy_ulong, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);            \
        PYCV_F_CASE_CONVOLVE(ULONGLONG, npy_ulonglong, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);    \
        PYCV_F_CASE_CONVOLVE(BYTE, npy_byte, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);              \
        PYCV_F_CASE_CONVOLVE(SHORT, npy_short, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);            \
        PYCV_F_CASE_CONVOLVE(INT, npy_int, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);                \
        PYCV_F_CASE_CONVOLVE(LONG, npy_long, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);              \
        PYCV_F_CASE_CONVOLVE(LONGLONG, npy_longlong, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);      \
        PYCV_F_CASE_CONVOLVE(FLOAT, npy_float, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);            \
        PYCV_F_CASE_CONVOLVE(DOUBLE, npy_double, _x, _h, _n, _flag, _offsets, _out, _mode, _c_val, _outside);          \
    }                                                                                                                  \
}

// #####################################################################################################################


int PYCV_convolve(PyArrayObject *input,
                  PyArrayObject *kernel,
                  PyArrayObject *output,
                  npy_intp *center,
                  PYCV_ExtendBorder mode,
                  npy_double c_val)
{
    npy_intp array_size;
    int num_type_i, num_type_o;
    npy_intp *offsets = NULL, flag, *ff;
    npy_bool *footprint = NULL;
    npy_intp ii, f_size;

    PYCV_ArrayIterator iter_o;
    NeighborhoodIterator iter_i;
    char *po = NULL, *pi = NULL;
    npy_double tmp = 0.0, *h = NULL;
    int outside;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);

    if (!PYCV_AllocateKernelFlip(kernel, &footprint, &h)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_ToConvolveKernel \n");
        goto exit;
    }

    PYCV_FOOTPRINT_NONZERO(footprint, PyArray_SIZE(kernel), f_size);

    PYCV_NeighborhoodIteratorInit(input, PyArray_DIMS(kernel), center, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input, PyArray_DIMS(kernel), center, footprint, &offsets, NULL, &flag, mode)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }
    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        outside = 0;
        PYCV_F_CONVOLVE(num_type_i, pi, h, f_size, flag, ff, tmp, mode, c_val, outside);
        if (!outside) {
            PYCV_SET_VALUE_F2A(num_type_o, po, tmp);
            PYCV_ARRAY_ITERATOR_NEXT(iter_o, po);
        }
        PYCV_NEIGHBORHOOD_ITERATOR_NEXT(iter_i, pi, ff);
    }

    NPY_END_THREADS;

    exit:
        free(offsets);
        free(footprint);
        free(h);
        return PyErr_Occurred() ? 0 : 1;
}


// #####################################################################################################################

static void QS_swap(npy_double *i1, npy_double *i2)
{
    npy_double tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static npy_double QuickSelect(npy_double *buffer, npy_intp low, npy_intp high, npy_intp rank)
{
    npy_intp ii, jj;
    npy_double v;
    while (low <= high) {
        ii = low;
        jj = high - 1;
        v = *(buffer + high);
        while (ii <= jj) {
            if (*(buffer + ii) > v && *(buffer + jj) < v) {
                QS_swap((buffer + ii), (buffer + jj));
            }
            if (*(buffer + ii) <= v) {
                ii++;
            }
            if (*(buffer + jj) >= v) {
                jj--;
            }
        }
        QS_swap((buffer + ii), (buffer + high));
        if (ii == rank - 1) {
            return *(buffer + ii);
        } else if (rank - 1 < ii) {
            high = ii - 1;
        } else {
            low = ii + 1;
        }
    }
    return *(buffer + ii);
}

#define PYCV_F_CASE_SELECT(_NTYPE, _dtype, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside)          \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _out = 0;                                                                                                          \
    for (_ii = 0; _ii < _n; _ii++) {                                                                                   \
        if (_offsets[_ii] == _flag) {                                                                                  \
            if (_mode == PYCV_EXTEND_VALID) {                                                                          \
                _outside = 1;                                                                                          \
                break;                                                                                                 \
            }                                                                                                          \
            _h[_ii] = _c_val;                                                                                          \
        }                                                                                                              \
        else {                                                                                                         \
            _h[_ii] = (npy_double)(*(_dtype *)(_x + _offsets[_ii]));                                                   \
        }                                                                                                              \
    }                                                                                                                  \
    if (!_outside) {                                                                                                   \
        _out = QuickSelect(_h, 0, _n - 1, _rank);                                                                      \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_P_SELECT(_NTYPE, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside)                       \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_F_CASE_SELECT(BOOL, npy_bool, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);         \
        PYCV_F_CASE_SELECT(UBYTE, npy_ubyte, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);       \
        PYCV_F_CASE_SELECT(USHORT, npy_ushort, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);     \
        PYCV_F_CASE_SELECT(UINT, npy_uint, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);         \
        PYCV_F_CASE_SELECT(ULONG, npy_ulong, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);       \
        PYCV_F_CASE_SELECT(ULONGLONG, npy_ulonglong, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);\
        PYCV_F_CASE_SELECT(BYTE, npy_byte, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);         \
        PYCV_F_CASE_SELECT(SHORT, npy_short, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);       \
        PYCV_F_CASE_SELECT(INT, npy_int, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);           \
        PYCV_F_CASE_SELECT(LONG, npy_long, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);         \
        PYCV_F_CASE_SELECT(LONGLONG, npy_longlong, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside); \
        PYCV_F_CASE_SELECT(FLOAT, npy_float, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);       \
        PYCV_F_CASE_SELECT(DOUBLE, npy_double, _x, _n, _flag, _offsets, _h, _rank, _out, _mode, _c_val, _outside);     \
    }                                                                                                                  \
}


int PYCV_rank_filter(PyArrayObject *input,
                     PyArrayObject *footprint,
                     PyArrayObject *output,
                     npy_intp rank,
                     npy_intp *center,
                     PYCV_ExtendBorder mode,
                     npy_double c_val)
{
    npy_intp array_size, f_size;
    int num_type_i, num_type_o;
    npy_intp *offsets = NULL, flag, *ff = NULL;
    npy_bool *fp = NULL;
    npy_intp ii;

    PYCV_ArrayIterator iter_o;
    NeighborhoodIterator iter_i;
    char *po = NULL, *pi = NULL;
    npy_double tmp = 0.0, *h = NULL;
    int outside = 0;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);

    if (!PYCV_AllocateToFootprint(footprint, &fp, &f_size, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_AllocateToFootprint \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(input, PyArray_DIMS(footprint), center, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input, PyArray_DIMS(footprint), center, fp, &offsets, NULL, &flag, mode)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    h = malloc(f_size * sizeof(npy_double));
    if (!h) {
        PyErr_NoMemory();
        goto exit;
    }

    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        outside = 0;
        PYCV_P_SELECT(num_type_i, pi, f_size, flag, ff, h, rank, tmp, mode, c_val, outside);
        if (!outside) {
            PYCV_SET_VALUE_F2A(num_type_o, po, tmp);
            PYCV_ARRAY_ITERATOR_NEXT(iter_o, po);
        }
        PYCV_NEIGHBORHOOD_ITERATOR_NEXT(iter_i, pi, ff);
    }

    NPY_END_THREADS;

    exit:
        free(offsets);
        free(fp);
        free(h);
        return PyErr_Occurred() ? 0 : 1;
}


// #####################################################################################################################

















