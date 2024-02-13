#include "c_pycv_base.h"
#include "c_pycv_morphology.h"

// #####################################################################################################################

#define PYCV_M_CASE_MASK_VALUE(_NTYPE, _dtype, _ma, _out)                                                              \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    _out = *(_dtype *)_ma ? 1 : 0;                                                                                     \
}                                                                                                                      \
break

#define PYCV_M_MASK_VALUE(_NTYPE, _ma, _out)                                                                           \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_M_CASE_MASK_VALUE(BOOL, npy_bool, _ma, _out);                                                             \
        PYCV_M_CASE_MASK_VALUE(UBYTE, npy_ubyte, _ma, _out);                                                           \
        PYCV_M_CASE_MASK_VALUE(USHORT, npy_ushort, _ma, _out);                                                         \
        PYCV_M_CASE_MASK_VALUE(UINT, npy_uint, _ma, _out);                                                             \
        PYCV_M_CASE_MASK_VALUE(ULONG, npy_ulong, _ma, _out);                                                           \
        PYCV_M_CASE_MASK_VALUE(ULONGLONG, npy_ulonglong, _ma, _out);                                                   \
        PYCV_M_CASE_MASK_VALUE(BYTE, npy_byte, _ma, _out);                                                             \
        PYCV_M_CASE_MASK_VALUE(SHORT, npy_short, _ma, _out);                                                           \
        PYCV_M_CASE_MASK_VALUE(INT, npy_int, _ma, _out);                                                               \
        PYCV_M_CASE_MASK_VALUE(LONG, npy_long, _ma, _out);                                                             \
        PYCV_M_CASE_MASK_VALUE(LONGLONG, npy_longlong, _ma, _out);                                                     \
        PYCV_M_CASE_MASK_VALUE(FLOAT, npy_float, _ma, _out);                                                           \
        PYCV_M_CASE_MASK_VALUE(DOUBLE, npy_double, _ma, _out);                                                         \
    }                                                                                                                  \
}

// #####################################################################################################################

#define PYCV_M_CASE_BINARY_EROSION(_NTYPE, _dtype, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val)     \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _out = *(_dtype *)_x ? _t_val : _f_val;                                                                            \
    if (_ma_val) {                                                                                                     \
        for (_ii = 0; _ii < _n; _ii++) {                                                                               \
            if (_offsets[_ii] == _flag) {                                                                              \
                _out = _c_val ? _t_val : _f_val;                                                                       \
            } else {                                                                                                   \
                _out = *(_dtype *)(_x + _offsets[_ii]) ? _t_val : _f_val;                                              \
            }                                                                                                          \
            if (!_out) {                                                                                               \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_M_BINARY_EROSION(_NTYPE, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val)                  \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_M_CASE_BINARY_EROSION(BOOL, npy_bool, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);    \
        PYCV_M_CASE_BINARY_EROSION(UBYTE, npy_ubyte, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);  \
        PYCV_M_CASE_BINARY_EROSION(USHORT, npy_ushort, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);\
        PYCV_M_CASE_BINARY_EROSION(UINT, npy_uint, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);    \
        PYCV_M_CASE_BINARY_EROSION(ULONG, npy_ulong, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);  \
        PYCV_M_CASE_BINARY_EROSION(ULONGLONG, npy_ulonglong, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);\
        PYCV_M_CASE_BINARY_EROSION(BYTE, npy_byte, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);    \
        PYCV_M_CASE_BINARY_EROSION(SHORT, npy_short, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);  \
        PYCV_M_CASE_BINARY_EROSION(INT, npy_int, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);      \
        PYCV_M_CASE_BINARY_EROSION(LONG, npy_long, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);    \
        PYCV_M_CASE_BINARY_EROSION(LONGLONG, npy_longlong, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);\
        PYCV_M_CASE_BINARY_EROSION(FLOAT, npy_float, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val);  \
        PYCV_M_CASE_BINARY_EROSION(DOUBLE, npy_double, _ma_val, _x, _n, _flag, _offsets, _out, _c_val, _t_val, _f_val); \
    }                                                                                                                  \
}

// #####################################################################################################################

int PYCV_binary_erosion(PyArrayObject *input,
                        PyArrayObject *strel,
                        PyArrayObject *output,
                        npy_intp *center,
                        int iterations,
                        PyArrayObject *mask,
                        PYCV_MorphOP op,
                        int c_val)
{
    PYCV_ArrayIterator iter_o, iter_m;
    NeighborhoodIterator iter_i;
    char *po = NULL, *pi = NULL, *ma = NULL;
    npy_bool *footprint;
    int num_type_i, num_type_o, num_type_m, out_val, ma_val = 1, op_true = 1, op_false = 0;
    npy_intp array_size, ii, flag, f_size, *offsets, *ff;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);

    if (!PYCV_AllocateToFootprint(strel, &footprint, &f_size)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_AllocateToFootprint \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(input, PyArray_DIMS(strel), center, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input, PyArray_DIMS(strel), center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }
    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);
    if (mask) {
        num_type_m = PyArray_TYPE(mask);
        PYCV_ArrayIteratorInit(mask, &iter_m);
    }

    if (op == PYCV_MORPH_OP_DIL) {
        op_true = 0;
        op_false = 1;
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        if (mask) {
            PYCV_M_MASK_VALUE(num_type_m, ma, ma_val);
        }
        PYCV_M_BINARY_EROSION(num_type_i, ma_val, pi, f_size, flag, ff, out_val, c_val, op_true, op_false);
        if (!op_true) {
            out_val = out_val ? 0 : 1;
        }
        PYCV_SET_VALUE(num_type_o, po, out_val);
        if (mask) {
            PYCV_NEIGHBORHOOD_ITERATOR_NEXT3(iter_i, pi, iter_o, po, iter_m, ma, ff);
        } else {
            PYCV_NEIGHBORHOOD_ITERATOR_NEXT2(iter_i, pi, iter_o, po, ff);
        }
    }
    NPY_END_THREADS;
    exit:
        free(offsets);
        free(footprint);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################


#define PYCV_M_CASE_GRAY_EROSION_DILATION(_NTYPE, _dtype, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val)     \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    npy_double _tmp;                                                                                                   \
    if (_ma_val) {                                                                                                     \
        _out = _offsets[0] == _flag ? _c_val : (npy_double)(*(_dtype *)(_x + _offsets[0]));                            \
        _out += _h[0];                                                                                                 \
        for (_ii = 1; _ii < _n; _ii++) {                                                                               \
            if (_offsets[_ii] == _flag) {                                                                              \
                _tmp = _c_val + _h[_ii];                                                                               \
            } else {                                                                                                   \
                _tmp = (npy_double)(*(_dtype *)(_x + _offsets[_ii])) + _h[_ii];                                        \
            }                                                                                                          \
            switch (_op) {                                                                                             \
                case PYCV_MORPH_OP_ERO:                                                                                \
                    _out = _out < _tmp ? _out : _tmp;                                                                  \
                    break;                                                                                             \
                case PYCV_MORPH_OP_DIL:                                                                                \
                    _out = _out > _tmp ? _out : _tmp;                                                                  \
                    break;                                                                                             \
            }                                                                                                          \
        }                                                                                                              \
    } else {                                                                                                           \
        _out = (npy_double)(*(_dtype *)_x);                                                                            \
    }                                                                                                                  \
}                                                                                                                      \
break


#define PYCV_M_GRAY_EROSION_DILATION(_NTYPE, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val)                  \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_M_CASE_GRAY_EROSION_DILATION(BOOL, npy_bool, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);    \
        PYCV_M_CASE_GRAY_EROSION_DILATION(UBYTE, npy_ubyte, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);  \
        PYCV_M_CASE_GRAY_EROSION_DILATION(USHORT, npy_ushort, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);\
        PYCV_M_CASE_GRAY_EROSION_DILATION(UINT, npy_uint, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);    \
        PYCV_M_CASE_GRAY_EROSION_DILATION(ULONG, npy_ulong, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);  \
        PYCV_M_CASE_GRAY_EROSION_DILATION(ULONGLONG, npy_ulonglong, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);\
        PYCV_M_CASE_GRAY_EROSION_DILATION(BYTE, npy_byte, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);    \
        PYCV_M_CASE_GRAY_EROSION_DILATION(SHORT, npy_short, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);  \
        PYCV_M_CASE_GRAY_EROSION_DILATION(INT, npy_int, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);      \
        PYCV_M_CASE_GRAY_EROSION_DILATION(LONG, npy_long, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);    \
        PYCV_M_CASE_GRAY_EROSION_DILATION(LONGLONG, npy_longlong, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);\
        PYCV_M_CASE_GRAY_EROSION_DILATION(FLOAT, npy_float, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);  \
        PYCV_M_CASE_GRAY_EROSION_DILATION(DOUBLE, npy_double, _op, _ma_val, _x, _h, _n, _flag, _offsets, _out, _c_val);\
    }                                                                                                                  \
}


int PYCV_gray_erosion_dilation(PyArrayObject *input,
                               PyArrayObject *flat_strel,
                               PyArrayObject *non_flat_strel,
                               PyArrayObject *output,
                               npy_intp *center,
                               PyArrayObject *mask,
                               PYCV_MorphOP op,
                               npy_double c_val)
{
    PYCV_ArrayIterator iter_o, iter_m, iter_s;
    NeighborhoodIterator iter_i;
    char *po = NULL, *pi = NULL, *ma = NULL, *ps = NULL;
    npy_bool *footprint = NULL;
    npy_double *h = NULL, out_val;
    int num_type_i, num_type_o, num_type_m, num_type_s, ma_val = 1;
    npy_intp array_size, ii, jj, flag, f_size, *offsets, *ff;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);

    if (!PYCV_AllocateKernelFlip(flat_strel, &footprint, NULL)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_ToConvolveKernel \n");
        goto exit;
    }

    PYCV_FOOTPRINT_NONZERO(footprint, PyArray_SIZE(flat_strel), f_size);

    h = malloc(f_size * sizeof(npy_double));
    if (!h) {
        PyErr_NoMemory();
        goto exit;
    }
    for (ii = 0; ii < f_size; ii++) {
        h[ii] = 0;
    }
    if (non_flat_strel) {
        PYCV_ArrayIteratorInit(non_flat_strel, &iter_s);
        num_type_s = PyArray_TYPE(non_flat_strel);
        ps = (void *)PyArray_DATA(non_flat_strel);

        jj = f_size - 1;
        for (ii = PyArray_SIZE(non_flat_strel) - 1; ii >= 0; ii--) {
            if (footprint[ii]) {
                PYCV_GET_VALUE(num_type_s, npy_double, ps, out_val);
                out_val = op == PYCV_MORPH_OP_ERO ? -out_val : out_val;
                h[jj] = out_val;
                jj--;
            }
            PYCV_ARRAY_ITERATOR_NEXT(iter_s, ps);
        }
    }

    PYCV_NeighborhoodIteratorInit(input, PyArray_DIMS(flat_strel), center, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input, PyArray_DIMS(flat_strel), center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    if (mask) {
        num_type_m = PyArray_TYPE(mask);
        PYCV_ArrayIteratorInit(mask, &iter_m);
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        if (mask) {
            PYCV_M_MASK_VALUE(num_type_m, ma, ma_val);
        }
        PYCV_M_GRAY_EROSION_DILATION(num_type_i, op, ma_val, pi, h, f_size, flag, ff, out_val, c_val);
        PYCV_SET_VALUE_F2A(num_type_o, po, out_val);

        if (mask) {
            PYCV_NEIGHBORHOOD_ITERATOR_NEXT3(iter_i, pi, iter_o, po, iter_m, ma, ff);
        } else {
            PYCV_NEIGHBORHOOD_ITERATOR_NEXT2(iter_i, pi, iter_o, po, ff);
        }
    }
    NPY_END_THREADS;
    exit:
        free(offsets);
        free(h);
        free(footprint);
        return PyErr_Occurred() ? 0 : 1;
}


// #####################################################################################################################

#define PYCV_M_CASE_FILL_IF_ZERO(_NTYPE, _dtype, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn)                  \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_ushort _outside, _visited;                                                                                     \
    npy_intp _ii, _jj, _ndim;                                                                                          \
    _ndim = (_iterator).nd_m1 + 1;                                                                                     \
    for (_ii = 0; _ii < _of_n; _ii++) {                                                                                \
        if (_of_r[_ii] == 0) {                                                                                         \
            continue;                                                                                                  \
        }                                                                                                              \
        _outside = 0;                                                                                                  \
        for (_jj = 0; _jj < _ndim; _jj++) {                                                                            \
            _pn[_jj] = _p0[_jj] + _of_ur[_ii * _ndim + _jj];                                                           \
            if (_pn[_jj] < 0 || _pn[_jj] > (_iterator).dims_m1[_jj]) {                                                 \
                _outside = 1;                                                                                          \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        if (!_outside) {                                                                                               \
            _visited = *(_dtype *)(_pi + _of_r[_ii]) ? 1 : 0;                                                          \
            if (!_visited) {                                                                                           \
                *(_dtype *)(_pi + _of_r[_ii]) = 1;                                                                     \
                _nn++;                                                                                                 \
                _pn += _ndim;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_M_FILL_IF_ZERO(_NTYPE, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn)                               \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_M_CASE_FILL_IF_ZERO(BOOL, npy_bool, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);                 \
        PYCV_M_CASE_FILL_IF_ZERO(UBYTE, npy_ubyte, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);               \
        PYCV_M_CASE_FILL_IF_ZERO(USHORT, npy_ushort, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);             \
        PYCV_M_CASE_FILL_IF_ZERO(UINT, npy_uint, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);                 \
        PYCV_M_CASE_FILL_IF_ZERO(ULONG, npy_ulong, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);               \
        PYCV_M_CASE_FILL_IF_ZERO(ULONGLONG, npy_ulonglong, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);       \
        PYCV_M_CASE_FILL_IF_ZERO(BYTE, npy_byte, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);                 \
        PYCV_M_CASE_FILL_IF_ZERO(SHORT, npy_short, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);               \
        PYCV_M_CASE_FILL_IF_ZERO(INT, npy_int, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);                   \
        PYCV_M_CASE_FILL_IF_ZERO(LONG, npy_long, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);                 \
        PYCV_M_CASE_FILL_IF_ZERO(LONGLONG, npy_longlong, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);         \
        PYCV_M_CASE_FILL_IF_ZERO(FLOAT, npy_float, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);               \
        PYCV_M_CASE_FILL_IF_ZERO(DOUBLE, npy_double, _iterator, _pi, _of_n, _of_r, _of_ur, _nn, _p0, _pn);             \
    }                                                                                                                  \
}

int PYCV_binary_region_fill(PyArrayObject *output,
                            npy_intp *seed_point,
                            PyArrayObject *strel,
                            npy_intp *center)
{
    PYCV_ArrayIterator iter_o;
    char *po = NULL, *po_base = NULL;
    npy_bool *footprint;
    int num_type_o;
    npy_intp ndim, array_size, ii, f_size, *offsets_r, *offsets_ur;
    npy_intp n0 = 0, nn = 0, *p, *p0, *pn;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(output);

    if (!PYCV_AllocateToFootprint(strel, &footprint, &f_size)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_AllocateToFootprint \n");
        goto exit;
    }

    PYCV_ArrayIteratorInit(output, &iter_o);

    if (!PYCV_InitOffsets(output, PyArray_DIMS(strel), center, footprint, &offsets_r, &offsets_ur)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitOffsets \n");
        goto exit;
    }

    ndim = iter_o.nd_m1 + 1;

    p = malloc((array_size + 1) * ndim * sizeof(npy_intp));
    if (!p) {
        PyErr_NoMemory();
        goto exit;
    }
    pn = p0 = p;
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    po_base = po = (void *)PyArray_DATA(output);

    for (ii = 0; ii < ndim; ii++) {
        *pn++ = seed_point[ii];
    }
    nn++;

    PYCV_ARRAY_ITERATOR_GOTO(iter_o, po_base, po, p0);
    PYCV_SET_VALUE(num_type_o, po, 1);

    while (n0 < nn) {
        PYCV_ARRAY_ITERATOR_GOTO(iter_o, po_base, po, p0);
        PYCV_M_FILL_IF_ZERO(num_type_o, iter_o, po, f_size, offsets_r, offsets_ur, nn, p0, pn);
        n0++;
        p0 += ndim;
    }
    NPY_END_THREADS;

    exit:
        free(p);
        free(offsets_r);
        free(offsets_ur);
        free(footprint);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

int PYCV_labeling(PyArrayObject *input,
                  npy_intp connectivity,
                  PyArrayObject *output,
                  int label_by_index)
{
    PYCV_ArrayIterator iter_o;
    NeighborhoodIterator iter_i;
    char *po = NULL, *pi = NULL;
    int num_type_o, num_type_i;
    npy_intp itemsize_i;
    npy_bool *footprint;
    npy_intp ndim, array_size, f_size, f_shape[NPY_MAXDIMS], f_center[NPY_MAXDIMS], *offsets, *ff, flag, offset_p;
    npy_intp *buffer, *parent, *labels, ii, jj, b_size, ln, lc, n_labels = 1;
    npy_double vi, vn;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);
    ndim = (npy_intp)PyArray_NDIM(input);
    itemsize_i = (npy_intp)PyArray_ITEMSIZE(input);

    for (ii = 0; ii < ndim; ii++) {
        f_shape[ii] = 3;
        f_center[ii] = 1;
    }

    if (!PYCV_DefaultFootprint(ndim, connectivity, &footprint, &f_size, 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_DefaultFootprint \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(input, f_shape, f_center, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input, f_shape, f_center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }


    buffer = malloc(f_size * sizeof(npy_intp));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    parent = malloc(array_size * sizeof(npy_intp));
    if (!parent) {
        PyErr_NoMemory();
        goto exit;
    }

    if (!label_by_index) {
        labels = malloc(array_size * sizeof(npy_intp));
        if (!labels) {
            PyErr_NoMemory();
            goto exit;
        }
    }

    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        parent[ii] = 0;
        if (!label_by_index) {
            labels[ii] = 0;
        }
        PYCV_GET_VALUE(num_type_i, npy_double, pi, vi);

        if (vi) {
            parent[ii] = ii;
            b_size = 0;
            for (jj = 0; jj < f_size; jj++) {
                if (ff[jj] == flag) {
                    continue;
                }
                offset_p = ii + (npy_intp)(ff[jj] / itemsize_i);
                PYCV_GET_VALUE(num_type_i, npy_double, (pi + ff[jj]), vn);
                if (!parent[offset_p] || vi != vn) {
                    continue;
                }
                parent[ii] = parent[ii] > parent[offset_p] ? parent[offset_p] : parent[ii];
                buffer[b_size] = offset_p;
                b_size++;
            }
            for (jj = 0; jj < b_size; jj++) {
                ln = buffer[jj];
                while (parent[ln] != parent[ii]) {
                    lc = parent[ln];
                    parent[ln] = parent[ii];
                    ln = lc;
                }
            }
        }
        PYCV_NEIGHBORHOOD_ITERATOR_NEXT(iter_i, pi, ff);
    }

    for (ii = 0; ii < array_size; ii++) {
        if (parent[ii]) {
            if (label_by_index) {
                PYCV_SET_VALUE(num_type_o, po, parent[ii]);
            } else {
                ln = parent[ii];
                if (!labels[ln]) {
                    labels[ln] = n_labels;
                    n_labels++;
                }
                PYCV_SET_VALUE(num_type_o, po, labels[ln]);
            }
        } else {
            PYCV_SET_VALUE(num_type_o, po, 0);
        }
        PYCV_ARRAY_ITERATOR_NEXT(iter_o, po);
    }

    NPY_END_THREADS;
    exit:
        free(footprint);
        free(offsets);
        free(buffer);
        free(parent);
        if (!label_by_index) {
            free(labels);
        }
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

static int PYCV_Uint8BinaryTable(unsigned int **table)
{
    const unsigned int change_point[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    unsigned int *tbl;
    unsigned int value, counter;
    int ii, jj;

    *table = calloc(256 * 8, sizeof(unsigned int));
    if (!*table) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < 8; ii++) {
        value = 0;
        counter = change_point[ii];
        tbl = *table + ii;
        for (jj = 0; jj < 256; jj++) {
            tbl[0] = value;
            counter -= 1;
            if (counter == 0) {
                counter = change_point[ii];
                value = value ? 0 : 1;
            }
            tbl += 8;
        }
    }

    exit:
        if (PyErr_Occurred()) {
            free(*table);
            return 0;
        }
       return 1;
}

#define PYCV_SKELETON_CONDITION(_neighbours, _v_lu)                                                                               \
{                                                                                                                                 \
    int _ii, _a_cond = 0, _b_cond = 0, _step_1 = 0, _step_2 = 0;                                                                  \
    _v_lu = 0;                                                                                                                    \
    for (_ii = 0; _ii < 8; _ii++) {                                                                                               \
        if (!_neighbours[_ii] && _neighbours[(_ii + 1) % 8]) {                                                                    \
            _a_cond += 1;                                                                                                         \
        }                                                                                                                         \
        _b_cond += _neighbours[_ii] ? 1 : 0;                                                                                      \
    }                                                                                                                             \
    _a_cond = _a_cond == 1 ? 1 : 0;                                                                                               \
    _b_cond = _b_cond >= 2 && _b_cond <= 6 ? 1 : 0;                                                                               \
    if (_a_cond && _b_cond) {                                                                                                     \
        if (!(_neighbours[0] && _neighbours[2] && _neighbours[4]) && !(_neighbours[2] && _neighbours[4] && _neighbours[6])) {     \
            _step_1 = 1;                                                                                                          \
        }                                                                                                                         \
        if (!(_neighbours[0] && _neighbours[2] && _neighbours[6]) && !(_neighbours[0] && _neighbours[4] && _neighbours[6])) {     \
            _step_2 = 1;                                                                                                          \
        }                                                                                                                         \
        if (_step_1 && _step_2) {                                                                                                 \
            _v_lu = 3;                                                                                                            \
        } else if (_step_1) {                                                                                                     \
            _v_lu = 1;                                                                                                            \
        } else if (_step_2) {                                                                                                     \
            _v_lu = 2;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
}

static int PYCV_SkeletonLUT(unsigned int **table)
{
    unsigned int *binary_table, *sk_tbl, *bin_tbl, v_lu, i_lu;
    int ii;

    if (!PYCV_Uint8BinaryTable(&binary_table)) {
        PyErr_SetString(PyExc_RuntimeError, "PYCV_Uint8BinaryTable");
        goto exit;
    }

    *table = calloc(256, sizeof(unsigned int));
    if (!*table) {
        PyErr_NoMemory();
        goto exit;
    }

    bin_tbl = binary_table;
    sk_tbl = *table;

    for (ii = 0; ii < 256; ii++) {
        PYCV_SKELETON_CONDITION(bin_tbl, v_lu);
        if (v_lu) {
            i_lu = bin_tbl[0] +
                   2 * bin_tbl[1] +
                   4 * bin_tbl[2] +
                   8 * bin_tbl[3] +
                   16 * bin_tbl[4] +
                   32 * bin_tbl[5] +
                   64 * bin_tbl[6] +
                   128 * bin_tbl[7];
            sk_tbl[i_lu] = v_lu;
        }
        bin_tbl += 8;
    }

    exit:
        free(binary_table);
        if (PyErr_Occurred()) {
            free(*table);
            return 0;
        }
       return 1;
}

#define PYCV_M_CASE_SKELETON_GET_LUV(_NTYPE, _dtype, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out)         \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    unsigned int _v_lu;                                                                                                \
    int _idx;                                                                                                          \
    _out = 0;                                                                                                          \
    if ((int)(*(_dtype *)_pointer_s)) {                                                                                \
        _idx = (int)(*((_dtype *)(_pointer_s + _offsets[1]))) +                                                        \
               2 * (int)(*((_dtype *)(_pointer_s + _offsets[2]))) +                                                    \
               4 * (int)(*((_dtype *)(_pointer_s + _offsets[5]))) +                                                    \
               8 * (int)(*((_dtype *)(_pointer_s + _offsets[8]))) +                                                    \
               16 * (int)(*((_dtype *)(_pointer_s + _offsets[7]))) +                                                   \
               32 * (int)(*((_dtype *)(_pointer_s + _offsets[6]))) +                                                   \
               64 * (int)(*((_dtype *)(_pointer_s + _offsets[3]))) +                                                   \
               128 * (int)(*((_dtype *)(_pointer_s + _offsets[0])));                                                   \
        _v_lu = _sk_lut[_idx];                                                                                         \
        if (_v_lu == 3 || (_step_n == 0 && _v_lu == 1) || (_step_n == 1 && _v_lu == 2)) {                              \
            _out = 1;                                                                                                  \
            *(_dtype *)_pointer_o = 0;                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_M_SKELETON_GET_LUV(_NTYPE, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out)                      \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_M_CASE_SKELETON_GET_LUV(BOOL, npy_bool, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);        \
        PYCV_M_CASE_SKELETON_GET_LUV(UBYTE, npy_ubyte, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);      \
        PYCV_M_CASE_SKELETON_GET_LUV(USHORT, npy_ushort, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);    \
        PYCV_M_CASE_SKELETON_GET_LUV(UINT, npy_uint, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);        \
        PYCV_M_CASE_SKELETON_GET_LUV(ULONG, npy_ulong, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);      \
        PYCV_M_CASE_SKELETON_GET_LUV(ULONGLONG, npy_ulonglong, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);\
        PYCV_M_CASE_SKELETON_GET_LUV(BYTE, npy_byte, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);        \
        PYCV_M_CASE_SKELETON_GET_LUV(SHORT, npy_short, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);      \
        PYCV_M_CASE_SKELETON_GET_LUV(INT, npy_int, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);          \
        PYCV_M_CASE_SKELETON_GET_LUV(LONG, npy_long, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);        \
        PYCV_M_CASE_SKELETON_GET_LUV(LONGLONG, npy_longlong, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);\
        PYCV_M_CASE_SKELETON_GET_LUV(FLOAT, npy_float, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);      \
        PYCV_M_CASE_SKELETON_GET_LUV(DOUBLE, npy_double, _sk_lut, _offsets, _pointer_s, _pointer_o, _step_n, _out);    \
    }                                                                                                                  \
}

PyArrayObject *PYCV_skeletonize(PyArrayObject *input)
{
    npy_intp f_shape[2] = {3, 3}, f_center[2] = {1, 1}, f_size = 9;
    unsigned int *skeleton_lut;
    PyArrayObject *skeleton, *output;
    NeighborhoodIterator iter_s;
    PYCV_ArrayIterator iter_o;
    char *po_base = NULL, *ps_base = NULL, *po = NULL, *ps = NULL;
    int num_type, px_change, st, v_sk;
    npy_intp a_size, *offsets, *ff, flag, ii;

    if (PyArray_NDIM(input) != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Error: input ndim need to be 2");
        goto exit;
    }

    a_size = PyArray_SIZE(input);
    num_type = PyArray_TYPE(input);

    if (!PYCV_SkeletonLUT(&skeleton_lut)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_SkeletonLUT");
        goto exit;
    }

    skeleton = (PyArrayObject *)PyArray_NewLikeArray(input, NPY_KEEPORDER, NULL, 1);
    output = (PyArrayObject *)PyArray_NewLikeArray(input, NPY_KEEPORDER, NULL, 1);

    if (PyArray_CopyInto(skeleton, input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_CopyInto \n");
        goto exit;
    }
    if (PyArray_CopyInto(output, input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_CopyInto \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(skeleton, f_shape, f_center, f_size, &iter_s);
    PYCV_ArrayIteratorInit(output, &iter_o);

    if (!PYCV_InitNeighborhoodOffsets(skeleton, f_shape, f_center, NULL, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }


    po_base = po = (void *)PyArray_DATA(output);
    ps_base = ps = (void *)PyArray_DATA(skeleton);
    ff = offsets;

    do {
        px_change = 0;
        for (st = 0; st < 2; st++) {
            for (ii = 0; ii < a_size; ii++) {
                if (ff[0] != flag && ff[f_size - 1] != flag) {
                    PYCV_M_SKELETON_GET_LUV(num_type, skeleton_lut, ff, ps, po, st, v_sk);
                    if (v_sk) {
                        px_change = 1;
                    }
                }
                PYCV_NEIGHBORHOOD_ITERATOR_NEXT2(iter_s, ps, iter_o, po, ff);
            }
            if (PyArray_CopyInto(skeleton, output)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_CopyInto");
                goto exit;
            }
            PYCV_NEIGHBORHOOD_ITERATOR_RESET(iter_s);
            PYCV_ARRAY_ITERATOR_RESET(iter_o);
            ff = offsets;
            po = po_base;
            ps = ps_base;
        }
    } while (px_change);

    exit:
        free(offsets);
        return PyErr_Occurred() ? NULL : output;
}

// #####################################################################################################################







