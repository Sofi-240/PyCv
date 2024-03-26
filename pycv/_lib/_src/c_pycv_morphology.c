#include "c_pycv_base.h"
#include "c_pycv_morphology.h"

// #####################################################################################################################

#define CASE_MORPH_AS_BOOLEAN(_NTYPE, _dtype, _ptr, _out)                                                              \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    _out = *(_dtype *)_ptr ? 1 : 0;                                                                                    \
}                                                                                                                      \
break

#define MORPH_AS_BOOLEAN(_NTYPE, _ptr, _out)                                                                           \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_MORPH_AS_BOOLEAN(BOOL, npy_bool, _ptr, _out);                                                             \
        CASE_MORPH_AS_BOOLEAN(UBYTE, npy_ubyte, _ptr, _out);                                                           \
        CASE_MORPH_AS_BOOLEAN(USHORT, npy_ushort, _ptr, _out);                                                         \
        CASE_MORPH_AS_BOOLEAN(UINT, npy_uint, _ptr, _out);                                                             \
        CASE_MORPH_AS_BOOLEAN(ULONG, npy_ulong, _ptr, _out);                                                           \
        CASE_MORPH_AS_BOOLEAN(ULONGLONG, npy_ulonglong, _ptr, _out);                                                   \
        CASE_MORPH_AS_BOOLEAN(BYTE, npy_byte, _ptr, _out);                                                             \
        CASE_MORPH_AS_BOOLEAN(SHORT, npy_short, _ptr, _out);                                                           \
        CASE_MORPH_AS_BOOLEAN(INT, npy_int, _ptr, _out);                                                               \
        CASE_MORPH_AS_BOOLEAN(LONG, npy_long, _ptr, _out);                                                             \
        CASE_MORPH_AS_BOOLEAN(LONGLONG, npy_longlong, _ptr, _out);                                                     \
        CASE_MORPH_AS_BOOLEAN(FLOAT, npy_float, _ptr, _out);                                                           \
        CASE_MORPH_AS_BOOLEAN(DOUBLE, npy_double, _ptr, _out);                                                         \
    }                                                                                                                  \
}

// #####################################################################################################################


#define CASE_MORPH_EROSION(_NTYPE, _dtype, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val)                      \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii = 0;                                                                                                  \
    _out = _n ? 1 : (*(_dtype *)_x ? _t_val : _f_val);                                                                 \
    while (_out && _ii < _n) {                                                                                         \
        if (*(_offsets + _ii) == _flag) {                                                                              \
            _out = _c_val ? _t_val : _f_val;                                                                           \
        } else {                                                                                                       \
            _out = *(_dtype *)(_x + *(_offsets + _ii)) ? _t_val : _f_val;                                              \
        }                                                                                                              \
        _ii++;                                                                                                         \
    }                                                                                                                  \
}                                                                                                                      \
break

#define MORPH_EROSION(_NTYPE, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val)                                   \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_MORPH_EROSION(BOOL, npy_bool, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                     \
        CASE_MORPH_EROSION(UBYTE, npy_ubyte, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                   \
        CASE_MORPH_EROSION(USHORT, npy_ushort, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                 \
        CASE_MORPH_EROSION(UINT, npy_uint, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                     \
        CASE_MORPH_EROSION(ULONG, npy_ulong, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                   \
        CASE_MORPH_EROSION(ULONGLONG, npy_ulonglong, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);           \
        CASE_MORPH_EROSION(BYTE, npy_byte, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                     \
        CASE_MORPH_EROSION(SHORT, npy_short, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                   \
        CASE_MORPH_EROSION(INT, npy_int, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                       \
        CASE_MORPH_EROSION(LONG, npy_long, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                     \
        CASE_MORPH_EROSION(LONGLONG, npy_longlong, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);             \
        CASE_MORPH_EROSION(FLOAT, npy_float, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                   \
        CASE_MORPH_EROSION(DOUBLE, npy_double, _x, _n, _offsets, _flag, _c_val, _out, _t_val, _f_val);                 \
    }                                                                                                                  \
}

// *********************************************************************************************************************

int PYCV_binary_erosion(PyArrayObject *input,
                        PyArrayObject *strel,
                        PyArrayObject *output,
                        npy_intp *center,
                        int iterations,
                        PyArrayObject *mask,
                        int invert,
                        int c_val)
{
    PYCV_ArrayIterator iter_m, iter_o;
    NeighborhoodIterator iter_i;
    PyArrayObject *input_w;
    npy_bool *footprint;
    npy_intp se_size, i_size, flag, *offsets, *ff, ii, jj, nn, *px_stack, bb = 0;
    npy_intp *stack = NULL, loc = 0, ns = 0, ns_nd, ndim;
    char *ma_base = NULL, *ma = NULL, *po_base = NULL, *po = NULL, *pi_base = NULL, *pi = NULL;
    int true_val = 1, false_val = 0, ma_val = 1, o_val, po_val, px_change = 1;

    if (iterations == 0) {
        return 1;
    }

    if (iterations != 1) {
        input_w = (PyArrayObject *)PyArray_NewLikeArray(output, NPY_KEEPORDER, NULL, 1);
        if (PyArray_CopyInto(input_w, input)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_CopyInto \n");
            goto exit;
        }
    } else {
        input_w = input;
    }

    if (!PYCV_AllocateToFootprint(strel, &footprint, &se_size, 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_AllocateToFootprint \n");
        goto exit;
    }

    if (!PYCV_InitNeighborhoodOffsets(input_w, PyArray_DIMS(strel), center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(input_w, PyArray_DIMS(strel), center, se_size, &iter_i);
    PYCV_ArrayIteratorInit(output, &iter_o);

    i_size = PyArray_SIZE(output);
    ndim = iter_i.nd_m1 + 1;
    if (iterations != 1) {
        stack = malloc(i_size * ndim * sizeof(npy_intp));
        if (!stack) {
            PyErr_NoMemory();
            goto exit;
        }
        loc = 1;
    }

    po_base = po = (void *)PyArray_DATA(output);
    pi_base = pi = (void *)PyArray_DATA(input_w);
    if (mask) {
        PYCV_ArrayIteratorInit(mask, &iter_m);
        ma_base = ma = (void *)PyArray_DATA(mask);
    }

    if (invert) {
        true_val = 0;
        false_val = 1;
    }

    ff = offsets;

    for (ii = 0; ii < i_size; ii++) {
        if (mask) {
            MORPH_AS_BOOLEAN(iter_m.numtype, ma, ma_val);
        }
        if (ma_val) {
            MORPH_EROSION(iter_i.numtype, pi, se_size, ff, flag, c_val, o_val, true_val, false_val);
            if (iterations != 1 && o_val) {
                ns_nd = ns * ndim;
                for (jj = 0; jj < ndim; jj++) {
                    *(stack + ns_nd + jj) = *(iter_i.coordinates + jj);
                }
                ns++;
            }
            if (!true_val) {
                o_val = o_val ? 0 : 1;
            }
        } else {
            MORPH_AS_BOOLEAN(iter_i.numtype, pi, o_val);
        }
        PYCV_SET_VALUE(iter_o.numtype, po, o_val);

        if (mask) {
            PYCV_NEIGHBORHOOD_ITERATOR_NEXT3(iter_i, pi, iter_o, po, iter_m, ma, ff);
        } else {
            PYCV_NEIGHBORHOOD_ITERATOR_NEXT2(iter_i, pi, iter_o, po, ff);
        }
    }

    iterations--;

    while (iterations && ns && px_change) {
        if (PyArray_CopyInto(input_w, output)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_CopyInto \n");
            goto exit;
        }
        nn = ns;
        ns = 0;
        px_stack = stack;
        px_change = 0;

        for (ii = 0; ii < nn; ii++) {
            if (mask) {
                PYCV_NEIGHBORHOOD_ITERATOR_GOTO3(iter_i, pi_base, pi, iter_o, po_base, po, iter_m, ma_base, ma, offsets, ff, px_stack);
                MORPH_AS_BOOLEAN(iter_m.numtype, ma, ma_val);
            } else {
                PYCV_NEIGHBORHOOD_ITERATOR_GOTO2(iter_i, pi_base, pi, iter_o, po_base, po, offsets, ff, px_stack);
            }
            if (ma_val) {
                MORPH_EROSION(iter_i.numtype, pi, se_size, ff, flag, c_val, o_val, true_val, false_val);
                if (iterations != 1 && o_val) {
                    ns_nd = ns * ndim;
                    for (jj = 0; jj < ndim; jj++) {
                        *(stack + ns_nd + jj) = *(iter_i.coordinates + jj);
                    }
                    ns++;
                }
                if (!true_val) {
                    o_val = o_val ? 0 : 1;
                }
                if (!px_change) {
                    MORPH_AS_BOOLEAN(iter_i.numtype, pi, po_val);
                    if (po_val != o_val) {
                        px_change = 1;
                    }
                }
            } else {
                MORPH_AS_BOOLEAN(iter_i.numtype, pi, o_val);
            }
            px_stack += ndim;
            PYCV_SET_VALUE(iter_o.numtype, po, o_val);
        }
        iterations--;
    }

    exit:
        free(offsets);
        free(footprint);
        if (loc) {
            free(stack);
        }
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define CASE_MORPH_GRAY_EROSION(_NTYPE, _dtype, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv)                       \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    npy_double _v;                                                                                                     \
    _out = (npy_double)(*(_dtype *)_x);                                                                                \
    for (_ii = 0; _ii < _n; _ii++) {                                                                                   \
        if (*(_offsets + _ii) == _flag) {                                                                              \
            _v = _c_val + *(_h + _ii);                                                                                 \
        } else {                                                                                                       \
            _v = (npy_double)(*(_dtype *)(_x + *(_offsets + _ii))) + *(_h + _ii);                                      \
        }                                                                                                              \
        if (!_ii || (!_inv && _v < _out) || (_inv && _v > _out)) {                                                     \
            _out = _v;                                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define MORPH_GRAY_EROSION(_NTYPE, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv)                                    \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_MORPH_GRAY_EROSION(BOOL, npy_bool, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                      \
        CASE_MORPH_GRAY_EROSION(UBYTE, npy_ubyte, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                    \
        CASE_MORPH_GRAY_EROSION(USHORT, npy_ushort, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                  \
        CASE_MORPH_GRAY_EROSION(UINT, npy_uint, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                      \
        CASE_MORPH_GRAY_EROSION(ULONG, npy_ulong, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                    \
        CASE_MORPH_GRAY_EROSION(ULONGLONG, npy_ulonglong, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);            \
        CASE_MORPH_GRAY_EROSION(BYTE, npy_byte, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                      \
        CASE_MORPH_GRAY_EROSION(SHORT, npy_short, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                    \
        CASE_MORPH_GRAY_EROSION(INT, npy_int, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                        \
        CASE_MORPH_GRAY_EROSION(LONG, npy_long, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                      \
        CASE_MORPH_GRAY_EROSION(LONGLONG, npy_longlong, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);              \
        CASE_MORPH_GRAY_EROSION(FLOAT, npy_float, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                    \
        CASE_MORPH_GRAY_EROSION(DOUBLE, npy_double, _x, _h, _n, _offsets, _flag, _c_val, _out, _inv);                  \
    }                                                                                                                  \
}

// *********************************************************************************************************************


int PYCV_gray_erosion(PyArrayObject *input,
                      PyArrayObject *strel,
                      PyArrayObject *output,
                      npy_intp *center,
                      PyArrayObject *mask,
                      int invert,
                      npy_double c_val)
{
    PYCV_ArrayIterator iter_o, iter_m;
    NeighborhoodIterator iter_i;
    npy_intp a_size, f_size, flag, ii, *offsets = NULL, *ff = NULL;
    npy_bool *footprint = NULL;
    npy_double *h = NULL, o_val;
    char *po = NULL, *pi = NULL, *ma = NULL;
    int ma_val = 1;

    NPY_BEGIN_THREADS_DEF;

    if (!PYCV_AllocateKernelFlip(strel, &footprint, &h)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_ToConvolveKernel \n");
        goto exit;
    }
    PYCV_FOOTPRINT_NONZERO(footprint, PyArray_SIZE(strel), f_size);

    if (PyArray_TYPE(strel) == NPY_BOOL) {
        for (ii = 0; ii < f_size; ii++) {
            *(h + ii) = 0;
        }
    } else if (!invert) {
        for (ii = 0; ii < f_size; ii++) {
            *(h + ii) = -*(h + ii);
        }
    }
    if (!PYCV_InitNeighborhoodOffsets(input, PyArray_DIMS(strel), center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(input, PyArray_DIMS(strel), center, f_size, &iter_i);
    PYCV_ArrayIteratorInit(output, &iter_o);

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    if (mask) {
        PYCV_ArrayIteratorInit(mask, &iter_m);
        ma = (void *)PyArray_DATA(mask);
    }

    a_size = PyArray_SIZE(input);

    ff = offsets;


    NPY_BEGIN_THREADS;

    for (ii = 0; ii < a_size; ii++) {
        if (mask) {
            MORPH_AS_BOOLEAN(iter_m.numtype, ma, ma_val);
        }
        if (ma_val) {
            MORPH_GRAY_EROSION(iter_i.numtype, pi, h, f_size, ff, flag, c_val, o_val, invert);
        } else {
            PYCV_GET_VALUE(iter_i.numtype, npy_double, pi, o_val);
        }
        PYCV_SET_VALUE_F2A(iter_o.numtype, po, o_val);
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

#define CASE_FILL_IF_ZERO(_NTYPE, _dtype, _pi, _pos, _of, _flag, _dfs, _n)                                             \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    if (_of != _flag && !*(_dtype *)(_pi + _of)) {                                                                     \
        *(_dtype *)(_pi + _of) = 1;                                                                                    \
        *(_dfs + _n) = _pos + _of;                                                                                     \
        _n++;                                                                                                          \
    }                                                                                                                  \
}                                                                                                                      \
break

#define FILL_IF_ZERO(_NTYPE, _pi, _pos, _of, _flag, _dfs, _n)                                                          \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_FILL_IF_ZERO(BOOL, npy_bool, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(UBYTE, npy_ubyte, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(USHORT, npy_ushort, _pi, _pos, _of, _flag, _dfs, _n);                                        \
        CASE_FILL_IF_ZERO(UINT, npy_uint, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(ULONG, npy_ulong, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(ULONGLONG, npy_ulonglong, _pi, _pos, _of, _flag, _dfs, _n);                                  \
        CASE_FILL_IF_ZERO(BYTE, npy_byte, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(SHORT, npy_short, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(INT, npy_int, _pi, _pos, _of, _flag, _dfs, _n);                                              \
        CASE_FILL_IF_ZERO(LONG, npy_long, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(LONGLONG, npy_longlong, _pi, _pos, _of, _flag, _dfs, _n);                                    \
        CASE_FILL_IF_ZERO(FLOAT, npy_float, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(DOUBLE, npy_double, _pi, _pos, _of, _flag, _dfs, _n);                                        \
    }                                                                                                                  \
}

int PYCV_binary_region_fill(PyArrayObject *output,
                            npy_intp *seed_point,
                            PyArrayObject *strel,
                            npy_intp *center)
{
    NeighborhoodIterator iter_o;
    char *po = NULL, *po_base = NULL;
    npy_bool *footprint;
    npy_intp array_size, f_size, *offsets, *ff, flag;
    npy_intp *dfs = NULL, n = 1, ni = 0, ii;

    array_size = PyArray_SIZE(output);

    if (!PYCV_AllocateToFootprint(strel, &footprint, &f_size, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_AllocateToFootprint \n");
        goto exit;
    }

    if (!PYCV_InitNeighborhoodOffsets(output, PyArray_DIMS(strel), center, footprint,
                                      &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }
    PYCV_NeighborhoodIteratorInit(output, PyArray_DIMS(strel), center, f_size, &iter_o);

    dfs = malloc(array_size * sizeof(npy_intp));

    if (!dfs) {
        PyErr_NoMemory();
        goto exit;
    }

    PYCV_RAVEL_COORDINATE(seed_point, iter_o.nd_m1 + 1, iter_o.strides, *dfs);

    po_base = po = (void *)PyArray_DATA(output);

    while (ni < n) {
        PYCV_NEIGHBORHOOD_ITERATOR_GOTO_RAVEL(iter_o, po_base, po, offsets, ff, *(dfs + ni));

        for (ii = 0; ii < f_size; ii++) {
            FILL_IF_ZERO(iter_o.numtype, po, *(dfs + ni), *(ff + ii), flag, dfs, n);
        }
        ni++;
    }

    exit:
        free(dfs);
        free(offsets);
        free(footprint);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define LABELING_VALID_LONGLONG(_array) PyArray_TYPE(_array) == NPY_INT64

int PYCV_labeling(PyArrayObject *input, npy_intp connectivity, PyArrayObject *output)
{
    PYCV_ArrayIterator iter_o;
    NeighborhoodIterator iter_i;
    PyArrayObject *c_output, *c_input;
    char *po_base = NULL, *po = NULL, *pi = NULL;
    npy_bool *footprint;
    npy_intp ndim, array_size, f_size, *f_shape = NULL, *f_center = NULL, *offsets = NULL, *ff = NULL, flag;
    npy_intp ii, jj, loc = 0;
    int vi, ne, pc, ppi, ce, n_labels = 1, itemsize = (int)NPY_SIZEOF_LONGLONG, *djs = NULL, *edges = NULL;

    if (!LABELING_VALID_LONGLONG(input) || !LABELING_VALID_LONGLONG(output)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: LABELING_VALID_LONGLONG \n");
        return 0;
    }

    NPY_BEGIN_THREADS_DEF;

    c_output = (PyArrayObject *)PyArray_GETCONTIGUOUS(output);
    c_input = (PyArrayObject *)PyArray_GETCONTIGUOUS(input);

    array_size = PyArray_SIZE(c_input);
    ndim = (npy_intp)PyArray_NDIM(c_input);

    f_shape = malloc(ndim * 2 * sizeof(npy_intp));
    if (!f_shape) {
        PyErr_NoMemory();
        goto exit;
    }
    f_center = f_shape + ndim;

    for (ii = 0; ii < ndim; ii++) {
        *(f_shape + ii) = 3;
        *(f_center + ii) = 1;
    }

    if (!PYCV_DefaultFootprint(ndim, connectivity, &footprint, &f_size, 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_DefaultFootprint \n");
        goto exit;
    }

    djs = malloc((array_size + f_size) * sizeof(int));
    if (!djs) {
        PyErr_NoMemory();
        goto exit;
    }
    edges = djs + array_size;


    if (!PYCV_InitNeighborhoodOffsets(c_input, f_shape, f_center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(c_input, f_shape, f_center, f_size, &iter_i);
    PYCV_ArrayIteratorInit(c_output, &iter_o);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(c_input);
    po_base = po = (void *)PyArray_DATA(output);
    ff = offsets;

    for (ii = 0; ii < array_size; ii++) {
        vi = (int)(*(npy_longlong *)pi);
        if (vi) {
            *(djs + ii) = (int)ii;
            ne = 0;
            for (jj = 0; jj < f_size; jj++) {
                if (*(ff + jj) == flag || vi != (int)(*(npy_longlong *)(pi + *(ff + jj)))) {
                    continue;
                }
                pc = (int)ii + (int)(*(ff + jj)) / itemsize;
                *(djs + ii) = *(djs + ii) > *(djs + pc) ? *(djs + pc) : *(djs + ii);
                *(edges + ne) = pc;
                ne++;
            }
            if (ne) {
                ppi = *(djs + ii);
                while (*(djs + ppi) != ppi) {
                    ppi = *(djs + ppi);
                }
                for (jj = 0; jj < ne; jj++) {
                    ce = *(edges + jj);
                    while (*(djs + ce) != ce) {
                        ce = *(djs + ce);
                    }
                    if (ce != ppi) {
                        *(djs + ce) = ppi;
                    }
                }
            }
        } else {
            *(djs + ii) = -1;
        }
        PYCV_NEIGHBORHOOD_ITERATOR_NEXT(iter_i, pi, ff);
    }

    for (ii = 0; ii < array_size; ii++) {
        if (*(djs + ii) != -1) {
            if (*(djs + ii) != *(djs + *(djs + ii))) {
                *(djs + ii) = *(djs + *(djs + ii));
            }

            if (!*(npy_longlong *)(po_base + (*(djs + ii) * itemsize))) {
                *(npy_longlong *)(po_base + (*(djs + ii) * itemsize)) = (npy_longlong)n_labels;
                n_labels++;
            }

            *(npy_longlong *)po = *(npy_longlong *)(po_base + (*(djs + ii) * itemsize));

        } else {
            *(npy_longlong *)po = 0;
        }
        PYCV_ARRAY_ITERATOR_NEXT(iter_o, po);
    }

    NPY_END_THREADS;
    exit:
        Py_XDECREF(c_output);
        Py_XDECREF(c_input);
        free(footprint);
        free(offsets);
        free(djs);
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

// *********************************************************************************************************************

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







