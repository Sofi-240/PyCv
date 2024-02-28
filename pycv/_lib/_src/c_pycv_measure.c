#include "c_pycv_base.h"
#include "c_pycv_measure.h"


// #####################################################################################################################

#define PYCV_ME_CASE_DFS_OBJECT_MAX(_NTYPE, _dtype, _itemsize, _i, _x, _n, _t, _n_cc,                                  \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max)              \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii, _index;                                                                                              \
    npy_double _val, _tmp_val;                                                                                         \
    _val = (npy_double)(*(_dtype *)_x);                                                                                \
    _is_max = 0;                                                                                                       \
    if (_n && _val > _t) {                                                                                             \
        _is_max = 1;                                                                                                   \
        for (_ii = 0; _ii < _n; _ii++) {                                                                               \
            if (_offsets[_ii] == _flag) {                                                                              \
                _tmp_val = _c_val;                                                                                     \
            } else {                                                                                                   \
                _tmp_val = (npy_double)(*(_dtype *)(_x + _offsets[_ii]));                                              \
            }                                                                                                          \
            if (_tmp_val > _val) {                                                                                     \
                _is_max = 0;                                                                                           \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    if (_is_max) {                                                                                                     \
        for (_ii = 0; _ii < _n_cc; _ii++) {                                                                            \
            if (_offsets[_cc_pointer[_ii]] == _flag) {                                                                 \
                continue;                                                                                              \
            }                                                                                                          \
            _index = _i + _offsets[_cc_pointer[_ii]] / _itemsize;                                                      \
            if (_visited[_index]) {                                                                                    \
                continue;                                                                                              \
            }                                                                                                          \
            _visited[_index] = 1;                                                                                      \
            *_stack++ = _index;                                                                                        \
            _sn++;                                                                                                     \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_ME_DFS_OBJECT_MAX(_NTYPE, _itemsize, _i, _x, _n, _t, _n_cc,                                               \
                               _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max)                   \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_ME_CASE_DFS_OBJECT_MAX(BOOL, npy_bool, _itemsize, _i, _x, _n, _t, _n_cc,                                  \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(UBYTE, npy_ubyte, _itemsize, _i, _x, _n, _t, _n_cc,                                \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(USHORT, npy_ushort, _itemsize, _i, _x, _n, _t, _n_cc,                              \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(UINT, npy_uint, _itemsize, _i, _x, _n, _t, _n_cc,                                  \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(ULONG, npy_ulong, _itemsize, _i, _x, _n, _t, _n_cc,                                \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(ULONGLONG, npy_ulonglong, _itemsize, _i, _x, _n, _t, _n_cc,                        \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(BYTE, npy_byte, _itemsize, _i, _x, _n, _t, _n_cc,                                  \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(SHORT, npy_short, _itemsize, _i, _x, _n, _t, _n_cc,                                \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(INT, npy_int, _itemsize, _i, _x, _n, _t, _n_cc,                                    \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(LONG, npy_long, _itemsize, _i, _x, _n, _t, _n_cc,                                  \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(LONGLONG, npy_longlong, _itemsize, _i, _x, _n, _t, _n_cc,                          \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(FLOAT, npy_float, _itemsize, _i, _x, _n, _t, _n_cc,                                \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
        PYCV_ME_CASE_DFS_OBJECT_MAX(DOUBLE, npy_double, _itemsize, _i, _x, _n, _t, _n_cc,                              \
                                    _offsets, _cc_pointer, _flag, _c_val, _visited, _stack, _sn, _is_max);             \
    }                                                                                                                  \
}

static int PYCV_init_cc_mask(npy_intp ndim, npy_intp *shape, npy_intp **cc_pointer)
{
    npy_intp dims[NPY_MAXDIMS], strides[NPY_MAXDIMS], f_size = 1, index, *cc, mid = 1;
    npy_intp ii, jj, init_position[NPY_MAXDIMS];
    PYCV_CoordinatesIterator iter;

    strides[ndim - 1] = 1;
    for (ii = 0; ii < ndim; ii++) {
        dims[ii] = 3;
        f_size *= 3;
        init_position[ii] = shape[ii] / 2;
    }

    for (ii = ndim - 2; ii >= 0; ii--) {
        strides[ii] = strides[ii + 1] * shape[ii + 1];
    }

    *cc_pointer = malloc(f_size * sizeof(npy_intp));
    if (!*cc_pointer) {
        PyErr_NoMemory();
        goto exit;
    }
    cc = *cc_pointer;

    PYCV_CoordinatesIteratorInit(ndim, dims, &iter);

    for (ii = 0; ii < f_size; ii++) {
        index = 0;
        for (jj = 0; jj < ndim; jj++) {
            index += (init_position[jj] + iter.coordinates[jj] - 1) * strides[jj];
        }
        *cc++ = index;
        PYCV_COORDINATES_ITERATOR_NEXT(iter);
    }

    exit:
        if (PyErr_Occurred()) {
            free(*cc_pointer);
            return 0;
        }
        return 1;
}


// #####################################################################################################################


int PYCV_find_object_peaks(PyArrayObject *input,
                           npy_intp *min_distance,
                           npy_double threshold,
                           PYCV_ExtendBorder mode,
                           npy_double c_val,
                           PyArrayObject *output)
{
    NeighborhoodIterator iter_i, iter_o;
    PyArray_ArgSortFunc *arg_sort_func;
    PYCV_CoordinatesList peaks;
    char *pi_base = NULL, *pi = NULL, *po = NULL, *po_base = NULL;
    int num_type_i, num_type_o;
    npy_intp ndim, array_size, itemsize_i, itemsize_o;
    npy_intp f_size, f_shape[NPY_MAXDIMS], *offsets, *ff, flag, *cc_pointer, cc_size, offsets_size;
    npy_intp ii, jj;
    npy_intp *sorted, index, index_pi, is_max, *visited, *stack, *stack_fill, stack_start, stack_end;
    npy_intp cum_sum[NPY_MAXDIMS], n;
    npy_double center, val;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(input);
    ndim = PyArray_NDIM(input);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    itemsize_i = PyArray_ITEMSIZE(input);
    itemsize_o = PyArray_ITEMSIZE(output);

    f_size = 1;
    cc_size = 1;
    offsets_size = 1;
    for (ii = 0; ii < ndim; ii++) {
        f_shape[ii] = min_distance[ii] * 2 + 1;
        f_size *= f_shape[ii];
        cc_size *= 3;
        offsets_size *= (PyArray_DIM(input, (int)ii) < f_shape[ii] ? PyArray_DIM(input, (int)ii) : f_shape[ii]);
    }

    PYCV_NeighborhoodIteratorInit(input, f_shape, NULL, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input, f_shape, NULL, NULL, &offsets, NULL, &flag, mode)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }
    offsets_size = itemsize_i == itemsize_o ? 0 : offsets_size * f_size;

    if (!PYCV_init_cc_mask(ndim, f_shape, &cc_pointer)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_init_cc_mask \n");
        goto exit;
    }

    sorted = malloc(array_size * sizeof(npy_intp));
    visited = malloc(array_size * sizeof(npy_intp));
    stack = malloc(array_size * sizeof(npy_intp));
    if (!sorted || !visited || !stack) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < array_size; ii++) {
        sorted[ii] = ii;
        visited[ii] = 0;
    }

    pi_base = pi = (void *)PyArray_DATA(input);
    po_base = po = (void *)PyArray_DATA(output);
    ff = offsets;

    arg_sort_func = PyArray_DESCR(input)->f->argsort[NPY_MERGESORT];

    if (!arg_sort_func || arg_sort_func(pi, sorted, array_size, input) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Couldn't perform argsort.\n");
        goto exit;
    }

    if (!PYCV_CoordinatesListInit(ndim, array_size, &peaks)) {
        PyErr_NoMemory();
        goto exit;
    }
    PYCV_NeighborhoodIteratorInit(output, f_shape, NULL, f_size, &iter_o);

    NPY_BEGIN_THREADS;

    for (ii = array_size - 1; ii >= 0; ii--) {
        index = sorted[ii];
        if (visited[index]) {
            continue;
        }
        visited[index] = 1;
        index_pi = index * itemsize_i;

        PYCV_NEIGHBORHOOD_ITERATOR_GOTO2_RAVEL(iter_i, pi_base, pi, iter_o, po_base, po, offsets, ff, index_pi);

        stack_fill = stack;
        stack_start = 0;
        stack_end = 0;

        PYCV_ME_DFS_OBJECT_MAX(num_type_i, itemsize_i, index, pi, f_size, threshold, cc_size,
                               ff, cc_pointer, flag, c_val, visited, stack_fill, stack_end, is_max);

        PYCV_SET_VALUE(num_type_o, po, is_max);

        if (is_max) {
            for (jj = 0; jj < ndim; jj++) {
                cum_sum[jj] = iter_i.coordinates[jj];
            }
            n = 1;
            while (stack_start < stack_end) {
                index = stack[stack_start];
                index_pi = index * itemsize_i;
                PYCV_NEIGHBORHOOD_ITERATOR_GOTO2_RAVEL(iter_i, pi_base, pi, iter_o, po_base, po, offsets, ff, index_pi);

                PYCV_ME_DFS_OBJECT_MAX(num_type_i, itemsize_i, index, pi, f_size, threshold,
                                       cc_size, ff, cc_pointer, flag, c_val, visited, stack_fill, stack_end, is_max);
                if (is_max) {
                    for (jj = 0; jj < ndim; jj++) {
                        cum_sum[jj] += iter_i.coordinates[jj];
                    }
                    n++;
                }
                stack_start++;
                PYCV_SET_VALUE(num_type_o, po, is_max);
            }
            for (jj = 0; jj < ndim; jj++) {
                center = (npy_double)cum_sum[jj] / (npy_double)n;
                cum_sum[jj] = (npy_intp)floor(center + 0.5);
            }
            PYCV_COORDINATES_LIST_APPEND(peaks, cum_sum);
        } else {
            PYCV_GET_VALUE(num_type_i, npy_double, pi, val);
            if (val <= threshold) {
                break;
            }
        }
    }

    for (ii = 0; ii < offsets_size; ii++) {
        offsets[ii] /= (npy_intp)itemsize_i;
        offsets[ii] *= (npy_intp)itemsize_o;
    }

    for (ii = 0; ii < peaks.coordinates_size; ii++) {
        PYCV_NEIGHBORHOOD_ITERATOR_GOTO(iter_o, po_base, po, offsets, ff, peaks.coordinates[ii]);
        PYCV_GET_VALUE(num_type_o, npy_intp, po, is_max);
        if (!is_max) {
            continue;
        }
        for (jj = 0; jj < f_size; jj++) {
            if (ff[jj] == 0 || ff[jj] == flag) {
                continue;
            }
            PYCV_SET_VALUE(num_type_o, (po + ff[jj]), 0);
        }
    }

    NPY_END_THREADS;

    PYCV_CoordinatesListFree(&peaks);

    exit:
        free(offsets);
        free(sorted);
        free(cc_pointer);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################


















