#include "ops_base.h"
#include "morphology.h"

// #####################################################################################################################

#define TYPE_CASE_BINARY_EROSION(_NUM_TYPE, _type, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val)     \
case _NUM_TYPE:                                                                                                                         \
{                                                                                                                                       \
    npy_intp _ii;                                                                                                                       \
    for (_ii = 0; _ii < _offsets_size; _ii++) {                                                                                         \
        if (_offsets[_ii] < _offsets_flag) {                                                                                            \
            _buffer_val = *(_type *)(_pi + _offsets[_ii]) ? _true_val : _false_val;                                                     \
            if (!_buffer_val) {                                                                                                         \
                break;                                                                                                                  \
            }                                                                                                                           \
        }                                                                                                                               \
    }                                                                                                                                   \
}                                                                                                                                       \
break

#define EX_BINARY_EROSION(_NUM_TYPE, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val)                               \
{                                                                                                                                                   \
    switch (_NUM_TYPE) {                                                                                                                            \
        TYPE_CASE_BINARY_EROSION(NPY_BOOL, npy_bool,  _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);             \
        TYPE_CASE_BINARY_EROSION(NPY_UBYTE, npy_ubyte, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);            \
        TYPE_CASE_BINARY_EROSION(NPY_USHORT, npy_ushort, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);          \
        TYPE_CASE_BINARY_EROSION(NPY_UINT, npy_uint, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);              \
        TYPE_CASE_BINARY_EROSION(NPY_ULONG, npy_ulong, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);            \
        TYPE_CASE_BINARY_EROSION(NPY_ULONGLONG, npy_ulonglong, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);    \
        TYPE_CASE_BINARY_EROSION(NPY_BYTE, npy_byte, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);              \
        TYPE_CASE_BINARY_EROSION(NPY_SHORT, npy_short, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);            \
        TYPE_CASE_BINARY_EROSION(NPY_INT, npy_int, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);                \
        TYPE_CASE_BINARY_EROSION(NPY_LONG, npy_long, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);              \
        TYPE_CASE_BINARY_EROSION(NPY_LONGLONG, npy_longlong, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);      \
        TYPE_CASE_BINARY_EROSION(NPY_FLOAT, npy_float, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);            \
        TYPE_CASE_BINARY_EROSION(NPY_DOUBLE, npy_double, _pi, _offsets_size, _offsets, _offsets_flag, _buffer_val, _true_val, _false_val);          \
    }                                                                                                                                               \
}

#define TYPE_CASE_GET_EROSION_VALUE(_NUM_TYPE, _type, _po, _buffer_val, _true_val, _false_val)   \
case _NUM_TYPE:                                                                                  \
    _buffer_val = *(_type *)_po ? _true_val : _false_val;                                        \
    break

#define EX_GET_EROSION_VALUE(_NUM_TYPE, _po, _buffer_val, _true_val, _false_val)                                       \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        TYPE_CASE_GET_EROSION_VALUE(NPY_BOOL, npy_bool, _po, _buffer_val, _true_val, _false_val);                      \
        TYPE_CASE_GET_EROSION_VALUE(NPY_UBYTE, npy_ubyte, _po, _buffer_val, _true_val, _false_val);                    \
        TYPE_CASE_GET_EROSION_VALUE(NPY_USHORT, npy_ushort, _po, _buffer_val, _true_val, _false_val);                  \
        TYPE_CASE_GET_EROSION_VALUE(NPY_UINT, npy_uint, _po, _buffer_val, _true_val, _false_val);                      \
        TYPE_CASE_GET_EROSION_VALUE(NPY_ULONG, npy_ulong, _po, _buffer_val, _true_val, _false_val);                    \
        TYPE_CASE_GET_EROSION_VALUE(NPY_ULONGLONG, npy_ulonglong, _po, _buffer_val, _true_val, _false_val);            \
        TYPE_CASE_GET_EROSION_VALUE(NPY_BYTE, npy_byte, _po, _buffer_val, _true_val, _false_val);                      \
        TYPE_CASE_GET_EROSION_VALUE(NPY_SHORT, npy_short, _po, _buffer_val, _true_val, _false_val);                    \
        TYPE_CASE_GET_EROSION_VALUE(NPY_INT, npy_int, _po, _buffer_val, _true_val, _false_val);                        \
        TYPE_CASE_GET_EROSION_VALUE(NPY_LONG, npy_long, _po, _buffer_val, _true_val, _false_val);                      \
        TYPE_CASE_GET_EROSION_VALUE(NPY_LONGLONG, npy_longlong, _po, _buffer_val, _true_val, _false_val);              \
        TYPE_CASE_GET_EROSION_VALUE(NPY_FLOAT, npy_float, _po, _buffer_val, _true_val, _false_val);                    \
        TYPE_CASE_GET_EROSION_VALUE(NPY_DOUBLE, npy_double, _po, _buffer_val, _true_val, _false_val);                  \
    }                                                                                                                  \
}

int ops_binary_erosion(PyArrayObject *input,
                       PyArrayObject *strel,
                       PyArrayObject *output,
                       npy_intp *origins,
                       int iterations,
                       PyArrayObject *mask,
                       int invert)
{
    ArrayIter iter_i, iter_o, iter_ma;
    char *po = NULL, *pi = NULL, *ma = NULL;
    npy_bool *footprint;
    int offsets_size, num_type_i, num_type_o, num_type_ma, buffer, op_true, op_false;
    npy_intp ii, offsets_flag, offsets_stride, *offsets_lookup, *offsets_run;
    npy_bool mask_val;

    NPY_BEGIN_THREADS_DEF;

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (mask && !valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(strel))) {
        PyErr_SetString(PyExc_RuntimeError, "strel dtype not supported");
        goto exit;
    }

    if (!array_to_footprint(strel, &footprint, &offsets_size)) {
        goto exit;
    }

    if (!init_offsets_lut(input, PyArray_DIMS(strel), origins, footprint, &offsets_lookup, &offsets_flag, &offsets_stride)) {
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    if (mask) {
        num_type_ma = PyArray_TYPE(mask);
    }

    offsets_run = offsets_lookup;
    op_true = 1;
    op_false = 0;
    if (invert) {
        op_true = 0;
        op_false = 1;
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        buffer = op_false;

        if (mask) {
            mask_val = NPY_FALSE;
            GET_VALUE_AS(num_type_ma, npy_bool, ma, mask_val);
        } else {
            mask_val = NPY_TRUE;
        }
        if (mask_val) {
            EX_BINARY_EROSION(num_type_i, pi, offsets_stride, offsets_run, offsets_flag, buffer, op_true, op_false);
        } else {
            EX_GET_EROSION_VALUE(num_type_i, pi, buffer, op_true, op_false);
        }

        if (!op_true) {
            buffer = buffer ? NPY_FALSE : NPY_TRUE;
        }

        SET_VALUE_TO(num_type_o, po, buffer);
        if (mask) {
            ARRAY_ITER_NEXT3(iter_i, pi, iter_o, po, iter_ma, ma);
        } else {
            ARRAY_ITER_NEXT2(iter_i, pi, iter_o, po);
        }
        offsets_run += offsets_stride;
    }
    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define TYPE_CASE_GRAY_ERO_OR_DIL(_NUM_TYPE, _type, _ero_dil_op,                                          \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val)     \
case _NUM_TYPE:                                                                                           \
{                                                                                                         \
    npy_intp _ii, _jj = 0;                                                                                \
    double _tmp;                                                                                          \
    _buffer_val = 0.0;                                                                                    \
    while (_jj < _offsets_size) {                                                                         \
        if (_offsets[_jj] < _offsets_flag) {                                                              \
            _buffer_val = (double)(*((_type *)(_pi + _offsets[_jj]))) + _weights[_jj];                    \
            break;                                                                                        \
        }                                                                                                 \
        _jj++;                                                                                            \
    }                                                                                                     \
    for (_ii = _jj + 1; _ii < _offsets_size; _ii++) {                                                     \
        if (_offsets[_ii] < _offsets_flag) {                                                              \
            _tmp = (double)(*((_type *)(_pi + _offsets[_ii]))) + _weights[_ii];                           \
            switch (_ero_dil_op) {                                                                        \
                case ERO:                                                                                 \
                    _buffer_val = _buffer_val < _tmp ? _buffer_val : _tmp;                                \
                    break;                                                                                \
                case DIL:                                                                                 \
                    _buffer_val = _buffer_val > _tmp ? _buffer_val : _tmp;                                \
                    break;                                                                                \
            }                                                                                             \
        }                                                                                                 \
    }                                                                                                     \
}                                                                                                         \
break

#define EX_GRAY_ERO_OR_DIL(_NUM_TYPE, _ero_dil_op,                                                                     \
                             _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val)                       \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_BOOL, npy_bool, _ero_dil_op,                                                     \
                                 _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                  \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_UBYTE, npy_ubyte, _ero_dil_op,                                                   \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_USHORT, npy_ushort, _ero_dil_op,                                                 \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_UINT, npy_uint, _ero_dil_op,                                                     \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_ULONG, npy_ulong, _ero_dil_op,                                                   \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_ULONGLONG, npy_ulonglong, _ero_dil_op,                                           \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_BYTE, npy_byte, _ero_dil_op,                                                     \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_SHORT, npy_short, _ero_dil_op,                                                   \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_INT, npy_int, _ero_dil_op,                                                       \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_LONG, npy_long, _ero_dil_op,                                                     \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_LONGLONG, npy_longlong, _ero_dil_op,                                             \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_FLOAT, npy_float, _ero_dil_op,                                                   \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
        TYPE_CASE_GRAY_ERO_OR_DIL(NPY_DOUBLE, npy_double, _ero_dil_op,                                                 \
                                  _pi, _offsets_size, _offsets, _offsets_flag, _weights, _buffer_val);                 \
    }                                                                                                                  \
}

int ops_gray_ero_or_dil(PyArrayObject *input,
                        PyArrayObject *flat_strel,
                        PyArrayObject *non_flat_strel,
                        PyArrayObject *output,
                        npy_intp *origins,
                        PyArrayObject *mask,
                        double cast_value,
                        ERO_OR_DIL_OP op)
{
    ArrayIter iter_i, iter_o, iter_ma;
    char *po = NULL, *pi = NULL, *ma = NULL;
    int offsets_size, num_type_i, num_type_o, num_type_ma;
    npy_intp ii, offsets_flag, offsets_stride, *offsets_lookup, *offsets_run;
    npy_bool mask_val, *footprint;
    double buffer, *weights;

    NPY_BEGIN_THREADS_DEF;

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (mask && !valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "flat strel dtype not supported");
        goto exit;
    }
    if (non_flat_strel && !valid_dtype(PyArray_TYPE(non_flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "non flat strel dtype not supported");
        goto exit;
    }

    if (!array_to_footprint(flat_strel, &footprint, &offsets_size)) {
        goto exit;
    }

    if (!init_offsets_lut(input, PyArray_DIMS(flat_strel), origins, footprint, &offsets_lookup, &offsets_flag, &offsets_stride)) {
        goto exit;
    }

    if (non_flat_strel) {
        if (!copy_data_as_double(non_flat_strel, &weights, footprint)) {
            goto exit;
        }
        if (op == ERO) {
            for (ii = 0; ii < offsets_stride; ii++) {
                weights[ii] *= -1;
            }
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

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);
    if (mask) {
        num_type_ma = PyArray_TYPE(mask);
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }

    offsets_run = offsets_lookup;

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        buffer = 0.0;
        if (mask) {
            mask_val = NPY_FALSE;
            GET_VALUE_AS(num_type_ma, npy_bool, ma, mask_val)
        } else {
            mask_val = NPY_TRUE;
        }
        if (mask_val) {
            EX_GRAY_ERO_OR_DIL(num_type_i, op, pi, offsets_stride, offsets_run, offsets_flag, weights, buffer);
        } else {
            GET_VALUE_AS(num_type_i, double, pi, buffer);
        }
        switch (op) {
            case ERO:
                buffer = buffer < cast_value ? cast_value : buffer;
                break;
            case DIL:
                buffer = buffer > cast_value ? cast_value : buffer;
                break;
        }
        SET_VALUE_TO(num_type_o, po, buffer);
        if (mask) {
            ARRAY_ITER_NEXT3(iter_i, pi, iter_o, po, iter_ma, ma);
        } else {
            ARRAY_ITER_NEXT2(iter_i, pi, iter_o, po);
        }
        offsets_run += offsets_stride;
    }
    NPY_END_THREADS;
    exit:
        free(offsets_lookup);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

int ops_binary_region_fill(PyArrayObject *output,
                           PyArrayObject *strel,
                           npy_intp *seed_point,
                           npy_intp *origins)
{
    ArrayIter iter;
    char *po = NULL, *po_base = NULL;
    int offsets_size, num_type;
    npy_intp nd, ii, jj, array_size, *offsets, *offsets_run;
    npy_bool is_valid, val, *footprint;
    npy_intp position[NPY_MAXDIMS], stack_start = 0, stack_end = 0, *stack, *stack_fill, *stack_go;

    NPY_BEGIN_THREADS_DEF;

    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(strel))) {
        PyErr_SetString(PyExc_RuntimeError, "strel dtype not supported");
        goto exit;
    }

    if (!array_to_footprint(strel, &footprint, &offsets_size)) {
        goto exit;
    }

    nd = PyArray_NDIM(output);
    if (!init_offsets_coordinates(nd, PyArray_DIMS(strel), origins, footprint, &offsets)) {
        goto exit;
    }

    array_size = PyArray_SIZE(output);

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
    }
    stack_end = 1;

    ArrayIterInit(output, &iter);
    num_type = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;
    po_base = po = (void *)PyArray_DATA(output);

    ARRAY_ITER_GOTO(iter, stack_go, po_base, po);
    SET_VALUE_TO(num_type, po, NPY_TRUE);

    while (stack_start < stack_end) {
        offsets_run = offsets;

        for (ii = 0; ii < offsets_size; ii++) {
            is_valid = NPY_TRUE;
            for (jj = 0; jj < nd; jj++) {
                position[jj] = stack_go[jj] - offsets_run[jj];
                if (position[jj] < 0 || position[jj] > iter.dims_m1[jj]) {
                    is_valid = NPY_FALSE;
                    break;
                }
            }
            if (is_valid) {
                ARRAY_ITER_GOTO(iter, position, po_base, po);
                val = NPY_TRUE;
                GET_VALUE_AS(num_type, npy_bool, po, val);
                if (!val) {
                    SET_VALUE_TO(num_type, po, NPY_TRUE);
                    for (jj = 0; jj < nd; jj++) {
                        *stack_fill++ = position[jj];
                    }
                    stack_end++;
                }
            }
            offsets_run += nd;
        }
        stack_start++;
        stack_go += nd;
    }
    NPY_END_THREADS;
    exit:
        free(stack);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

int ops_labeling(PyArrayObject *input,
                 int connectivity,
                 PyArrayObject *values_map,
                 PyArrayObject *output)
{
    ArrayIter iter_i, iter_o;
    npy_intp nd, ii, jj, array_size, values_map_size, footprint_shape[NPY_MAXDIMS], origins[NPY_MAXDIMS];
    npy_intp offsets_flag, offsets_stride, *offsets_lookup, *offsets_run;
    npy_bool *footprint = NULL;
    char *po = NULL, *po_base = NULL, *pi = NULL, *vm = NULL;
    int footprint_size, num_type_i, num_type_o, num_type_vm;
    int *parent, *buffer, *labels;
    int pivot = 0, con = 0, pi_con = 0, buffer_size = 0, n_labels = 1, pivot_rank = -1, con_rank = -1, label, node1, node2;

    NPY_BEGIN_THREADS_DEF;

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (values_map && !valid_dtype(PyArray_TYPE(values_map))) {
        PyErr_SetString(PyExc_RuntimeError, "values map dtype not supported");
        goto exit;
    }
    if (values_map && PyArray_NDIM(values_map) > 1) {
        PyErr_SetString(PyExc_RuntimeError, "values map need to be 1D array");
        goto exit;
    }

    nd = PyArray_NDIM(input);
    array_size = PyArray_SIZE(input);
    values_map_size = values_map ? PyArray_SIZE(values_map) : -1;

    if (!footprint_for_cc(nd, connectivity, &footprint, &footprint_size)) {
        goto exit;
    }

    footprint_size = 1;
    for (ii = 0; ii < nd; ii++) {
        footprint_size *= 3;
        footprint_shape[ii] = 3;
        origins[ii] = 1;
    }

    if (!init_offsets_lut(input, footprint_shape, origins, footprint, &offsets_lookup, &offsets_flag, &offsets_stride)) {
        goto exit;
    }

    parent = (int *)calloc(array_size, sizeof(int));
    if (!parent) {
        PyErr_NoMemory();
        goto exit;
    }

    buffer = malloc(offsets_stride * sizeof(int));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    labels = (int *)calloc(array_size, sizeof(int));
    if (!labels) {
        PyErr_NoMemory();
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);
    num_type_vm = values_map ? PyArray_TYPE(values_map) : -1;

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po_base = po = (void *)PyArray_DATA(output);
    if (values_map) {
        vm = (void *)PyArray_DATA(values_map);
    }
    offsets_run = offsets_lookup;

    for (ii = 0; ii < array_size; ii++) {
        GET_VALUE_AS(num_type_i, int, pi, pivot);
        if (!pivot) {
            SET_VALUE_TO(num_type_o, po, pivot);
            offsets_run += offsets_stride;
            ARRAY_ITER_NEXT2(iter_i, pi, iter_o, po);
            continue;
        }
        if (values_map) {
            if (pivot >= values_map_size) {
                PyErr_SetString(PyExc_RuntimeError, "pivot out of range for value map");
                NPY_END_THREADS;
                goto exit;
            }
            GET_VALUE_AS(num_type_vm, int, vm + pivot, pivot_rank);
        }
        buffer_size = 0;
        for (jj = 0; jj < offsets_stride; jj++) {
            if (offsets_run[jj] >= offsets_flag) {
                continue;
            }
            GET_VALUE_AS(num_type_o, int, po + offsets_run[jj], con);
            if (!con) {
                continue;
            }
            if (values_map) {
                GET_VALUE_AS(num_type_i, int, pi + offsets_run[jj], pi_con);
                if (pi_con >= values_map_size) {
                    PyErr_SetString(PyExc_RuntimeError, "con out of range for value map");
                    NPY_END_THREADS;
                    goto exit;
                }
                GET_VALUE_AS(num_type_vm, int, vm + pi_con, con_rank);
            }

            if (con_rank == pivot_rank) {
                buffer[buffer_size] = con;
                buffer_size++;
            }
        }

        if (!buffer_size) {
            SET_VALUE_TO(num_type_o, po, n_labels);
            n_labels++;
            offsets_run += offsets_stride;
            ARRAY_ITER_NEXT2(iter_i, pi, iter_o, po);
            continue;
        }
        label = buffer[0];
        for (jj = 1; jj < buffer_size; jj++) {
            label = label > buffer[jj] ? buffer[jj] : label;
        }
        SET_VALUE_TO(num_type_o, po, label);
        node1 = label;
        while (parent[node1]) {
            node1 = parent[node1];
        }
        for (jj = 0; jj < buffer_size; jj++) {
            if (buffer[jj] != label) {
                node2 = buffer[jj];
                while (parent[node2]) {
                    node2 = parent[node2];
                }
                if (node2 != node1) {
                    parent[node2] = node1;
                }
            }
        }
        offsets_run += offsets_stride;
        ARRAY_ITER_NEXT2(iter_i, pi, iter_o, po);
    }

    po = po_base;
    ARRAY_ITER_RESET(iter_o);
    n_labels = 1;

    for (ii = 0; ii < array_size; ii++) {
        GET_VALUE_AS(num_type_o, int, po, pivot);
        if (pivot) {
            while (parent[pivot]) {
                pivot = parent[pivot];
            }
            if (!labels[pivot]) {
                labels[pivot] = n_labels;
                n_labels++;
            }
            label = labels[pivot];
            SET_VALUE_TO(num_type_o, po, label);
        }
        ARRAY_ITER_NEXT(iter_o, po);
    }
    NPY_END_THREADS;
    exit:
        free(parent);
        free(buffer);
        free(labels);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################


