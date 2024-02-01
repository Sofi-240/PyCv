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
    FilterIter iter_f;
    char *po = NULL, *pi = NULL, *ma = NULL;
    npy_bool *footprint;
    int offsets_size, num_type_i, num_type_o, num_type_ma, buffer, op_true, op_false;
    npy_intp ii, offsets_flag, *offsets_lookup, *offsets_run;
    npy_bool mask_val;

    NPY_BEGIN_THREADS_DEF;

    if (!array_to_footprint(strel, &footprint, &offsets_size)) {
        goto exit;
    }

    if (!init_filter_offsets(input, PyArray_DIMS(strel), origins, footprint, &offsets_lookup, &offsets_flag, BORDER_FLAG)) {
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }
    FilterIterInit(PyArray_NDIM(input), PyArray_DIMS(input), PyArray_DIMS(strel), origins, offsets_size, &iter_f);

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
            EX_BINARY_EROSION(num_type_i, pi, offsets_size, offsets_run, offsets_flag, buffer, op_true, op_false);
        } else {
            EX_GET_EROSION_VALUE(num_type_i, pi, buffer, op_true, op_false);
        }

        if (!op_true) {
            buffer = buffer ? NPY_FALSE : NPY_TRUE;
        }

        SET_VALUE_TO(num_type_o, po, buffer);

        if (mask) {
            FILTER_ITER_NEXT3(iter_f, offsets_run, iter_i, pi, iter_o, po, iter_ma, ma);
        } else {
            FILTER_ITER_NEXT2(iter_f, offsets_run, iter_i, pi, iter_o, po);
        }
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
    FilterIter iter_f;
    char *po = NULL, *pi = NULL, *ma = NULL;
    int offsets_size, num_type_i, num_type_o, num_type_ma;
    npy_intp ii, offsets_flag, *offsets_lookup, *offsets_run;
    npy_bool mask_val, *footprint;
    double buffer, *weights;

    NPY_BEGIN_THREADS_DEF;

    if (!array_to_footprint(flat_strel, &footprint, &offsets_size)) {
        goto exit;
    }

    if (!init_filter_offsets(input, PyArray_DIMS(flat_strel), origins, footprint, &offsets_lookup, &offsets_flag, BORDER_FLAG)) {
        goto exit;
    }

    if (non_flat_strel) {
        if (!copy_data_as_double(non_flat_strel, &weights, footprint)) {
            goto exit;
        }
        if (op == ERO) {
            for (ii = 0; ii < offsets_size; ii++) {
                weights[ii] *= -1;
            }
        }
    } else {
        weights = (double *)malloc(offsets_size * sizeof(double));
        if (!weights) {
            PyErr_NoMemory();
            goto exit;
        }
        for (ii = 0; ii < offsets_size; ii++) {
            weights[ii] = 0.0;
        }
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }
    FilterIterInit(PyArray_NDIM(input), PyArray_DIMS(input), PyArray_DIMS(flat_strel), origins, offsets_size, &iter_f);

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
            EX_GRAY_ERO_OR_DIL(num_type_i, op, pi, offsets_size, offsets_run, offsets_flag, weights, buffer);
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
            FILTER_ITER_NEXT3(iter_f, offsets_run, iter_i, pi, iter_o, po, iter_ma, ma);
        } else {
            FILTER_ITER_NEXT2(iter_f, offsets_run, iter_i, pi, iter_o, po);
        }
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
    char *po = NULL;
    int offsets_size, num_type;
    npy_intp nd, ii, jj, array_size, *offsets, index, index_ch, cc, position;
    npy_bool val, *footprint;
    int outside;
    npy_intp stack_start = 0, stack_end = 0, *stack, *stack_fill, *strides, *dims;

    NPY_BEGIN_THREADS_DEF;

    if (!array_to_footprint(strel, &footprint, &offsets_size)) {
        goto exit;
    }

    nd = PyArray_NDIM(output);
    array_size = PyArray_SIZE(output);
    strides = PyArray_STRIDES(output);
    dims = PyArray_DIMS(output);

    if (!init_offsets_coordinates(nd, PyArray_DIMS(strel), origins, footprint, &offsets)){
        goto exit;
    }

    stack = (npy_intp *)malloc(array_size * sizeof(npy_intp));
    if (!stack) {
        PyErr_NoMemory();
        goto exit;
    }
    stack_fill = stack;

    index = 0;
    for (jj = 0; jj < nd; jj++) {
        index += seed_point[jj] * strides[jj];
    }
    *stack_fill++ = index;
    stack_end = 1;

    num_type = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;
    po = (void *)PyArray_DATA(output);

    SET_VALUE_TO(num_type, (po + index), NPY_TRUE);

    while (stack_start < stack_end) {

        for (ii = 0; ii < offsets_size; ii++) {
            outside = 0;
            index = 0;

            index_ch = stack[stack_start];

            for (jj = 0; jj < nd; jj++) {
                cc = index_ch / strides[jj];
                cc += offsets[ii * nd + jj];
                if (cc < 0 || cc >= dims[jj]) {
                    outside = 1;
                    break;
                }
                index += cc * strides[jj];
                index_ch -= (cc - offsets[ii * nd + jj]) * strides[jj];
            }

            if (!outside) {
                val = NPY_TRUE;
                GET_VALUE_AS(num_type, npy_bool, (po + index), val);
                if (!val) {
                    SET_VALUE_TO(num_type, (po + index), NPY_TRUE);
                    *stack_fill++ = index;
                    stack_end++;
                }
            }
        }
        stack_start++;
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
                 PyArrayObject *output,
                 LabelMode label_mode)
{
    ArrayIter iter_i, iter_o;
    FilterIter iter_f;
    npy_intp nd, ii, jj, array_size, values_map_size, footprint_shape[NPY_MAXDIMS], origins[NPY_MAXDIMS], itemsize_vm;
    npy_intp offsets_flag, *offsets_lookup, *offsets_run;
    npy_bool *footprint = NULL;
    char *po = NULL, *po_base = NULL, *pi = NULL, *vm = NULL;
    int footprint_size, num_type_i, num_type_o, num_type_vm;
    int *parent, *buffer, *labels;
    int pivot = 0, con = 0, pi_con = 0, buffer_size = 0, n_labels = 1, pivot_rank = -1, con_rank = -1, label, node1, node2;

    NPY_BEGIN_THREADS_DEF;

    nd = PyArray_NDIM(input);
    array_size = PyArray_SIZE(input);

    if (values_map && !PyArray_ISCONTIGUOUS(values_map)) {
        PyErr_SetString(PyExc_RuntimeError, "values map need to be contiguous\n");
        return 0;
    }

    if (values_map) {
        itemsize_vm = PyArray_ITEMSIZE(values_map);
        values_map_size = PyArray_SIZE(values_map);
    }

    if (!footprint_for_cc(nd, connectivity, &footprint, &footprint_size)) {
        goto exit;
    }

    for (ii = 0; ii < nd; ii++) {
        footprint_shape[ii] = 3;
        origins[ii] = 1;
    }


    if (!init_filter_offsets(input, footprint_shape, origins, footprint, &offsets_lookup, &offsets_flag, BORDER_FLAG)) {
        goto exit;
    }

    parent = (int *)calloc(array_size, sizeof(int));
    if (!parent) {
        PyErr_NoMemory();
        goto exit;
    }

    buffer = malloc(footprint_size * sizeof(int));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    labels = malloc(array_size * sizeof(int));
    if (!labels) {
        PyErr_NoMemory();
        goto exit;
    }

    ArrayIterInit(input, &iter_i);
    ArrayIterInit(output, &iter_o);
    FilterIterInit(PyArray_NDIM(input), PyArray_DIMS(input), footprint_shape, origins, footprint_size, &iter_f);

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
        labels[ii] = -1;
        GET_VALUE_AS(num_type_i, int, pi, pivot);

        if (!pivot) {
            SET_VALUE_TO(num_type_o, po, pivot);
            FILTER_ITER_NEXT2(iter_f, offsets_run, iter_i, pi, iter_o, po);
            continue;
        }
        if (values_map) {
            if (pivot >= values_map_size) {
                PyErr_SetString(PyExc_RuntimeError, "pivot out of range for value map");
                NPY_END_THREADS;
                goto exit;
            }
            GET_VALUE_AS(num_type_vm, int, (vm + pivot * itemsize_vm), pivot_rank);
        }
        buffer_size = 0;
        for (jj = 0; jj < footprint_size; jj++) {
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
                GET_VALUE_AS(num_type_vm, int, (vm + pi_con * itemsize_vm), con_rank);
            }
            if (con_rank == pivot_rank) {
                buffer[buffer_size] = con;
                buffer_size++;
            }
        }

        if (!buffer_size) {
            SET_VALUE_TO(num_type_o, po, n_labels);
            n_labels++;
            FILTER_ITER_NEXT2(iter_f, offsets_run, iter_i, pi, iter_o, po);
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
        FILTER_ITER_NEXT2(iter_f, offsets_run, iter_i, pi, iter_o, po);
    }

    po = po_base;
    ARRAY_ITER_RESET(iter_o);
    n_labels = 1;
    pivot = 0;

    for (ii = 0; ii < array_size; ii++) {
        GET_VALUE_AS(num_type_o, int, po, pivot);
        if (pivot) {
            while (parent[pivot]) {
                pivot = parent[pivot];
            }
            if (labels[pivot] < 0) {
                switch (label_mode) {
                    case LABEL_BY_NLABELS:
                        labels[pivot] = n_labels;
                        n_labels++;
                        break;
                    case LABEL_BY_INDEX:
                        labels[pivot] = ii;
                        break;
                }
            }
            label = labels[pivot];
            SET_VALUE_TO(num_type_o, po, label);
        }
        ARRAY_ITER_NEXT(iter_o, po);
    }
    NPY_END_THREADS;
    exit:
        free(offsets_lookup);
        free(parent);
        free(buffer);
        free(labels);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define SKELETON_LUT_CONDITION_CHECK(_neighbours, _lookup_value)                                                                  \
{                                                                                                                                 \
    int _ii, _a_cond = 0, _b_cond = 0, _step_1 = 0, _step_2 = 0;                                                                  \
    _lookup_value = 0;                                                                                                            \
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
            _lookup_value = 3;                                                                                                    \
        } else if (_step_1) {                                                                                                     \
            _lookup_value = 1;                                                                                                    \
        } else if (_step_2) {                                                                                                     \
            _lookup_value = 2;                                                                                                    \
        }                                                                                                                         \
    }                                                                                                                             \
}

static int init_skeleton_lut(unsigned int **skeleton_lut)
{
    unsigned int lookup_value = 0, lut_index;
    unsigned int *binary_table, *bt_run, *lut_run;
    int ii, jj;

    if (!init_uint8_binary_table(&binary_table)) {
        goto exit;
    }

    *skeleton_lut = calloc(256, sizeof(unsigned int));
    if (!*skeleton_lut) {
        PyErr_NoMemory();
        goto exit;
    }

    bt_run = binary_table;
    lut_run = *skeleton_lut;

    for (ii = 0; ii < 256; ii++) {
        SKELETON_LUT_CONDITION_CHECK(bt_run, lookup_value);
        if (lookup_value) {
            lut_index = bt_run[0] +
                        2 * bt_run[1] +
                        4 * bt_run[2] +
                        8 * bt_run[3] +
                        16 * bt_run[4] +
                        32 * bt_run[5] +
                        64 * bt_run[6] +
                        128 * bt_run[7];
            lut_run[lut_index] = lookup_value;
        }
        bt_run += 8;
    }

    exit:
        free(binary_table);
        if (PyErr_Occurred()) {
            free(*skeleton_lut);
            return 0;
        }
       return 1;
}

#define TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(_NUM_TYPE, _type, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel)     \
case _NUM_TYPE:                                                                                                           \
{                                                                                                                         \
    int _index = 0;                                                                                                       \
    unsigned int _tmp;                                                                                                    \
    _change_pixel = 0;                                                                                                    \
    _index = (int)(*((_type *)(_sp + _offsets[1]))) +                                                                     \
             2 * (int)(*((_type *)(_sp + _offsets[2]))) +                                                                 \
             4 * (int)(*((_type *)(_sp + _offsets[5]))) +                                                                 \
             8 * (int)(*((_type *)(_sp + _offsets[8]))) +                                                                 \
             16 * (int)(*((_type *)(_sp + _offsets[7]))) +                                                                \
             32 * (int)(*((_type *)(_sp + _offsets[6]))) +                                                                \
             64 * (int)(*((_type *)(_sp + _offsets[3]))) +                                                                \
             128 * (int)(*((_type *)(_sp + _offsets[0])));                                                                \
    _tmp = _skeleton_lut[_index];                                                                                         \
    if (_tmp == 3 || (_step_num == 0 && _tmp == 1) || (_step_num == 1 && _tmp == 2)) {                                    \
        _change_pixel = 1;                                                                                                \
    }                                                                                                                     \
}                                                                                                                         \
break

#define EX_SKELETON_GET_LOOKUP_VALUE(_NUM_TYPE, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel)                              \
{                                                                                                                                    \
    switch (_NUM_TYPE) {                                                                                                             \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_BOOL, npy_bool, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);             \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_UBYTE, npy_ubyte, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);           \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_USHORT, npy_ushort, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);         \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_UINT, npy_uint, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);             \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_ULONG, npy_ulong, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);           \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_ULONGLONG, npy_ulonglong, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);   \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_BYTE, npy_byte, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);             \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_SHORT, npy_short, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);           \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_INT, npy_int, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);               \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_LONG, npy_long, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);             \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_LONGLONG, npy_longlong, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);     \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_FLOAT, npy_float, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);           \
        TYPE_CASE_SKELETON_GET_LOOKUP_VALUE(NPY_DOUBLE, npy_double, _skeleton_lut, _offsets, _sp, _step_num, _change_pixel);         \
    }                                                                                                                                \
}


int ops_skeletonize(PyArrayObject *input, PyArrayObject *output)
{
    const npy_intp shape[2] = {3, 3};
    const npy_intp origins[2] = {1, 1};

    unsigned int *skeleton_lut, to_change_pixel = 0;
    PyArrayObject *skeleton;
    ArrayIter iter_o, iter_s;
    npy_intp nd, ii, ss, num_type, size, *offsets;
    char *po_base = NULL, *ps_base = NULL, *po = NULL, *ps = NULL;
    npy_bool *borders_lookup;
    int pixel_change = 0, s_val = 0;

    num_type = PyArray_TYPE(input);
    nd = PyArray_NDIM(input);
    size = PyArray_SIZE(input);

    if (!init_skeleton_lut(&skeleton_lut)) {
        goto exit;
    }

    if (!init_offsets_ravel(input, shape, origins, NULL, &offsets)) {
        goto exit;
    }

    if (!init_borders_lut(nd, PyArray_DIMS(input), shape, origins, &borders_lookup)) {
        goto exit;
    }

    if (PyArray_CopyInto(output, input)) {
        goto exit;
    }

    skeleton = (PyArrayObject *)PyArray_NewLikeArray(output, NPY_KEEPORDER, NULL, 1);
    if (PyArray_CopyInto(skeleton, input)) {
        goto exit;
    }

    ArrayIterInit(output, &iter_o);
    ArrayIterInit(skeleton, &iter_s);


    po_base = po = (void *)PyArray_DATA(output);
    ps_base = ps = (void *)PyArray_DATA(skeleton);

    do {
        pixel_change = 0;
        for (ss = 0; ss < 2; ss++) {
            for (ii = 0; ii < size; ii++) {
                if (borders_lookup[ii]) {
                    ARRAY_ITER_NEXT2(iter_o, po, iter_s, ps);
                    continue;
                }
                GET_VALUE_AS(num_type, int, ps, s_val);
                if (s_val) {
                    EX_SKELETON_GET_LOOKUP_VALUE(num_type, skeleton_lut, offsets, ps, ss, to_change_pixel);
                    if (to_change_pixel) {
                        SET_VALUE_TO(num_type, po, 0);
                        pixel_change = 1;
                    }
                }
                ARRAY_ITER_NEXT2(iter_o, po, iter_s, ps);
            }
            if (PyArray_CopyInto(skeleton, output)) {
                goto exit;
            }
            ARRAY_ITER_RESET(iter_o);
            ARRAY_ITER_RESET(iter_s);
            po = po_base;
            ps = ps_base;
        }
    } while (pixel_change);


    exit:
        free(borders_lookup);
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################
