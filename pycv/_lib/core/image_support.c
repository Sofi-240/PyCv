#include "ops_base.h"
#include "image_support.h"

// #####################################################################################################################

/*
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


is_up and is_right:
gy >= 0 and gx >= 0

  |
   __ ---->  / ---> \ gradient direction

is_down and is_left:
gy <= 0 and gx <= 0

  __     ---> / ---> \ gradient direction
     |

is_down and is_right:
gy <= 0 and gx >= 0

    __   ---> \---> / gradient direction
   |

is_up and is_left:
gy >= 0 and gx <= 0

    |  ---> \---> / gradient direction
  __

[0, 1, 2],
[3, 4, 5]
[6, 7, 8]

90 - 135                    135 - 180
[22, 21, 0],                [22,  0,  0],
[0,  1,  0],      or        [21,  1, 11],
[0, 11, 12]                 [ 0,  0, 12],

|gy| > |gx|                 |gy| <= |gx|
e = |gx| / |gy|             e = |gy| / |gx|

0 - 45                     45 - 90
[ 0,  0, 12],              [ 0, 11, 12],
[21,  1, 11],      or      [ 0,  1,  0],
[22,  0,  0]               [22, 21,  0],

|gx| > |gy|                |gx| <= |gy|
e = |gy| / |gx|            e = |gx| / |gy|

upper = (12 * w + 11 * (1.0 - w))
lower = (22 * w + 21 * (1.0 - w))
*/

#define CASE_TYPE_CANNY_INTERPOLATE(_NUM_TYPE, _type, _offsets, _th, _mag_po, _gy, _gx, _out_val)                      \
case _NUM_TYPE:                                                                                                        \
{                                                                                                                      \
    unsigned int _grad_up, _grad_down, _grad_left, _grad_right, _diag_pos, _diag_neg;                                  \
    double _abs_gy, _abs_gx, _est, _mag, _p11, _p12, _p21, _p22;                                                       \
    _mag = (double)(*((_type *)_mag_po));                                                                              \
    _out_val = 0.0;                                                                                                    \
    if (_mag >= _th) {                                                                                                 \
        _grad_up = _gy >= 0 ? 1 : 0;                                                                                   \
        _grad_down = _gy <= 0 ? 1 : 0;                                                                                 \
        _grad_right = _gx >= 0 ? 1 : 0;                                                                                \
        _grad_left = _gx <= 0 ? 1 : 0;                                                                                 \
        _diag_neg = (_grad_up && _grad_right) || (_grad_down && _grad_left) ? 1 : 0;                                   \
        _diag_pos = (_grad_down && _grad_right) || (_grad_up && _grad_left) ? 1 : 0;                                   \
        if (_diag_neg || _diag_pos) {                                                                                  \
            _abs_gy = fabs(_gy);                                                                                       \
            _abs_gx = fabs(_gx);                                                                                       \
            if (_diag_neg && _abs_gy > _abs_gx) {                                                                      \
                _est = _abs_gx / _abs_gy;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[7])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[8])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[1])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[0])));                                                  \
            } else if (_diag_neg) {                                                                                    \
                _est = _abs_gy / _abs_gx;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[5])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[8])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[3])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[0])));                                                  \
            } else if (_diag_pos && _abs_gy < _abs_gx) {                                                               \
                _est = _abs_gy / _abs_gx;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[5])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[2])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[3])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[6])));                                                  \
            } else {                                                                                                   \
                _est = _abs_gx / _abs_gy;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[1])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[2])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[7])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[6])));                                                  \
            }                                                                                                          \
            if ((_p12 * _est + _p11 * (1 - _est)) <= _mag && (_p22 * _est + _p21 * (1 - _est)) <= _mag) {              \
                _out_val = _mag;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define EX_CANNY_INTERPOLATE(_NUM_TYPE, _offsets, _th, _mag_po, _gy, _gx, _out_val)                                    \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_BOOL, npy_bool, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_UBYTE, npy_ubyte, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_USHORT, npy_ushort, _offsets, _th, _mag_po, _gy, _gx, _out_val);               \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_UINT, npy_uint, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_ULONG, npy_ulong, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_ULONGLONG, npy_ulonglong, _offsets, _th, _mag_po, _gy, _gx, _out_val);         \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_BYTE, npy_byte, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_SHORT, npy_short, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_INT, npy_int, _offsets, _th, _mag_po, _gy, _gx, _out_val);                     \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_LONG, npy_long, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_LONGLONG, npy_longlong, _offsets, _th, _mag_po, _gy, _gx, _out_val);           \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_FLOAT, npy_float, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_DOUBLE, npy_double, _offsets, _th, _mag_po, _gy, _gx, _out_val);               \
    }                                                                                                                  \
}

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output)
{
    const npy_intp shape[2] = {3, 3};

    npy_intp array_size;
    npy_intp *offsets;
    npy_bool *borders_lookup;
    npy_intp ii;

    ArrayIter iter_o, iter_ma, iter_m, iter_y, iter_x;
    char *po = NULL, *pma = NULL, *pm = NULL, *py = NULL, *px = NULL;
    int num_type_o, num_type_ma, num_type_m, num_type_y, num_type_x;

    double gy = 0.0, gx = 0.0, mag;
    int ma;

    NPY_BEGIN_THREADS_DEF;

    if (!init_offsets_ravel(magnitude, shape, NULL, NULL, &offsets)) {
        goto exit;
    }
    if (!init_borders_lut(PyArray_NDIM(magnitude), PyArray_DIMS(magnitude), shape, NULL, &borders_lookup)) {
        goto exit;
    }

    array_size = PyArray_SIZE(magnitude);

    num_type_o = PyArray_TYPE(output);
    if (mask) {
        num_type_ma = PyArray_TYPE(mask);
    }
    num_type_m = PyArray_TYPE(magnitude);
    num_type_y = PyArray_TYPE(grad_y);
    num_type_x = PyArray_TYPE(grad_x);

    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }
    ArrayIterInit(magnitude, &iter_m);
    ArrayIterInit(grad_y, &iter_y);
    ArrayIterInit(grad_x, &iter_x);

    NPY_BEGIN_THREADS;

    po = (void *)PyArray_DATA(output);
    if (mask) {
        pma = (void *)PyArray_DATA(mask);
    }
    pm = (void *)PyArray_DATA(magnitude);
    py = (void *)PyArray_DATA(grad_y);
    px = (void *)PyArray_DATA(grad_x);

    for (ii = 0; ii < array_size; ii++) {
        mag = 0.0;
        ma = 0;
        if (!borders_lookup[ii]) {
            ma = 1;
            if (mask) {
                GET_VALUE_AS(num_type_ma, int, pma, ma);
            }
        }
        if (ma) {
            GET_VALUE_AS(num_type_y, double, py, gy);
            GET_VALUE_AS(num_type_x, double, px, gx);
            EX_CANNY_INTERPOLATE(num_type_m, offsets, threshold, pm, gy, gx, mag);
        }
        SET_VALUE_TO(num_type_o, po, mag);

        ARRAY_ITER_NEXT3(iter_m, pm, iter_y, py, iter_x, px);
        if (mask) {
            ARRAY_ITER_NEXT2(iter_ma, pma, iter_o, po);
        } else {
            ARRAY_ITER_NEXT(iter_o, po);
        }
    }
    NPY_END_THREADS;
    exit:
        free(offsets);
        free(borders_lookup);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

int ops_build_max_tree(PyArrayObject *input,
                       PyArrayObject *traverser,
                       PyArrayObject *parent,
                       int connectivity,
                       PyArrayObject *values_map)
{
    char *pi = NULL, *pi_base = NULL, *pt = NULL, *pp = NULL, *vm = NULL;
    int is_contiguous_p, num_type_p, num_type_t, num_type_i, num_type_vm;
    npy_intp itemsize_vm, itemsize_p, itemsize_i;
    npy_intp size, nd, index_strides[NPY_MAXDIMS], vm_size, *strides_p, *dims;
    ArrayIter iter_t;

    npy_bool *footprint = NULL;
    npy_intp *offsets, *offsets_run;
    int offsets_size, is_outside = 0;
    npy_intp shape[NPY_MAXDIMS];

    npy_intp *traver, *nodes, ii, jj, index, index_p = 0, index_n = 0, index_t = 0, node, root;
    PyArrayObject *input_flt;
    PyArray_ArgSortFunc *arg_sort_func;
    int undef = -1;

    npy_intp c_q = 0, c_qq = 0, c_qi = 0, c_qqi = 0;
    double c_vq = 0, c_vqq = 0;
    npy_intp c_rq = 0, c_rqq = 0, c_ivq = 0, c_ivqq = 0;

    NPY_BEGIN_THREADS_DEF;

    if (values_map && !PyArray_ISCONTIGUOUS(values_map)) {
        PyErr_SetString(PyExc_RuntimeError, "values map need to be contiguous\n");
        goto exit;
    } else if (values_map) {
        itemsize_vm = PyArray_ITEMSIZE(values_map);
        num_type_vm = PyArray_TYPE(values_map);
        vm_size = PyArray_SIZE(values_map);
    }

    size = PyArray_SIZE(input);
    nd = PyArray_NDIM(input);
    dims = PyArray_DIMS(input);
    num_type_i = PyArray_TYPE(input);
    itemsize_i = PyArray_ITEMSIZE(input);

    index_strides[nd - 1] = 1;
    shape[nd - 1] = 3;
    for (ii = nd - 2; ii >= 0; ii--) {
        index_strides[ii] = index_strides[ii + 1] * PyArray_DIM(input, ii + 1);
        shape[ii] = 3;
    }

    is_contiguous_p = PyArray_ISCONTIGUOUS(parent);
    itemsize_p = PyArray_ITEMSIZE(parent);
    strides_p = PyArray_STRIDES(parent);
    num_type_p = PyArray_TYPE(parent);

    num_type_t = PyArray_TYPE(traverser);
    input_flt = (PyArrayObject *)PyArray_GETCONTIGUOUS(input);

    traver = (npy_intp *)malloc(size * sizeof(npy_intp));
    if (!traver) {
        PyErr_NoMemory();
        goto exit;
    }

    nodes = (npy_intp *)malloc(size * sizeof(npy_intp));
    if (!nodes) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < size; ii++) {
        traver[ii] = ii;
        nodes[ii] = undef;
    }

    pi_base = pi = (void *)PyArray_DATA(input_flt);
    pt = (void *)PyArray_DATA(traverser);
    pp = (void *)PyArray_DATA(parent);
    if (values_map) {
        vm = (void *)PyArray_DATA(values_map);
    }

    arg_sort_func = PyArray_DESCR(input_flt)->f->argsort[NPY_MERGESORT];

    if (!arg_sort_func || arg_sort_func(pi, traver, size, input_flt) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Couldn't perform argsort.\n");
        goto exit;
    }

    if (!footprint_as_con(nd, connectivity, &footprint, &offsets_size, 1)) {
        goto exit;
    }

    if (!init_offsets_coordinates(nd, shape, NULL, footprint, &offsets)){
        goto exit;
    }

    ArrayIterInit(traverser, &iter_t);

    NPY_BEGIN_THREADS;

    for (ii = size - 1; ii >= 0; ii--) {
        index = traver[ii];
        nodes[index] = index;

        GET_INDEX_AS_CONTIGUOUS(index, index_strides, is_contiguous_p, itemsize_p, nd, strides_p, index_p);
        SET_VALUE_TO(num_type_p, (pp + index_p), index);

        offsets_run = offsets;

        for (jj = 0; jj < offsets_size; jj++) {
            GET_INDEX_AS_CONTIGUOUS_WITH_OFFSET(index, index_strides, offsets_run, nd, dims, index_strides, index_n, is_outside);

            if (!is_outside && nodes[index_n] != undef) {
                node = index_n;
                while (nodes[node] != nodes[nodes[node]]) {
                    nodes[node] = nodes[nodes[node]];
                }

                root = nodes[node];
                if (root != index) {
                    nodes[root] = index;
                    GET_INDEX_AS_CONTIGUOUS(root, index_strides, is_contiguous_p, itemsize_p, nd, strides_p, index_p);
                    SET_VALUE_TO(num_type_p, (pp + index_p), index);
                }
            }
            offsets_run += nd;
        }
    }

    for (ii = 0; ii < size; ii++) {
        index = traver[ii];

        // traverser[ii] = index
        SET_VALUE_TO(num_type_t, pt, index);
        ARRAY_ITER_NEXT(iter_t, pt);

        GET_INDEX_AS_CONTIGUOUS(index, index_strides, is_contiguous_p, itemsize_p, nd, strides_p, c_qi);
        // c_q = parent[index]
        GET_VALUE_AS(num_type_p, npy_intp, (pp + c_qi), c_q);

        GET_INDEX_AS_CONTIGUOUS(c_q, index_strides, is_contiguous_p, itemsize_p, nd, strides_p, c_qqi);
        // c_qq = parent[parent[index]] = parent[c_q]
        GET_VALUE_AS(num_type_p, npy_intp, (pp + c_qqi), c_qq);

        if (values_map) {
            GET_VALUE_AS(num_type_i, npy_intp, (pi_base + c_q * itemsize_i), c_ivq);
            GET_VALUE_AS(num_type_i, npy_intp, (pi_base + c_qq * itemsize_i), c_ivqq);

            if (c_ivq >= vm_size || c_ivqq >= vm_size) {
                PyErr_SetString(PyExc_RuntimeError, "con out of range for value map");
                NPY_END_THREADS;
                goto exit;
            }

            GET_VALUE_AS(num_type_vm, npy_intp, (vm + c_ivq * itemsize_vm), c_rq);
            GET_VALUE_AS(num_type_vm, npy_intp, (vm + c_ivqq * itemsize_vm), c_rqq);

            if (c_rq == c_rqq) {
                SET_VALUE_TO(num_type_p, (pp + c_qi), c_qq);
            }

        } else {
            GET_VALUE_AS(num_type_i, double, (pi_base + c_q * itemsize_i), c_vq);
            GET_VALUE_AS(num_type_i, double, (pi_base + c_qq * itemsize_i), c_vqq);

            if (c_vq == c_vqq) {
                SET_VALUE_TO(num_type_p, (pp + c_qi), c_qq);
            }
        }

    }

    NPY_END_THREADS;

    exit:
        free(traver);
        free(nodes);
        free(footprint);
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}

int ops_area_threshold(PyArrayObject *input,
                       int connectivity,
                       int threshold,
                       PyArrayObject *output,
                       PyArrayObject *traverser,
                       PyArrayObject *parent)
{
    char *pi = NULL, *pt = NULL, *pp = NULL, *po = NULL;
    npy_intp itemsize_p, itemsize_t, itemsize_i, itemsize_o, *o_strides, size, nd, index_strides[NPY_MAXDIMS], ii;
    int num_type_t, num_type_p, num_type_i, num_type_o;
    int o_is_contiguous;
    PyArrayObject *traverser_c, *parent_c, *input_c;

    npy_intp *area;
    // node = traverser[ii], c_p = parent[node], c_q = parent[parent[node]]
    npy_intp node = 0, c_p = 0, c_q = 0;
    // c_pv = pi[node], c_qv = pi[pi[node]], c_nv = pi[node]
    double out_val = 0, c_pv = 0, c_qv = 0, c_nv = 0;
    // root_t = traverser[0], p_index = index for root
    npy_intp root_t = 0, p_index = 0;

    nd = PyArray_NDIM(input);
    size = PyArray_SIZE(input);
    if (!traverser || !parent) {
        const npy_intp pt_dims[1] = {size};
        traverser_c = (PyArrayObject *)PyArray_EMPTY(1, pt_dims, NPY_INT64, 0);
        parent_c = (PyArrayObject *)PyArray_EMPTY(1, pt_dims, NPY_INT64, 0);
        if (!ops_build_max_tree(input, traverser_c, parent_c, connectivity, NULL)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: ops_build_max_tree \n");
            goto exit;
        }
    } else {
        traverser_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(traverser);
        parent_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(parent);
    }
    o_strides = PyArray_STRIDES(output);
    o_is_contiguous = PyArray_ISCONTIGUOUS(output);

    index_strides[nd - 1] = 1;
    for (ii = nd - 2; ii >= 0; ii--) {
        index_strides[ii] = index_strides[ii + 1] * PyArray_DIM(input, ii + 1);
    }

    area = (npy_intp *)malloc(size * sizeof(npy_intp));
    if (!area) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < size; ii++) {
        area[ii] = 1;
    }

    input_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(input);

    itemsize_t = PyArray_ITEMSIZE(traverser_c);
    itemsize_p = PyArray_ITEMSIZE(parent_c);
    itemsize_i = PyArray_ITEMSIZE(input_c);
    itemsize_o = PyArray_ITEMSIZE(output);

    num_type_t = PyArray_TYPE(traverser_c);
    num_type_p = PyArray_TYPE(parent_c);
    num_type_i = PyArray_TYPE(input_c);
    num_type_o = PyArray_TYPE(output);

    pi = (void *)PyArray_DATA(input_c);
    pt = (void *)PyArray_DATA(traverser_c);
    pp = (void *)PyArray_DATA(parent_c);
    po = (void *)PyArray_DATA(output);


    GET_VALUE_AS(num_type_p, npy_intp, pt, root_t);
    GET_INDEX_AS_CONTIGUOUS(root_t, index_strides, o_is_contiguous, itemsize_o, nd, o_strides, p_index);

    for (ii = size - 1; ii >= 0; ii--) {
        GET_VALUE_AS(num_type_t, npy_intp, (pt + ii * itemsize_t), node);

        if (node == root_t) {
            continue;
        }
        GET_VALUE_AS(num_type_p, npy_intp, (pp + node * itemsize_p), c_p);
        area[c_p] += area[node];

    }

    if (area[root_t] >= threshold) {
        GET_VALUE_AS(num_type_i, double, (pi + root_t * itemsize_i), out_val);
        SET_VALUE_TO(num_type_o, (po + p_index), out_val);
    } else {
        SET_VALUE_TO(num_type_o, (po + p_index), 0);
    }

    for (ii = 0; ii < size; ii++) {
        GET_VALUE_AS(num_type_t, npy_intp, pt, node);

        if (node != root_t) {
            GET_VALUE_AS(num_type_p, npy_intp, (pp + node * itemsize_p), c_p);

            GET_VALUE_AS(num_type_i, double, (pi + node * itemsize_i), c_nv);
            GET_VALUE_AS(num_type_i, double, (pi + c_p * itemsize_i), c_pv);

            if (c_nv == c_pv || area[node] < threshold) {

                GET_INDEX_AS_CONTIGUOUS(c_p, index_strides, o_is_contiguous, itemsize_o, nd, o_strides, p_index);
                GET_VALUE_AS(num_type_o, double, (po + p_index), out_val);

                GET_INDEX_AS_CONTIGUOUS(node, index_strides, o_is_contiguous, itemsize_o, nd, o_strides, p_index);
                SET_VALUE_TO(num_type_o, (po + p_index), out_val);

            } else {
                GET_INDEX_AS_CONTIGUOUS(node, index_strides, o_is_contiguous, itemsize_o, nd, o_strides, p_index);

                SET_VALUE_TO(num_type_o, (po + p_index), c_nv);
            }
        }
        pt += itemsize_t;
    }

    exit:
        free(area);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define DRAW_SET_YX(_y, _x, _pointer)                     \
{                                                         \
    *(npy_longlong *)_pointer = (npy_longlong)_y;         \
    _pointer += 8;                                        \
    *(npy_longlong *)_pointer = (npy_longlong)_x;         \
    _pointer += 8;                                        \
}

#define DRAW_SWAP_ARGS(_a1, _a2)                     \
{                                                    \
    npy_intp _tmp = _a1;                             \
    _a1 = _a2;                                       \
    _a2 = _tmp;                                      \
}

PyArrayObject *ops_draw_line(npy_intp *point1, npy_intp *point2)
{
    npy_intp size, y1, y2, x1, x2, dy, dx, step_y, step_x, p, yy, xx, m, ii;
    npy_intp p_dims[2];
    int flag;

    PyArrayObject *yx;
    char *pyx = NULL;

    NPY_BEGIN_THREADS_DEF;

    y1 = point1[0];
    x1 = point1[1];

    y2 = point2[0];
    x2 = point2[1];

    dy = abs(y2 - y1);
    dx = abs(x2 - x1);

    size = dx > dy ? dx + 1 : dy + 1;

    p_dims[0] = size;
    p_dims[1] = 2;

    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    step_y = y2 - y1 > 0 ? 1 : -1;
    step_x = x2 - x1 > 0 ? 1 : -1;

    flag = dy > dx ? 1 : 0;

    if (flag) {
        DRAW_SWAP_ARGS(x1, y1);
        DRAW_SWAP_ARGS(dx, dy);
        DRAW_SWAP_ARGS(step_x, step_y);
    }

    p = 2 * dy - dx;
    yy = y1;
    xx = x1;

    NPY_BEGIN_THREADS;

    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii <= dx; ii++) {
        if (flag) {
            DRAW_SET_YX(xx, yy, pyx);
        } else {
            DRAW_SET_YX(yy, xx, pyx);
        }

        xx += step_x;
        m = p >= 0 ? 1 : 0;

        p += (2 * dy) - (2 * dx) * m;
        yy += step_y * m;
    }

    NPY_END_THREADS;

    exit:
        return PyErr_Occurred() ? NULL : yx;
}

PyArrayObject *ops_draw_circle(npy_intp *center_point, int radius)
{
    npy_intp y0, x0, xx, yy = 0, size, ii, err = 0;

    npy_intp p_dims[2];
    PyArrayObject *yx;
    char *pyx = NULL;

    NPY_BEGIN_THREADS_DEF;

    y0 = center_point[0];
    x0 = center_point[1];

    xx = radius;

    size = (radius * 8) + 8;

    p_dims[0] = size;
    p_dims[1] = 2;

    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    NPY_BEGIN_THREADS;

    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii <= radius; ii++) {

        DRAW_SET_YX(y0 + yy, x0 + xx, pyx);
        DRAW_SET_YX(y0 + xx, x0 + yy, pyx);
        DRAW_SET_YX(y0 + xx, x0 - yy, pyx);
        DRAW_SET_YX(y0 + yy, x0 - xx, pyx);
        DRAW_SET_YX(y0 - yy, x0 - xx, pyx);
        DRAW_SET_YX(y0 - xx, x0 - yy, pyx);
        DRAW_SET_YX(y0 - xx, x0 + yy, pyx);
        DRAW_SET_YX(y0 - yy, x0 + xx, pyx);

        if (err + yy + 1 > xx) {
            err += 1 - 2 * xx;
            xx -= 1;
        } else {
            err += 1 + 2 * yy;
            yy += 1;
        }
    }

    NPY_END_THREADS;
    exit:
        return PyErr_Occurred() ? NULL : yx;
}

#define DRAW_ELLIPSE_PUSH_POINTS(_y0, _x0, _yy, _xx, _buffer, _buffer_end)                     \
{                                                                                              \
    _buffer[_buffer_end] = _y0 + _yy;                                                          \
    _buffer[_buffer_end + 1] = _x0 + _xx;                                                      \
    _buffer[_buffer_end + 2] = _y0 + _yy;                                                      \
    _buffer[_buffer_end + 3] = _x0 - _xx;                                                      \
    _buffer[_buffer_end + 4] = _y0 - _yy;                                                      \
    _buffer[_buffer_end + 5] = _x0 + _xx;                                                      \
    _buffer[_buffer_end + 6] = _y0 - _yy;                                                      \
    _buffer[_buffer_end + 7] = _x0 - _xx;                                                      \
    _buffer_end += 8;                                                                          \
}

PyArrayObject *ops_draw_ellipse(npy_intp *center_point, int a, int b)
{
    npy_intp y0, x0, ry, rx, tr_y, tr_x, yy, xx = 0, py, px = 0, ii, jj, p;
    npy_intp *buffer, max_size, buffer_end = 0, size;

    npy_intp p_dims[2];
    PyArrayObject *yx;
    char *pyx = NULL;

    y0 = center_point[0];
    x0 = center_point[1];

    rx = a * a;
    ry = b * b;

    tr_x = rx + rx;
    tr_y = ry + ry;

    yy = b;
    py = tr_x * yy;

    max_size = a > b ? a : b;
    max_size = max_size * 8 + 8;

    buffer = (npy_intp *)malloc(max_size * 2 * sizeof(npy_intp));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    DRAW_ELLIPSE_PUSH_POINTS(y0, x0, yy, xx, buffer, buffer_end);

    p = (npy_intp)((float)ry - (float)(rx * b) + (0.25 * (float)rx));

    while (px < py) {
        xx += 1;
        px += tr_y;
        if (p < 0) {
            p += ry + px;
        } else {
            yy -= 1;
            py -= tr_x;
            p += ry + px - py;
        }
        DRAW_ELLIPSE_PUSH_POINTS(y0, x0, yy, xx, buffer, buffer_end);
    }

    p = (npy_intp)((float)ry * ((float)xx + 0.5) * ((float)xx + 0.5) + (float)(rx * (yy - 1) * (yy - 1)) - (float)(rx * ry));

    while (yy > 0) {
        yy -= 1;
        py -= tr_x;
        if (p > 0) {
            p += rx - py;
        } else {
            xx += 1;
            px += tr_y;
            p += rx - py + px;
        }
        DRAW_ELLIPSE_PUSH_POINTS(y0, x0, yy, xx, buffer, buffer_end);
    }

    size = buffer_end / 2;
    p_dims[0] = size;
    p_dims[1] = 2;

    yx = (PyArrayObject *)PyArray_EMPTY(2, p_dims, NPY_INT64, 0);

    if (!yx) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    pyx = (void *)PyArray_DATA(yx);

    for (ii = 0; ii < size; ii++) {
        DRAW_SET_YX(buffer[ii * 2], buffer[ii * 2 + 1], pyx);
    }

    exit:
        free(buffer);
        return PyErr_Occurred() ? NULL : yx;
}

// #####################################################################################################################










