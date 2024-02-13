#include "c_pycv_base.h"
#include "c_pycv_maxtree.h"

// #####################################################################################################################

int PYCV_build_max_tree(PyArrayObject *input,
                        PyArrayObject *traverser,
                        PyArrayObject *parent,
                        npy_intp connectivity)
{
    NeighborhoodIterator iter_i;
    PyArray_ArgSortFunc *arg_sort_func;
    PyArrayObject *input_c, *traverser_c, *parent_c;
    char *pi_base = NULL, *pi = NULL, *pt = NULL,  *pp = NULL;
    int num_type_p, num_type_t, num_type_i;
    npy_intp itemsize_t, itemsize_p, itemsize_i;
    npy_bool *footprint;
    npy_intp ndim, a_size, f_size, f_shape[NPY_MAXDIMS], f_center[NPY_MAXDIMS], *offsets, *ff, flag;
    npy_intp ii, jj;
    npy_intp *nodes, *s_index;
    npy_intp coordinates[NPY_MAXDIMS], ind, p_ind, pp_ind, n, r;
    npy_double p_ind_v, pp_ind_v;

    NPY_BEGIN_THREADS_DEF;

    input_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(input);
    a_size = PyArray_SIZE(input_c);
    ndim = PyArray_NDIM(input_c);

    for (ii = 0; ii < ndim; ii++) {
        f_shape[ii] = 3;
        f_center[ii] = 1;
    }

    if (!PYCV_DefaultFootprint(ndim, connectivity, &footprint, &f_size, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_DefaultFootprint \n");
        goto exit;
    }

    PYCV_NeighborhoodIteratorInit(input_c, f_shape, f_center, f_size, &iter_i);

    if (!PYCV_InitNeighborhoodOffsets(input_c, f_shape, f_center, footprint, &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }

    parent_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(parent);
    traverser_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(traverser);

    itemsize_p = PyArray_ITEMSIZE(parent_c);
    itemsize_t = PyArray_ITEMSIZE(traverser_c);
    itemsize_i = PyArray_ITEMSIZE(input_c);

    num_type_p = PyArray_TYPE(parent_c);
    num_type_t = PyArray_TYPE(traverser_c);
    num_type_i = PyArray_TYPE(input_c);

    nodes = malloc(a_size * sizeof(npy_intp));
    if (!nodes) {
        PyErr_NoMemory();
        goto exit;
    }

    s_index = malloc(a_size * sizeof(npy_intp));
    if (!s_index) {
        PyErr_NoMemory();
        goto exit;
    }
    for (ii = 0; ii < a_size; ii++) {
        s_index[ii] = ii;
        nodes[ii] = -1;
    }

    pi_base = pi = (void *)PyArray_DATA(input_c);
    pt = (void *)PyArray_DATA(traverser_c);
    pp = (void *)PyArray_DATA(parent_c);
    ff = offsets;

    arg_sort_func = PyArray_DESCR(input_c)->f->argsort[NPY_MERGESORT];

    if (!arg_sort_func || arg_sort_func(pi, s_index, a_size, input_c) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Couldn't perform argsort.\n");
        goto exit;
    }

    NPY_BEGIN_THREADS;

    for (ii = a_size - 1; ii >= 0; ii--) {
        nodes[s_index[ii]] = s_index[ii];

        PYCV_SET_VALUE(num_type_p, (pp + s_index[ii] * itemsize_p), s_index[ii]);

        ind = s_index[ii] * itemsize_i;
        for (jj = 0; jj < ndim; jj++) {
            coordinates[jj] = ind / iter_i.strides[jj];
            ind -= coordinates[jj] * iter_i.strides[jj];
        }

        PYCV_NEIGHBORHOOD_ITERATOR_GOTO(iter_i, pi_base, pi, offsets, ff, coordinates);

        for (jj = 0; jj < f_size; jj++) {
            ind = s_index[ii] + ff[jj] / itemsize_i;
            if (!ff[jj] || ff[jj] == flag || nodes[ind] == -1) {
                continue;
            }
            n = nodes[ind];
            while (nodes[n] != nodes[nodes[n]]) {
                nodes[n] = nodes[nodes[n]];
            }
            r = nodes[n];
            if (r != s_index[ii]) {
                nodes[r] = s_index[ii];
                PYCV_SET_VALUE(num_type_p, (pp + r * itemsize_p), s_index[ii]);
            }
        }
    }

    pi = pi_base;
    for (ii = 0; ii < a_size; ii++) {
        PYCV_SET_VALUE(num_type_t, pt, s_index[ii]);
        PYCV_GET_VALUE(num_type_p, npy_intp, (pp + s_index[ii] * itemsize_p), p_ind);
        PYCV_GET_VALUE(num_type_p, npy_intp, (pp + p_ind * itemsize_p), pp_ind);

        PYCV_GET_VALUE(num_type_i, npy_double, (pi + p_ind * itemsize_i), p_ind_v);
        PYCV_GET_VALUE(num_type_i, npy_double, (pi + pp_ind * itemsize_i), pp_ind_v);

        if (p_ind_v == pp_ind_v) {
            PYCV_SET_VALUE(num_type_p, (pp + s_index[ii] * itemsize_p), pp_ind);
        }
        pt += itemsize_t;
    }

    NPY_END_THREADS;

    exit:
        free(footprint);
        free(offsets);
        free(nodes);
        free(s_index);
        return PyErr_Occurred() ? 0 : 1;
}

#define PYCV_M_CASE_MAX_TREE_AREA_ADD(_NTYPE, _dtype, _pointer, _ind_set, _ind_get)                                    \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    *(_dtype *)(_pointer + _ind_set) += (_dtype)(*((_dtype *)(_pointer + _ind_get)));                                  \
}                                                                                                                      \
break

#define PYCV_M_MAX_TREE_AREA_ADD(_NTYPE, _pointer, _ind_set, _ind_get)                                                 \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(BOOL, npy_bool, _pointer, _ind_set, _ind_get);                                   \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(UBYTE, npy_ubyte, _pointer, _ind_set, _ind_get);                                 \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(USHORT, npy_ushort, _pointer, _ind_set, _ind_get);                               \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(UINT, npy_uint, _pointer, _ind_set, _ind_get);                                   \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(ULONG, npy_ulong, _pointer, _ind_set, _ind_get);                                 \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(ULONGLONG, npy_ulonglong, _pointer, _ind_set, _ind_get);                         \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(BYTE, npy_byte, _pointer, _ind_set, _ind_get);                                   \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(SHORT, npy_short, _pointer, _ind_set, _ind_get);                                 \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(INT, npy_int, _pointer, _ind_set, _ind_get);                                     \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(LONG, npy_long, _pointer, _ind_set, _ind_get);                                   \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(LONGLONG, npy_longlong, _pointer, _ind_set, _ind_get);                           \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(FLOAT, npy_float, _pointer, _ind_set, _ind_get);                                 \
        PYCV_M_CASE_MAX_TREE_AREA_ADD(DOUBLE, npy_double, _pointer, _ind_set, _ind_get);                               \
    }                                                                                                                  \
}


int PYCV_max_tree_compute_area(PyArrayObject *input,
                               PyArrayObject *output,
                               npy_intp connectivity,
                               PyArrayObject *traverser,
                               PyArrayObject *parent)
{
    PyArrayObject *traverser_c, *parent_c, *output_c;
    npy_intp a_size;
    char *po = NULL, *po_base = NULL, *pt = NULL,  *pp = NULL;
    int num_type_p, num_type_t, num_type_o;
    npy_intp itemsize_t, itemsize_p, itemsize_o;
    npy_intp p_ind, n_ind, r_ind, ii;

    NPY_BEGIN_THREADS_DEF;

    if ((!traverser || !parent) && !input){
        PyErr_SetString(PyExc_RuntimeError, "Error: input or traverser and parent need to be given \n");
        goto exit;
    } else if (!traverser || !parent) {
        a_size = PyArray_SIZE(input);
        npy_intp dims[1] = {a_size};
        traverser_c = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
        parent_c = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
        if (!PYCV_build_max_tree(input, traverser_c, parent_c, connectivity)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_build_max_tree \n");
            goto exit;
        }
    } else {
        a_size = PyArray_SIZE(traverser);
        traverser_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(traverser);
        parent_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(parent);
    }

    output_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(output);

    itemsize_p = PyArray_ITEMSIZE(parent_c);
    itemsize_t = PyArray_ITEMSIZE(traverser_c);
    itemsize_o = PyArray_ITEMSIZE(output_c);

    num_type_p = PyArray_TYPE(parent_c);
    num_type_t = PyArray_TYPE(traverser_c);
    num_type_o = PyArray_TYPE(output_c);

    NPY_BEGIN_THREADS;

    pt = (void *)PyArray_DATA(traverser_c);
    pp = (void *)PyArray_DATA(parent_c);
    po_base = po = (void *)PyArray_DATA(output_c);

    for (ii = 0; ii < a_size; ii++) {
        PYCV_SET_VALUE(num_type_o, po, 1);
        po += itemsize_o;
    }
    po = po_base;

    PYCV_GET_VALUE(num_type_t, npy_intp, pt, r_ind);

    for (ii = 0; ii < a_size; ii++) {
        PYCV_GET_VALUE(num_type_t, npy_intp, (pt + ii * itemsize_t), n_ind);
        if (r_ind == n_ind) {
            continue;
        }
        PYCV_GET_VALUE(num_type_p, npy_intp, (pp + n_ind * itemsize_p), p_ind);
        PYCV_M_MAX_TREE_AREA_ADD(num_type_o, po, (p_ind * itemsize_o), (n_ind * itemsize_o));
    }

    NPY_END_THREADS;

    exit:
        return PyErr_Occurred() ? 0 : 1;
}


int PYCV_max_tree_filter(PyArrayObject *input,
                         npy_double threshold,
                         PyArrayObject *values_map,
                         PyArrayObject *output,
                         npy_intp connectivity,
                         PyArrayObject *traverser,
                         PyArrayObject *parent)
{
    PyArrayObject *traverser_c, *parent_c, *output_c, *input_c, *values_map_c;
    npy_intp a_size;
    char *po = NULL, *pt = NULL,  *pp = NULL, *pi = NULL,  *pvm = NULL;
    int num_type_p, num_type_t, num_type_o, num_type_i, num_type_vm;
    npy_intp itemsize_t, itemsize_p, itemsize_o, itemsize_i, itemsize_vm;
    npy_intp p_ind, n_ind, r_ind, ii;
    npy_double m_val, o_val, n_val, p_val;

    NPY_BEGIN_THREADS_DEF;

    a_size = PyArray_SIZE(input);

    if (!traverser || !parent) {
        npy_intp dims[1] = {a_size};
        traverser_c = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
        parent_c = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
        if (!PYCV_build_max_tree(input, traverser_c, parent_c, connectivity)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_build_max_tree \n");
            goto exit;
        }
    } else {
        traverser_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(traverser);
        parent_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(parent);
    }

    input_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(input);
    output_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(output);
    values_map_c = (PyArrayObject *)PyArray_GETCONTIGUOUS(values_map);

    itemsize_p = PyArray_ITEMSIZE(parent_c);
    itemsize_t = PyArray_ITEMSIZE(traverser_c);
    itemsize_o = PyArray_ITEMSIZE(output_c);
    itemsize_i = PyArray_ITEMSIZE(input_c);
    itemsize_vm = PyArray_ITEMSIZE(values_map_c);

    num_type_p = PyArray_TYPE(parent_c);
    num_type_t = PyArray_TYPE(traverser_c);
    num_type_o = PyArray_TYPE(output_c);
    num_type_i = PyArray_TYPE(input_c);
    num_type_vm = PyArray_TYPE(values_map_c);

    NPY_BEGIN_THREADS;

    pt = (void *)PyArray_DATA(traverser_c);
    pp = (void *)PyArray_DATA(parent_c);
    po = (void *)PyArray_DATA(output_c);
    pi = (void *)PyArray_DATA(input_c);
    pvm = (void *)PyArray_DATA(values_map_c);

    PYCV_GET_VALUE(num_type_t, npy_intp, pt, r_ind);
    PYCV_GET_VALUE(num_type_vm, npy_double, (pvm + r_ind * itemsize_vm), m_val);

    if (m_val >= threshold) {
        PYCV_GET_VALUE(num_type_i, npy_double, (pi + r_ind * itemsize_i), o_val);
        PYCV_SET_VALUE_F2A(num_type_o, (po + r_ind * itemsize_o), o_val);
    } else {
        PYCV_SET_VALUE(num_type_i, (po + r_ind * itemsize_o), 0);
    }

    for (ii = 0; ii < a_size; ii++) {
        PYCV_GET_VALUE(num_type_t, npy_intp, pt, n_ind);
        if (n_ind != r_ind) {
            PYCV_GET_VALUE(num_type_p, npy_intp, (pp + n_ind * itemsize_p), p_ind);
            PYCV_GET_VALUE(num_type_vm, npy_double, (pvm + n_ind * itemsize_vm), m_val);

            PYCV_GET_VALUE(num_type_i, npy_double, (pi + n_ind * itemsize_i), n_val);
            PYCV_GET_VALUE(num_type_i, npy_double, (pi + p_ind * itemsize_i), p_val);

            if (n_val == p_val || m_val < threshold) {
                PYCV_GET_VALUE(num_type_o, npy_double, (po + p_ind * itemsize_i), o_val);
                PYCV_SET_VALUE_F2A(num_type_o, (po + n_ind * itemsize_o), o_val);
            } else {
                PYCV_SET_VALUE_F2A(num_type_o, (po + n_ind * itemsize_o), n_val);
            }
        }
        pt += itemsize_p;
    }
    NPY_END_THREADS;

    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################




















