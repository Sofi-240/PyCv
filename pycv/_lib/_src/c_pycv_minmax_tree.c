#include "c_pycv_base.h"
#include "c_pycv_minmax_tree.h"

// #####################################################################################################################

#define MM_DTYPE npy_longlong

#define MM_CON_DTYPE NPY_INT64

#define MM_ITEMSIZE (int)NPY_SIZEOF_LONGLONG

#define MM_UNDEF -1

// #####################################################################################################################

static void heapsort_swap(char *p1, char *p2)
{
    npy_longlong tmp = *(npy_longlong *)p1;
    *(npy_longlong *)p1 = *(npy_longlong *)p2;
    *(npy_longlong *)p2 = tmp;
}

#define MM_CASE_HEAPSORT_COMPARE(_NTYPE, _dtype, _itemsize, _heap, _priority, _c1, _c2, _max, _out)                    \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    int _h1 = (int)(*(MM_DTYPE *)(_heap + _c1 * MM_ITEMSIZE));                                                         \
    int _h2 = (int)(*(MM_DTYPE *)(_heap + _c2 * MM_ITEMSIZE));                                                         \
    _out = 0;                                                                                                          \
    if ((_max && *(_dtype *)(_priority + _h1 * _itemsize) < *(_dtype *)(_priority + _h2 * _itemsize)) ||               \
        (!_max && *(_dtype *)(_priority + _h1 * _itemsize) > *(_dtype *)(_priority + _h2 * _itemsize)) ||              \
        (*(_dtype *)(_priority + _h1 * _itemsize) == *(_dtype *)(_priority + _h2 * _itemsize) && _h1 < _h2)) {         \
        _out = 1;                                                                                                      \
    }                                                                                                                  \
}                                                                                                                      \
break

#define MM_HEAPSORT_COMPARE(_NTYPE, _itemsize, _heap, _priority, _c1, _c2, _max, _out)                                 \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        MM_CASE_HEAPSORT_COMPARE(BOOL, npy_bool, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                   \
        MM_CASE_HEAPSORT_COMPARE(UBYTE, npy_ubyte, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                 \
        MM_CASE_HEAPSORT_COMPARE(USHORT, npy_ushort, _itemsize, _heap, _priority, _c1, _c2, _max, _out);               \
        MM_CASE_HEAPSORT_COMPARE(UINT, npy_uint, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                   \
        MM_CASE_HEAPSORT_COMPARE(ULONG, npy_ulong, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                 \
        MM_CASE_HEAPSORT_COMPARE(ULONGLONG, npy_ulonglong, _itemsize, _heap, _priority, _c1, _c2, _max, _out);         \
        MM_CASE_HEAPSORT_COMPARE(BYTE, npy_byte, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                   \
        MM_CASE_HEAPSORT_COMPARE(SHORT, npy_short, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                 \
        MM_CASE_HEAPSORT_COMPARE(INT, npy_int, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                     \
        MM_CASE_HEAPSORT_COMPARE(LONG, npy_long, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                   \
        MM_CASE_HEAPSORT_COMPARE(LONGLONG, npy_longlong, _itemsize, _heap, _priority, _c1, _c2, _max, _out);           \
        MM_CASE_HEAPSORT_COMPARE(FLOAT, npy_float, _itemsize, _heap, _priority, _c1, _c2, _max, _out);                 \
        MM_CASE_HEAPSORT_COMPARE(DOUBLE, npy_double, _itemsize, _heap, _priority, _c1, _c2, _max, _out);               \
    }                                                                                                                  \
}

static void minmax_heapsort_traverser(CMinMaxTree *self)
{
    char *priority = NULL, *heap = NULL;
    int numtype, itemsize, finish_max_heap = 0, ii, end_pos = self->size, child_pos, right_child, parent_pos, cmp;

    numtype = (int)PyArray_TYPE(self->data);
    itemsize = (int)PyArray_ITEMSIZE(self->data);

    priority = (void *)PyArray_DATA(self->data);
    heap = (void *)PyArray_DATA(self->traverser);

    ii = end_pos / 2 + 1;

    while (ii >= 0 && end_pos >= 0) {

        if (finish_max_heap) {
            heapsort_swap(heap, heap + end_pos * MM_ITEMSIZE);
        }
        parent_pos = ii;
        child_pos = 2 * parent_pos + 1;
        while (child_pos < end_pos) {
            right_child = child_pos + 1;

            if (right_child < end_pos) {
                MM_HEAPSORT_COMPARE(numtype, itemsize, heap, priority, child_pos, right_child, self->_is_max, cmp);
                child_pos = cmp ? child_pos + 1 : child_pos;
            }
            MM_HEAPSORT_COMPARE(numtype, itemsize, heap, priority, parent_pos, child_pos, self->_is_max, cmp);

            if (cmp) {
                heapsort_swap(heap + child_pos * MM_ITEMSIZE, heap + parent_pos * MM_ITEMSIZE);
                parent_pos = child_pos;
                child_pos = 2 * parent_pos + 1;
                continue;
            }
            break;
        }

        ii--;
        if (ii < 0 && !finish_max_heap) {
            finish_max_heap = 1;
        }
        if (finish_max_heap) {
            ii = 0;
            end_pos--;
        }
    }
}

// *********************************************************************************************************************

typedef struct {
    int n;
    npy_intp *offsets;
    npy_intp flag;
    NeighborhoodIterator iterator;
} build_iterator;

static int build_iterator_init(build_iterator *self, PyArrayObject *array, int connectivity)
{
    npy_intp *shape = NULL, *center = NULL;
    int ii, ndim;
    npy_bool *footprint;
    npy_intp n;

    ndim = (int)PyArray_NDIM(array);

    if (!PYCV_DefaultFootprint((npy_intp)ndim, (npy_intp)connectivity, &footprint, &n, 0)) {
        self->n = 0;
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_DefaultFootprint \n");
        return 0;
    }

    self->n = (int)n;

    shape = malloc(ndim * 2 * sizeof(npy_intp));
    if (!shape) {
        self->n = 0;
        free(footprint);
        PyErr_NoMemory();
        return 0;
    }
    center = shape + ndim;

    for (ii = 0; ii < ndim; ii++) {
        *(shape + ii) = 3;
        *(center + ii) = 1;
    }

    if (!PYCV_InitNeighborhoodOffsets(array, shape, center, footprint, &(self->offsets), NULL, &(self->flag), PYCV_EXTEND_CONSTANT)) {
        self->n = 0;
        free(footprint);
        free(shape);
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        return 0;
    }

    PYCV_NeighborhoodIteratorInit(array, shape, center, self->n, &(self->iterator));
    free(footprint);
    free(shape);
    return 1;
}

static void build_iterator_free(build_iterator *self)
{
    if (self->n) {
        free(self->offsets);
        self->n = 0;
    }
}

// *********************************************************************************************************************

#define MM_CASE_IS_EQUAL(_NTYPE, _dtype, _p1, _p2, _out)                                                               \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    _out = *(_dtype *)_p1 == *(_dtype *)_p2 ? 1 : 0;                                                                   \
}                                                                                                                      \
break

#define MM_IS_EQUAL(_NTYPE, _p1, _p2, _out)                                                                            \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        MM_CASE_IS_EQUAL(BOOL, npy_bool, _p1, _p2, _out);                                                              \
        MM_CASE_IS_EQUAL(UBYTE, npy_ubyte, _p1, _p2, _out);                                                            \
        MM_CASE_IS_EQUAL(USHORT, npy_ushort, _p1, _p2, _out);                                                          \
        MM_CASE_IS_EQUAL(UINT, npy_uint, _p1, _p2, _out);                                                              \
        MM_CASE_IS_EQUAL(ULONG, npy_ulong, _p1, _p2, _out);                                                            \
        MM_CASE_IS_EQUAL(ULONGLONG, npy_ulonglong, _p1, _p2, _out);                                                    \
        MM_CASE_IS_EQUAL(BYTE, npy_byte, _p1, _p2, _out);                                                              \
        MM_CASE_IS_EQUAL(SHORT, npy_short, _p1, _p2, _out);                                                            \
        MM_CASE_IS_EQUAL(INT, npy_int, _p1, _p2, _out);                                                                \
        MM_CASE_IS_EQUAL(LONG, npy_long, _p1, _p2, _out);                                                              \
        MM_CASE_IS_EQUAL(LONGLONG, npy_longlong, _p1, _p2, _out);                                                      \
        MM_CASE_IS_EQUAL(FLOAT, npy_float, _p1, _p2, _out);                                                            \
        MM_CASE_IS_EQUAL(DOUBLE, npy_double, _p1, _p2, _out);                                                          \
    }                                                                                                                  \
}

// *********************************************************************************************************************

static int mm_build_tree(CMinMaxTree *self)
{
    build_iterator iterator;
    char *ptr_t = NULL, *ptr_t_base = NULL, *ptr_n = NULL, *ptr_n_base = NULL;
    char *ptr_dei = NULL, *ptr_dpi = NULL, *ptr_d_base = NULL;
    int *edges = NULL, numtype, itemsize;
    int ii, jj, vi, vi_sized, vj, ej, ei, pi, cmp;
    npy_intp *offsets;

    if (!build_iterator_init(&iterator, self->nodes, self->connectivity)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: build_iterator_init \n");
        goto exit;
    }

    edges = malloc(self->size * sizeof(int));
    if (!edges) {
        PyErr_NoMemory();
        goto exit;
    }

    ptr_d_base = (void *)PyArray_DATA(self->data);
    ptr_t_base = ptr_t = (void *)PyArray_DATA(self->traverser);
    ptr_n_base = ptr_n = (void *)PyArray_DATA(self->nodes);

    for (ii = 0; ii < self->size; ii++) {
        *(edges + ii) = (int)MM_UNDEF;
        *(MM_DTYPE *)ptr_t = (MM_DTYPE)ii;
        *(MM_DTYPE *)ptr_n = (MM_DTYPE)MM_UNDEF;
        ptr_t += MM_ITEMSIZE;
        ptr_n += MM_ITEMSIZE;
    }

    minmax_heapsort_traverser(self);

    offsets = iterator.offsets;
    numtype = (int)PyArray_TYPE(self->data);
    itemsize = (int)PyArray_ITEMSIZE(self->data);

    ptr_n = ptr_n_base;
    ptr_t = ptr_t_base + (self->size - 1) * MM_ITEMSIZE;

    for (ii = 0; ii < self->size; ii++) {
        vi = (int)(*(MM_DTYPE *)ptr_t);
        vi_sized = vi * MM_ITEMSIZE;

        PYCV_NEIGHBORHOOD_ITERATOR_GOTO_RAVEL(iterator.iterator, ptr_n_base, ptr_n, iterator.offsets, offsets, vi_sized);

        *(MM_DTYPE *)ptr_n = (MM_DTYPE)vi;
        *(edges + vi) = vi;

        for (jj = 0; jj < iterator.n; jj++) {
            vj = vi + ((int)(*(offsets + jj)) / MM_ITEMSIZE);

            if (!*(offsets + jj) || *(offsets + jj) == iterator.flag || *(edges + vj) == (int)MM_UNDEF) {
                continue;
            }

            ej = *(edges + vj);
            while (*(edges + ej) != *(edges + *(edges + ej))) {
                *(edges + ej) = *(edges + *(edges + ej));
            }

            ej = *(edges + ej);
            if (ej != vi) {
                *(edges + ej) = vi;
                *(MM_DTYPE *)(ptr_n_base + ej * MM_ITEMSIZE) = (MM_DTYPE)vi;
            }
        }
        ptr_t -= MM_ITEMSIZE;
    }

    ptr_t = ptr_t_base;
    ptr_n = ptr_n_base;

    for (ii = 0; ii < self->size; ii++) {
        vi = (int)(*(MM_DTYPE *)ptr_t);
        vi_sized = vi * MM_ITEMSIZE;

        ei = (int)(*(MM_DTYPE *)(ptr_n + vi_sized));

        pi = (int)(*(MM_DTYPE *)(ptr_n + (ei * MM_ITEMSIZE)));

        ptr_dei = ptr_d_base + ei * itemsize;
        ptr_dpi = ptr_d_base + pi * itemsize;

        MM_IS_EQUAL(numtype, ptr_dei, ptr_dpi, cmp);

        if (cmp) {
            *(MM_DTYPE *)(ptr_n + vi_sized) = (MM_DTYPE)pi;
        }
        ptr_t += MM_ITEMSIZE;
    }

    exit:
        build_iterator_free(&iterator);
        if (edges) {
            free(edges);
        }
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

static void mm_compute_area(CMinMaxTree *self, char *output)
{
    char *ptr_t = NULL, *ptr_n = NULL, *po = output;
    int ii, root, vi, vi_sized, vj;

    ptr_t = (void *)PyArray_DATA(self->traverser);
    ptr_n = (void *)PyArray_DATA(self->nodes);

    for (ii = 0; ii < self->size; ii++) {
        *(MM_DTYPE *)po = 1;
        po += MM_ITEMSIZE;
    }

    po = output;
    root = (int)*(MM_DTYPE *)ptr_t;
    ptr_t = ptr_t + (self->size - 1) * MM_ITEMSIZE;

    for (ii = self->size - 1; ii >= 0; ii--) {
        vi = (int)*(MM_DTYPE *)ptr_t;
        ptr_t -= MM_ITEMSIZE;

        if (vi == root) {
            continue;
        }

        vi_sized = vi * MM_ITEMSIZE;
        vj = (int)(*(MM_DTYPE *)(ptr_n + vi_sized));
        *(MM_DTYPE *)(po + vj * MM_ITEMSIZE) = *(MM_DTYPE *)(po + vj * MM_ITEMSIZE) + *(MM_DTYPE *)(po + vi_sized);
    }
}

// #####################################################################################################################

#define MM_CASE_FILTER_SET_FROM(_NTYPE, _dtype, _src, _dst)                                                            \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    *(_dtype *)_dst = *(_dtype *)_src;                                                                                 \
}                                                                                                                      \
break

#define MM_FILTER_SET_FROM(_NTYPE, _src, _dst)                                                                         \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        MM_CASE_FILTER_SET_FROM(BOOL, npy_bool, _src, _dst);                                                           \
        MM_CASE_FILTER_SET_FROM(UBYTE, npy_ubyte, _src, _dst);                                                         \
        MM_CASE_FILTER_SET_FROM(USHORT, npy_ushort, _src, _dst);                                                       \
        MM_CASE_FILTER_SET_FROM(UINT, npy_uint, _src, _dst);                                                           \
        MM_CASE_FILTER_SET_FROM(ULONG, npy_ulong, _src, _dst);                                                         \
        MM_CASE_FILTER_SET_FROM(ULONGLONG, npy_ulonglong, _src, _dst);                                                 \
        MM_CASE_FILTER_SET_FROM(BYTE, npy_byte, _src, _dst);                                                           \
        MM_CASE_FILTER_SET_FROM(SHORT, npy_short, _src, _dst);                                                         \
        MM_CASE_FILTER_SET_FROM(INT, npy_int, _src, _dst);                                                             \
        MM_CASE_FILTER_SET_FROM(LONG, npy_long, _src, _dst);                                                           \
        MM_CASE_FILTER_SET_FROM(LONGLONG, npy_longlong, _src, _dst);                                                   \
        MM_CASE_FILTER_SET_FROM(FLOAT, npy_float, _src, _dst);                                                         \
        MM_CASE_FILTER_SET_FROM(DOUBLE, npy_double, _src, _dst);                                                       \
    }                                                                                                                  \
}

static void mm_tree_filter_from_values_map(CMinMaxTree *self, PyArrayObject *values_map, double threshold, char *output)
{
    PYCV_ArrayIterator iterator;
    char *ptr_t = NULL, *ptr_m = NULL, *ptr_m_base = NULL, *ptr_n = NULL, *ptr_d = NULL, *ptr_o = output;
    int ii, root, root_sized, vi, vi_sized, vj, numtype, itemsize;
    double val, vi_val, vj_val;

    numtype = (int)PyArray_TYPE(self->data);
    itemsize = (int)PyArray_ITEMSIZE(self->data);

    ptr_t = (void *)PyArray_DATA(self->traverser);
    ptr_n = (void *)PyArray_DATA(self->nodes);
    ptr_d = (void *)PyArray_DATA(self->data);
    ptr_m_base = ptr_m = (void *)PyArray_DATA(values_map);

    PYCV_ArrayIteratorInit(values_map, &iterator);

    root = (int)*(MM_DTYPE *)ptr_t;
    root_sized = root * (int)iterator.strides[iterator.nd_m1];

    PYCV_ARRAY_ITERATOR_GOTO_RAVEL(iterator, ptr_m_base, ptr_m, root_sized);

    PYCV_GET_VALUE(iterator.numtype, double, ptr_m, val);

    if (val >= threshold) {
        MM_FILTER_SET_FROM(numtype, (ptr_d + root * itemsize), (ptr_o + root * itemsize));
    } else {
        PYCV_SET_VALUE(numtype, (ptr_o + root * itemsize), 0);
    }

    for (ii = 0; ii < self->size; ii++) {
        vi = (int)*(MM_DTYPE *)ptr_t;
        ptr_t += MM_ITEMSIZE;
        if (vi == root) {
            continue;
        }

        vi_sized = vi * (int)iterator.strides[iterator.nd_m1];
        PYCV_ARRAY_ITERATOR_GOTO_RAVEL(iterator, ptr_m_base, ptr_m, vi_sized);
        PYCV_GET_VALUE(iterator.numtype, double, ptr_m, val);

        vj = (int)(*(MM_DTYPE *)(ptr_n + vi * MM_ITEMSIZE));

        PYCV_GET_VALUE(numtype, double, (ptr_d + vi * itemsize), vi_val);
        PYCV_GET_VALUE(numtype, double, (ptr_d + vj * itemsize), vj_val);

        if (vi_val == vj_val || val < threshold) {
            MM_FILTER_SET_FROM(numtype, (ptr_o + vj * itemsize), (ptr_o + vi * itemsize));
        } else {
            PYCV_SET_VALUE_F2A(numtype, (ptr_o + vi * itemsize), vi_val);
        }
    }
}

// #####################################################################################################################

static int mm_label_image(CMinMaxTree *self, char *output)
{
    char *ptr_t = NULL, *ptr_n = NULL, *ptr_d = NULL, *ptr_o = output;
    int ii, root, vi, vj, n_labels = 1, numtype, itemsize;
    double val, vi_val, vj_val;

    numtype = (int)PyArray_TYPE(self->data);
    itemsize = (int)PyArray_ITEMSIZE(self->data);

    ptr_t = (void *)PyArray_DATA(self->traverser);
    ptr_n = (void *)PyArray_DATA(self->nodes);
    ptr_d = (void *)PyArray_DATA(self->data);

    for (ii = 0; ii < self->size; ii++) {
        *(MM_DTYPE *)ptr_o = 0;
        ptr_o += MM_ITEMSIZE;
    }

    ptr_o = output;
    root = (int)(*(MM_DTYPE *)ptr_t);

    PYCV_GET_VALUE(numtype, double, (ptr_d + root * itemsize), val);
    if (val) {
        *(MM_DTYPE *)(ptr_o + root * MM_ITEMSIZE) = 1;
        n_labels++;
    }
    ptr_t += MM_ITEMSIZE;

    for (ii = 1; ii < self->size; ii++) {
        vi = (int)(*(MM_DTYPE *)ptr_t);
        ptr_t += MM_ITEMSIZE;

        vj = (int)(*(MM_DTYPE *)(ptr_n + vi * MM_ITEMSIZE));

        PYCV_GET_VALUE(numtype, double, (ptr_d + vi * itemsize), vi_val);
        PYCV_GET_VALUE(numtype, double, (ptr_d + vj * itemsize), vj_val);

        if (vi_val == vj_val) {
            *(MM_DTYPE *)(ptr_o + vi * MM_ITEMSIZE) = *(MM_DTYPE *)(ptr_o + vj * MM_ITEMSIZE);
        } else {
            *(MM_DTYPE *)(ptr_o + vi * MM_ITEMSIZE) = (MM_DTYPE)n_labels;
            n_labels++;
        }
    }
    return n_labels - 1;
}

// #####################################################################################################################

void CMinMaxTreePy_dealloc(CMinMaxTree *self)
{
    Py_XDECREF(self->data);
    Py_XDECREF(self->traverser);
    Py_XDECREF(self->nodes);
    if (self->dims) {
        Py_TYPE(self->dims)->tp_free((PyObject *)(self->dims));
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *CMinMaxTreePy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CMinMaxTree *self;
    self = (CMinMaxTree *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->connectivity = 1;
        self->ndim = 0;
        self->size = 0;
        self->_is_max = 1;

        self->dims = NULL;
        self->data = NULL;
        self->traverser = NULL;
        self->nodes = NULL;
    }

    return (PyObject *)self;
}

static int mm_input_data(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    if (*output != NULL) {
        *output = (PyArrayObject *)PyArray_GETCONTIGUOUS(*output);
    }
    return *output != NULL;
}

int CMinMaxTreePy_init(CMinMaxTree *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"", "connectivity", "max_tree", NULL};
    int ii;
    npy_intp dims[1] = {0};

    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "O&|ii", kwlist,
            mm_input_data, &(self->data), &(self->connectivity), &(self->_is_max))
        ) {
        return -1;
    }

    self->ndim = (int)PyArray_NDIM(self->data);
    self->size = (int)PyArray_SIZE(self->data);
    self->dims = PyTuple_New((Py_ssize_t)self->ndim);

    for (ii = 0; ii < self->ndim; ii++) {
        if (PyTuple_SetItem(self->dims, (Py_ssize_t)ii, Py_BuildValue("i", (int)PyArray_DIM(self->data, ii)))) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyTuple_SetItem \n");
            return -1;
        }
    }

    dims[0] = (npy_intp)self->size;
    self->traverser = (PyArrayObject *)PyArray_EMPTY(1, dims, MM_CON_DTYPE, 0);
    self->nodes = (PyArrayObject *)PyArray_EMPTY(self->ndim, PyArray_DIMS(self->data), MM_CON_DTYPE, 0);

    if (!self->traverser || !self->nodes) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return -1;
    }

    if (!mm_build_tree(self)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: mm_build_tree");
        return -1;
    }

    return 0;
}

PyObject *CMinMaxTreePy_compute_area(CMinMaxTree *self)
{
    PyArrayObject *output;
    char *ptr = NULL;

    output = (PyArrayObject *)PyArray_EMPTY(self->ndim, PyArray_DIMS(self->data), MM_CON_DTYPE, 0);
    if (!output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return Py_BuildValue("");
    }
    ptr = (void *)PyArray_DATA(output);
    mm_compute_area(self, ptr);

    return (PyObject *)output;
}

PyObject *CMinMaxTreePy_tree_filter(CMinMaxTree *self, PyObject *args, PyObject *kw)
{
    PyArrayObject *values_map, *output;
    double threshold;
    char *ptr = NULL;

    if (!PyArg_ParseTuple(args, "O&d", mm_input_data, &values_map, &threshold)) {
        return Py_BuildValue("");
    }

    output = (PyArrayObject *)PyArray_EMPTY(self->ndim, PyArray_DIMS(self->data), PyArray_TYPE(self->data), 0);
    if (!output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return Py_BuildValue("");
    }
    ptr = (void *)PyArray_DATA(output);

    mm_tree_filter_from_values_map(self, values_map, threshold, ptr);
    return (PyObject *)output;
}

PyObject *CMinMaxTreePy_label_image(CMinMaxTree *self)
{
    PyArrayObject *output;
    char *ptr = NULL;
    int n_labels;

    output = (PyArrayObject *)PyArray_EMPTY(self->ndim, PyArray_DIMS(self->data), MM_CON_DTYPE, 0);
    if (!output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return Py_BuildValue("");
    }
    ptr = (void *)PyArray_DATA(output);
    n_labels = mm_label_image(self, ptr);

    return Py_BuildValue("(O,i)", (PyObject *)output, n_labels);
}

// #####################################################################################################################
















