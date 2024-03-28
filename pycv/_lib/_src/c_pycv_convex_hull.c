#include "c_pycv_base.h"
#include "c_pycv_convex_hull.h"


// #####################################################################################################################

#define chull_int_sign(_v) (_v < 0 ? -1 : _v) > 0 ? 1 : _v

// #####################################################################################################################

#define chull_distance_1d(_p1, _p2) _p1 < _p2 ? _p2 - _p1 : _p1 - _p2


static int chull_distance_nd(char *p1, char *p2, int ndim)
{
    int dist = 0;
    char *pp1 = p1, *pp2 = p2;
    while (ndim--) {
       dist += chull_distance_1d((int)(*(ch_dtype *)pp1), (int)(*(ch_dtype *)pp2));
       pp1 += ch_dtype_stride;
       pp2 += ch_dtype_stride;
    }
    return dist;
}

static void chull_to_vector(char *p1, char *p2, int ndim, int *vector)
{
    char *pp1 = p1, *pp2 = p2;
    int *vec = vector;

    while (ndim--) {
       *vec++ = (int)(*(ch_dtype *)pp2) - (int)(*(ch_dtype *)pp1);
       pp1 += ch_dtype_stride;
       pp2 += ch_dtype_stride;
    }
}

static int chull_cross_product(int *v1, int *v2, int ndim)
{
    return v1[1] * v2[0] - v1[0] * v2[1];
}

static int chull_direction(int *v1, int *v2, int ndim)
{
    return chull_int_sign(chull_cross_product(v1, v2, ndim));
}

static int chull_tiebreaker(int *v1, int *v2, int ndim)
{
    int ii, dist1 = 0, dist2 = 0;
    for (ii = 0; ii < ndim; ii++) {
        dist1 += *(v1 + ii) < 0 ? -*(v1 + ii) : *(v1 + ii);
        dist2 += *(v2 + ii) < 0 ? -*(v2 + ii) : *(v2 + ii);
    }
    return dist1 < dist2 ? -1 : 1;
}

// #####################################################################################################################

static int chull_find_leftmost(char *points, int n, int ndim)
{
    int out = 0, change = 0, ii, jj;
    char *ptr = points + ch_dtype_stride, *l_ptr = points, *l_tmp;
    for (ii = 1; ii < n; ii++) {
        if (*((ch_dtype *)ptr) < *((ch_dtype *)l_ptr)) {
            out = ii;
            l_ptr = ptr;
            ptr += ndim * ch_dtype_stride;
        } else if (*((ch_dtype *)ptr) == *((ch_dtype *)l_ptr)) {
            change = 1;
            l_tmp = l_ptr + ch_dtype_stride;
            ptr += ch_dtype_stride;
            for (jj = 1; jj < ndim; jj++) {
                if (change && (*((ch_dtype *)ptr) > *((ch_dtype *)l_tmp))) {
                    change = 0;
                }
                l_tmp += ch_dtype_stride;
                ptr += ch_dtype_stride;
            }
            if (change) {
                out = ii;
                l_ptr = ptr;
            }
        }
    }
    return out;
}

// #####################################################################################################################

static void chull_swap(int *p1, int *p2)
{
    int tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}

static void chull_heapsort_heapify(char *p0, char *points, int *indices, int pos, int size, int ndim, int *v1, int *v2)
{
    int end_pos = pos, l = 2 * pos + 1, r = 2 * pos + 2;

    if (l < size) {
        chull_to_vector(p0, points + (*(indices + end_pos) * ndim) * ch_dtype_stride, ndim, v1);
        chull_to_vector(p0, points + (*(indices + l) * ndim) * ch_dtype_stride, ndim, v2);
        int _d1 = chull_direction(v1, v2, ndim);
        if (!_d1) {
            _d1 = chull_tiebreaker(v1, v2, ndim);
        }
        if (_d1 < 0) {
            end_pos = l;
        }
    }

    if (r < size) {
        chull_to_vector(p0, points + (*(indices + end_pos) * ndim) * ch_dtype_stride, ndim, v1);
        chull_to_vector(p0, points + (*(indices + r) * ndim) * ch_dtype_stride, ndim, v2);
        int _d2 = chull_direction(v1, v2, ndim);
        if (!_d2) {
            _d2 = chull_tiebreaker(v1, v2, ndim);
        }
        if (_d2 < 0) {
            end_pos = r;
        }
    }

    if (end_pos != pos) {
        chull_swap(indices + pos, indices + end_pos);
        chull_heapsort_heapify(p0, points, indices, end_pos, size, ndim, v1, v2);
    }
}

static void chull_heapsort(char *p0, char *points, int *indices, int size, int ndim, int *v1, int *v2)
{
    int ii;

    for (ii = size / 2 - 1; ii >= 0; ii--) {
        chull_heapsort_heapify(p0, points, indices, ii, size, ndim, v1, v2);
    }

    for (ii = size - 1; ii >= 0; ii--) {
        chull_swap(indices, indices + ii);
        chull_heapsort_heapify(p0, points, indices, 0, ii, ndim, v1, v2);
    }
}

// #####################################################################################################################

static int chull_graham_scan_make(char *points, int n, int ndim, int **output, int *chull_size)
{
    char *p0, *p1, *p2;
    int *indices, *ind, *v1, *v2, *vec, ii, leftmost, der, candidate_size, convex_size;

    *output = malloc(n * sizeof(int));
    *chull_size = 0;
    vec = malloc(ndim * 2 * sizeof(int));

    if (!*output || !vec) {
        PyErr_NoMemory();
        return 0;
    }

    indices = *output;

    v1 = vec;
    v2 = v1 + ndim;

    ind = indices;
    for (ii = 0; ii < n; ii++) {
        *ind++ = ii;
    }

    leftmost = chull_find_leftmost(points, n, ndim);
    if (leftmost) {
        chull_swap(indices, indices + leftmost);
    }

    p0 = points + (leftmost * ndim) * ch_dtype_stride;

    if (n > 2) {
        chull_heapsort(p0, points, (indices + 1), n - 1, ndim, v1, v2);
    }

    ind = indices + 1;
    candidate_size = 1;

    for (ii = 1; ii < n - 1; ii++) {
        p1 = points + (*(indices + ii) * ndim) * ch_dtype_stride;
        p2 = points + (*(indices + ii + 1) * ndim) * ch_dtype_stride;

        chull_to_vector(p0, p1, ndim, v1);
        chull_to_vector(p0, p2, ndim, v2);

        der = chull_direction(v1, v2, ndim);
        if (der) {
            *ind++ = *(indices + ii);
            candidate_size++;
        }
    }
    *ind++ = *(indices + n - 1);
    candidate_size++;

    convex_size = 3;

    for (ii = 3; ii < candidate_size; ii++) {
        while (convex_size > 2) {
            p0 = points + (*(indices + convex_size - 2) * ndim) * ch_dtype_stride;
            p1 = points + (*(indices + convex_size - 1) * ndim) * ch_dtype_stride;
            p2 = points + (*(indices + ii) * ndim) * ch_dtype_stride;

            chull_to_vector(p0, p1, ndim, v1);
            chull_to_vector(p0, p2, ndim, v2);

            der = chull_direction(v1, v2, ndim);

            if (der < 0) {
                break;
            } else {
                convex_size--;
            }
        }
        *(indices + convex_size) = *(indices + ii);
        convex_size++;
    }

    *chull_size = convex_size;
    free(vec);
    return 1;
}

static int chull_jarvis_march_make(char *points, int n, int ndim, int **output, int *chull_size)
{
    char *p0, *p1, *p2;
    int *indices, *v1, *v2, *vec, ii, leftmost, left, convex_size, candidate;

    *output = malloc(n * sizeof(int));
    vec = malloc(ndim * 2 * sizeof(int));

    if (!*output || !vec) {
        PyErr_NoMemory();
        return 0;
    }

    indices = *output;

    v1 = vec;
    v2 = v1 + ndim;
    leftmost = chull_find_leftmost(points, n, ndim);
    *indices++ = leftmost;

    left = leftmost;
    convex_size = 1;

    while (1) {
        candidate = (left + 1) % n;
        for (ii = 0; ii < n; ii++) {
            if (ii == left) {
                continue;
            }
            p0 = points + (left * ndim) * ch_dtype_stride;
            p1 = points + (ii * ndim) * ch_dtype_stride;
            p2 = points + (candidate * ndim) * ch_dtype_stride;

            chull_to_vector(p0, p1, ndim, v1);
            chull_to_vector(p0, p2, ndim, v2);
            int _d1 = chull_direction(v1, v2, ndim);
            if (!_d1) {
                _d1 = chull_tiebreaker(v1, v2, ndim);
            }
            if (_d1 > 0) {
                candidate = ii;
            }
        }
        left = candidate;
        if (leftmost == left) {
            break;
        }
        *indices++ = left;
        convex_size++;
    }

    *chull_size = convex_size;
    free(vec);
    return 1;
}

// #####################################################################################################################

static int chull_query_point(char *points, char *vertices, int n_vertices, int ndim, int *point)
{
    int ii, jj, i0, i1, pii0, pii1, pjj0, pjj1, a, b, c, d, sign = 0, sign_d = 0;
    char *p0, *p1;

    for (ii = 0; ii < n_vertices; ii++) {
        jj = (ii + 1) % n_vertices;

        i0 = (int)(*((ch_dtype *)(vertices + ii * ch_dtype_stride)));
        i1 = (int)(*((ch_dtype *)(vertices + jj * ch_dtype_stride)));

        p0 = points + (i0 * ndim) * ch_dtype_stride;
        p1 = points + (i1 * ndim) * ch_dtype_stride;

        pii0 = (int)(*((ch_dtype *)p0));
        pii1 = (int)(*((ch_dtype *)(p0 + ch_dtype_stride)));

        pjj0 = (int)(*((ch_dtype *)p1));
        pjj1 = (int)(*((ch_dtype *)(p1 + ch_dtype_stride)));

        a = pii0 - pjj0;
        b = pjj1 - pii1;
        c = -a * pii1 - b * pii0;

        d = a * *(point + 1) + b * *point + c;

        if (d) {
            sign_d = d < 0 ? -1 : 1;
        } else {
            sign_d = sign;
        }

        if (!sign && sign_d) {
            sign = sign_d;
        } else if (sign_d != sign) {
            return 0;
        }
    }
    return 1;
}

static int chull_to_bw_image(CConvexHull *self, PyArrayObject *output)
{
    int array_size, ii, num_type_o, stride, jj, *coordinates, *dims_m1;
    char *po = NULL, *points = (void *)PyArray_DATA(self->points), *vertices = (void *)PyArray_DATA(self->vertices);

    num_type_o = (int)PyArray_TYPE(output);
    array_size = (int)PyArray_SIZE(output);
    stride = (int)PyArray_ITEMSIZE(output);

    coordinates = calloc(self->ndim * 2, sizeof(int));
    if (!coordinates) {
        PyErr_NoMemory();
        return 0;
    }
    dims_m1 = coordinates + self->ndim;
    for (ii = 0; ii < self->ndim; ii++) {
        *(dims_m1 + ii) = (int)PyArray_DIM(output, ii) - 1;
    }

    po = (void *)PyArray_DATA(output);

    for (ii = 0; ii < array_size; ii++) {
        if (chull_query_point(points, vertices, self->n_vertices, self->ndim, coordinates)) {
            PYCV_SET_VALUE(num_type_o, po, 1);
        } else {
            PYCV_SET_VALUE(num_type_o, po, 0);
        }
        po += stride;
        for (jj = self->ndim - 1; jj >= 0; jj--) {
            if (*(coordinates + jj) < *(dims_m1 + jj)) {
                *(coordinates + jj) += 1;
                break;
            } else {
                *(coordinates + jj) = 0;
            }
        }
    }
    return 1;
}

// #####################################################################################################################

void CConvexHullPy_dealloc(CConvexHull *self)
{
    Py_XDECREF(self->points);
    Py_XDECREF(self->vertices);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *CConvexHullPy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CConvexHull *self;
    self = (CConvexHull *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->ndim = 0;
        self->n_vertices = 0;
        self->points = NULL;
        self->vertices = NULL;
    }

    return (PyObject *)self;
}

static int chull_input_data(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    if (*output && PyArray_TYPE(*output) != NPY_INT64) {
        PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be NPY_INT64 \n");
       *output = NULL;
    }

    *output = (PyArrayObject *)PyArray_GETCONTIGUOUS(*output);
    return *output != NULL;
}

static int chull_input_to_array(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

int CConvexHullPy_init(CConvexHull *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"", "method", NULL};
    int *indices, chull_size = 0, n, method = 1, ii;
    npy_intp convex_dims[1] = {0};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O&|i", kwlist, chull_input_data, &(self->points), &method)) {
        return -1;
    }

    n = (int)PyArray_DIM(self->points, 0);
    self->ndim = (int)PyArray_DIM(self->points, 1);

    if (self->ndim != 2) {
        PyErr_SetString(PyExc_RuntimeError, "currently convex hull supported just for 2d");
        return -1;
    }

    char *points = (void *)PyArray_DATA(self->points);

    switch ((CHull_Method)method) {
        case CHULL_GRAHAM_SCAN:
            if (!chull_graham_scan_make(points, n, self->ndim, &indices, &chull_size)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: chull_graham_scan_make");
                return -1;
            }
            break;
        case CHULL_JARVIS_MARCH:
            if (!chull_jarvis_march_make(points, n, self->ndim, &indices, &chull_size)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: chull_jarvis_march_make");
                return -1;
            }
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "unsupported method");
            return -1;
    }

    convex_dims[0] = chull_size;
    self->vertices = (PyArrayObject *)PyArray_EMPTY(1, convex_dims, ch_con_dtype, 0);

    if (!self->vertices) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        free(indices);
        return -1;
    }

    char *ptr = (void *)PyArray_DATA(self->vertices);
    self->n_vertices = chull_size;
    for (ii = 0; ii < chull_size; ii++) {
        *(ch_dtype *)ptr = (ch_dtype)(*(indices + ii));
        ptr += ch_dtype_stride;
    }
    free(indices);
    return 0;
}

PyObject *CConvexHullPy_convex_to_image(CConvexHull *self, PyObject *args)
{
    PyArray_Dims shape = {NULL, 0};
    PyArrayObject *output;

    if (!PyArg_ParseTuple(args, "O&", PyArray_IntpConverter, &shape)) {
        goto exit;
    }

    if (shape.len != self->ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: shape size need to be equal to convex ndim \n");
        goto exit;
    }

    output = (PyArrayObject *)PyArray_EMPTY(self->ndim, shape.ptr, NPY_BOOL, 0);

    if (!output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    if (!chull_to_bw_image(self, output)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: chull_to_bw_image \n");
        goto exit;
    }

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        PyDimMem_FREE(shape.ptr);
        return PyErr_Occurred() ? Py_BuildValue("") : (PyObject *)output;
}

PyObject *CConvexHullPy_query_point(CConvexHull *self, PyObject *args)
{
    PyArrayObject *query_points;
    PyObject *output;
    int n_points, ii, jj, *point, num_type, pp, out;
    char *dptr = NULL, *points = (void *)PyArray_DATA(self->points), *vertices = (void *)PyArray_DATA(self->vertices);
    PYCV_ArrayIterator iter;

    if (!PyArg_ParseTuple(args, "O&", chull_input_to_array, &query_points)) {
        goto exit;
    }

    if ((int)PyArray_DIM(query_points, 1) != self->ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: points ndim need to be equal to convex ndim \n");
        goto exit;
    }

    n_points = (int)PyArray_DIM(query_points, 0);
    output = PyList_New((Py_ssize_t)n_points);

    point = malloc(self->ndim * sizeof(int));

    if (!point) {
        PyErr_NoMemory();
        goto exit;
    }

    PYCV_ArrayIteratorInit(query_points, &iter);
    num_type = (int)PyArray_TYPE(query_points);
    dptr = (void *)PyArray_DATA(query_points);

    for (ii = 0; ii < n_points; ii++) {
        for (jj = 0; jj < self->ndim; jj++) {
            PYCV_GET_VALUE(num_type, int, dptr, pp);
            *(point + jj) = pp;
            PYCV_ARRAY_ITERATOR_NEXT(iter, dptr);
        }
        out = chull_query_point(points, vertices, self->n_vertices, self->ndim, point);
        if (PyList_SetItem(output, (Py_ssize_t)ii, Py_BuildValue("i", out))) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyList_SetItem \n");
            goto exit;
        }
    }

    exit:
        Py_XDECREF(points);
        if (point) {
            free(point);
        }
        return PyErr_Occurred() ? Py_BuildValue("") : output;
}

// #####################################################################################################################

