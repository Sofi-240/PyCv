#include "c_pycv_base.h"
#include "c_pycv_haar_like.h"
#include <string.h>

// #####################################################################################################################

static int dims_convert_to_int(PyObject *obj, int **output)
{
    PyArray_Dims dims = {NULL, 0};
    if (!PyArray_IntpConverter(obj, &dims)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_IntpConverter");
        return 0;
    }
    *output = malloc(dims.len * sizeof(int));
    if (!*output) {
        PyErr_NoMemory();
        return 0;
    }
    for (int ii = 0; ii < dims.len; ii++) {
        *(*output + ii) = (int)(*(dims.ptr + ii));
    }
    PyDimMem_FREE(dims.ptr);
    return 1;
}

static int convert_input_array(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    if (*output != NULL) {
        *output = (PyArrayObject *)PyArray_GETCONTIGUOUS(*output);
    }
    return *output != NULL;
}

// *********************************************************************************************************************

#define HAAR_MAX_NDIM 3

static int haar_feature_n_rectangles(HaarType htype)
{
    switch (htype) {
        case HAAR_EDGE:
            return 2;
        case HAAR_LINE:
            return 3;
        case HAAR_DIAG:
            return 4;
        default:
            return 0;
    }
}

static void haar_feature_get_h_axis(HaarType htype, int ndim, int *h_axis, int *axis)
{
    switch (htype) {
        case HAAR_EDGE:
            for (int ii = 0; ii < ndim; ii++) {
                if (*axis == ii) {
                    *(h_axis + ii) = 2;
                } else {
                    *(h_axis + ii) = 1;
                }
            }
            break;
        case HAAR_LINE:
            for (int ii = 0; ii < ndim; ii++) {
                if (*axis == ii) {
                    *(h_axis + ii) = 3;
                } else {
                    *(h_axis + ii) = 1;
                }
            }
            break;
        case HAAR_DIAG:
            for (int ii = 0; ii < ndim; ii++) {
                if (*axis == ii || *(axis + 1) == ii) {
                    *(h_axis + ii) = 2;
                } else {
                    *(h_axis + ii) = 1;
                }
            }
            break;
    }
}

static void haar_feature_htype_to_srt(HaarType htype, int ndim, int *h_axis, PyObject **output)
{
    char as_str[15] = "type-";
    ndim = ndim > 3 ? 3 : ndim;

    switch (htype) {
        case HAAR_EDGE:
            strcat(as_str, "edge-");
            break;
        case HAAR_LINE:
            strcat(as_str, "line-");
            break;
        case HAAR_DIAG:
            strcat(as_str, "diag-");
            break;
    }

    for (int ii = 0; ii < ndim; ii++) {
        if (*(h_axis + ii) > 1) {
            switch (ii) {
                case 0:
                    strcat(as_str, "y");
                    break;
                case 1:
                    strcat(as_str, "x");
                    break;
                case 2:
                    strcat(as_str, "z");
                    break;
            }
        }
    }
    *output = Py_BuildValue("s", as_str);
}

static int haar_feature_n_features(int ndim, int *h_axis, int *feature_dims, int *dims, int *top_left)
{
    int m, mii, tl[HAAR_MAX_NDIM], delta[HAAR_MAX_NDIM];
    int ii, flag = 0, n_features = 0;

    for (ii = 0; ii < ndim; ii++) {
        *(delta + ii) = 1;
        *(tl + ii) = top_left ? *(top_left + ii) : 0;
    }

    while (!flag) {
        m = 1;
        for (ii = 0; ii < ndim; ii++) {
            mii = *(dims + ii) - *(tl + ii) - (*(h_axis + ii) * *(delta + ii)) + 1;
            if (mii < 0) {
                m = 0;
                break;
            } else {
                m *= mii;
            }
        }

        n_features += m;

        for (ii = ndim - 1; ii >= 0; ii--) {
            if (*(delta + ii) < *(feature_dims + ii)) {
                *(delta + ii) += 1;
                break;
            } else {
                *(delta + ii) = 1;
                if (!ii) {
                    flag = 1;
                }
            }
        }
    }

    return n_features;
}

// #####################################################################################################################

typedef struct {
    int *tl;
    int *br;
} Rectangle;

// *********************************************************************************************************************

static int haar_feature_init_rectangles(CHaarFeatures *self, int **points, Rectangle **rects)
{
    *points = calloc(self->ndim * self->n_rect * 2, sizeof(int));
    *rects = malloc(self->n_rect * sizeof(Rectangle));
    if (!*points || !*rects) {
        PyErr_NoMemory();
        return 0;
    }
    for (int ii = 0; ii < self->n_rect; ii++) {
        (*rects + ii)->tl = *points + ii * 2 * self->ndim;
        (*rects + ii)->br = *points + ii * 2 * self->ndim + self->ndim;
    }
    return 1;
}


#define CASE_HAAR_FEATURES_INTEGRATE(_NTYPE, _dtype, _n_rect, _rects, _st, _ndim, _ptr, _out)                          \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    for (int _ii = 0; _ii < _n_rect; _ii++) {                                                                          \
        int *_tl = (_rects + _ii)->tl, *_br = (_rects + _ii)->br;                                                      \
        *(_out + _ii) = 0;                                                                                             \
        switch (_ndim) {                                                                                               \
            case 2:                                                                                                    \
                *(_out + _ii) += (double)(*(_dtype *)(_ptr + *_br * *_st + *(_br + 1) * *(_st + 1)));                  \
                if (*_tl - 1 >= 0 && *(_tl + 1) - 1 >= 0) {                                                            \
                    *(_out + _ii) += (double)(*(_dtype *)(_ptr + (*_tl - 1) * *_st + (*(_tl + 1) - 1) * *(_st + 1)));  \
                }                                                                                                      \
                if (*_tl - 1 >= 0) {                                                                                   \
                    *(_out + _ii) -= (double)(*(_dtype *)(_ptr + (*_tl - 1) * *_st + *(_br + 1) * *(_st + 1)));        \
                }                                                                                                      \
                if (*(_tl + 1) - 1 >= 0) {                                                                             \
                    *(_out + _ii) -= (double)(*(_dtype *)(_ptr + *_br * *_st + (*(_tl + 1) - 1) * *(_st + 1)));        \
                }                                                                                                      \
                break;                                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break;

#define HAAR_FEATURES_INTEGRATE(_NTYPE, _n_rect, _rects, _st, _ndim, _ptr, _out)                                       \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_HAAR_FEATURES_INTEGRATE(BOOL, npy_bool, _n_rect, _rects, _st, _ndim, _ptr, _out);                         \
        CASE_HAAR_FEATURES_INTEGRATE(UBYTE, npy_ubyte, _n_rect, _rects, _st, _ndim, _ptr, _out);                       \
        CASE_HAAR_FEATURES_INTEGRATE(USHORT, npy_ushort, _n_rect, _rects, _st, _ndim, _ptr, _out);                     \
        CASE_HAAR_FEATURES_INTEGRATE(UINT, npy_uint, _n_rect, _rects, _st, _ndim, _ptr, _out);                         \
        CASE_HAAR_FEATURES_INTEGRATE(ULONG, npy_ulong, _n_rect, _rects, _st, _ndim, _ptr, _out);                       \
        CASE_HAAR_FEATURES_INTEGRATE(ULONGLONG, npy_ulonglong, _n_rect, _rects, _st, _ndim, _ptr, _out);               \
        CASE_HAAR_FEATURES_INTEGRATE(BYTE, npy_byte, _n_rect, _rects, _st, _ndim, _ptr, _out);                         \
        CASE_HAAR_FEATURES_INTEGRATE(SHORT, npy_short, _n_rect, _rects, _st, _ndim, _ptr, _out);                       \
        CASE_HAAR_FEATURES_INTEGRATE(INT, npy_int, _n_rect, _rects, _st, _ndim, _ptr, _out);                           \
        CASE_HAAR_FEATURES_INTEGRATE(LONG, npy_long, _n_rect, _rects, _st, _ndim, _ptr, _out);                         \
        CASE_HAAR_FEATURES_INTEGRATE(LONGLONG, npy_longlong, _n_rect, _rects, _st, _ndim, _ptr, _out);                 \
        CASE_HAAR_FEATURES_INTEGRATE(FLOAT, npy_float, _n_rect, _rects, _st, _ndim, _ptr, _out);                       \
        CASE_HAAR_FEATURES_INTEGRATE(DOUBLE, npy_double, _n_rect, _rects, _st, _ndim, _ptr, _out);                     \
    }                                                                                                                  \
}

// *********************************************************************************************************************

static void haar_feature_extend(Rectangle *rects, HaarType htype, int n_rect, int dim, int delta, int h_axis)
{
    const int case_diag_2d[8] = {0, 0, 0, 1, 1, 1, 1, 0};
    int *tl = rects->tl;

    for (int ii = 0; ii < n_rect; ii++) {
        switch (htype) {
            case HAAR_EDGE:
            case HAAR_LINE:
            case HAAR_DIAG:
                if (h_axis > 1) {
                    int c = htype == HAAR_DIAG ? *(case_diag_2d + ii * 2 + dim) : ii;
                    *((rects + ii)->tl + dim) = *(tl + dim) + c * delta;
                    *((rects + ii)->br + dim) = *(tl + dim) + (c + 1) * delta - 1;
                } else {
                    *((rects + ii)->tl + dim) = *(tl + dim);
                    *((rects + ii)->br + dim) = *(tl + dim) + delta - 1;
                }
                break;
        }
    }
}

// *********************************************************************************************************************

static int haar_feature_coordinates(CHaarFeatures *self, int *inp_dims, PyArrayObject **output)
{
    int *h_axis, *f_dims, *points, delta[HAAR_MAX_NDIM];
    npy_intp coord_dims[4] = {0, 0, 0, 0};
    Rectangle *rects;
    int n_features, ndim, ii, stride, itemsize, has_next = 1, delta_flag = 0, n, valid;
    char *ptr = NULL;

    ndim = self->ndim;
    itemsize = (int)NPY_SIZEOF_LONGLONG;

    if (!dims_convert_to_int(self->feature_dims, &f_dims) || !dims_convert_to_int(self->axis, &h_axis)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: dims_convert_to_int");
        return 0;
    }
    if (!haar_feature_init_rectangles(self, &points, &rects)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: haar_feature_init_rectangles");
        free(f_dims);
        free(h_axis);
        return 0;
    }

    if (!inp_dims) {
        inp_dims = f_dims;
    }

    n_features = haar_feature_n_features(ndim, h_axis, f_dims, inp_dims, NULL);

    coord_dims[0] = (npy_intp)n_features;
    coord_dims[1] = (npy_intp)(self->n_rect);
    coord_dims[2] = 2;
    coord_dims[3] = (npy_intp)ndim;

    *output = (PyArrayObject *)PyArray_EMPTY(4, coord_dims, NPY_INT64, 0);
    if (!*output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        goto exit;
    }
    if (!n_features) {
        goto exit;
    }

    stride = ndim * itemsize;
    ptr = (void *)PyArray_DATA(*output);
    n = self->n_rect * self->ndim * 2;

    for (ii = 0; ii < ndim; ii++) {
        *(delta + ii) = 1;
        haar_feature_extend(rects, self->_htype, self->n_rect, ii, 1, *(h_axis + ii));
    }

    while (has_next) {

        if (!delta_flag) {
            for (ii = 0; ii < n; ii++) {
                *(npy_longlong *)ptr = (npy_longlong)(*(points + ii));
                ptr += itemsize;
            }
            for (ii = ndim - 1; ii >= 0; ii--) {
                if ((*(delta + ii) < *(f_dims + ii)) &&
                    (*(rects->tl + ii) + (*(h_axis + ii) * (*(delta + ii) + 1)) <= *(inp_dims + ii))) {
                    *(delta + ii) += 1;
                    haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                    break;
                } else {
                    *(delta + ii) = 1;
                    haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                    if (!ii) {
                        delta_flag = 1;
                    }
                }
            }
        }

        if (!delta_flag) {
            continue;
        }

        for (ii = ndim - 1; ii >= 0; ii--) {
            if (*(rects->tl + ii) < *(inp_dims + ii) - 1) {
                *(rects->tl + ii) += 1;
                haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                break;
            } else {
                *(rects->tl + ii) = 0;
                haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                if (!ii) {
                    has_next = 0;
                    break;
                }
            }
        }

        valid = 1;
        delta_flag = 0;
        for (ii = 0; ii < ndim; ii++) {
            if ((*(rects->tl + ii) + (*(h_axis + ii) * *(delta + ii)) > *(inp_dims + ii))) {
                valid = 0;
                delta_flag = 1;
                break;
            }
        }
    }

    exit:
        free(f_dims);
        free(h_axis);
        free(points);
        free(rects);
        return PyErr_Occurred() ? 0 : 1;
}

static int haar_feature_like(CHaarFeatures *self, PyArrayObject *integral, int *top_left, PyArrayObject **output)
{
    int *h_axis, *f_dims, *points, out_numtype;
    int delta[HAAR_MAX_NDIM], inp_dims[HAAR_MAX_NDIM], strides[HAAR_MAX_NDIM];
    double integral_v[HAAR_MAX_NDIM], v;
    npy_intp features_dims[1] = {0};
    Rectangle *rects;
    int n_features, ndim, ii, itemsize, has_next = 1, delta_flag = 0, valid, numtype;
    char *ptr_o = NULL, *ptr_i = NULL;

    ndim = self->ndim;
    numtype = (int)PyArray_TYPE(integral);

    if (!dims_convert_to_int(self->feature_dims, &f_dims) || !dims_convert_to_int(self->axis, &h_axis)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: dims_convert_to_int");
        return 0;
    }
    if (!haar_feature_init_rectangles(self, &points, &rects)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: haar_feature_init_rectangles");
        free(f_dims);
        free(h_axis);
        return 0;
    }

    for (ii = 0; ii < ndim; ii++) {
        *(inp_dims + ii) = (int)PyArray_DIM(integral, ii);
        *(strides + ii) = (int)PyArray_STRIDE(integral, ii);
        *(integral_v + ii) = 0;
    }

    n_features = haar_feature_n_features(ndim, h_axis, f_dims, inp_dims, top_left);
    features_dims[0] = (npy_intp)n_features;

    switch (numtype) {
        case NPY_BOOL:
        case NPY_UBYTE:
        case NPY_USHORT:
        case NPY_UINT:
        case NPY_ULONG:
        case NPY_ULONGLONG:
        case NPY_BYTE:
        case NPY_SHORT:
        case NPY_INT:
        case NPY_LONG:
        case NPY_LONGLONG:
            out_numtype = NPY_LONGLONG;
            break;
        case NPY_FLOAT:
        case NPY_DOUBLE:
            out_numtype = NPY_DOUBLE;
            break;
        default:
            out_numtype = NPY_DOUBLE;
    }

    *output = (PyArrayObject *)PyArray_EMPTY(1, features_dims, out_numtype, 0);
    if (!*output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        goto exit;
    }
    if (!n_features) {
        goto exit;
    }

    itemsize = (int)PyArray_ITEMSIZE(*output);
    ptr_o = (void *)PyArray_DATA(*output);
    ptr_i = (void *)PyArray_DATA(integral);

    for (ii = 0; ii < ndim; ii++) {
        *(delta + ii) = 1;
        haar_feature_extend(rects, self->_htype, self->n_rect, ii, 1, *(h_axis + ii));
    }

    while (has_next) {

        if (!delta_flag) {
            v = 0;

            HAAR_FEATURES_INTEGRATE(numtype, (self->n_rect), rects, strides, ndim, ptr_i, integral_v);

            for (ii = 0; ii < self->n_rect; ii++) {
                if (ii % 2) {
                    v += *(integral_v + ii);
                } else {
                    v -= *(integral_v + ii);
                }
            }

            PYCV_SET_VALUE_F2A(out_numtype, ptr_o, v);
            ptr_o += itemsize;

            for (ii = ndim - 1; ii >= 0; ii--) {
                if ((*(delta + ii) < *(f_dims + ii)) &&
                    (*(rects->tl + ii) + (*(h_axis + ii) * (*(delta + ii) + 1)) <= *(inp_dims + ii))) {
                    *(delta + ii) += 1;
                    haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                    break;
                } else {
                    *(delta + ii) = 1;
                    haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                    if (!ii) {
                        delta_flag = 1;
                    }
                }
            }
        }

        if (!delta_flag) {
            continue;
        }

        for (ii = ndim - 1; ii >= 0; ii--) {
            if (*(rects->tl + ii) < *(inp_dims + ii) - 1) {
                *(rects->tl + ii) += 1;
                haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                break;
            } else {
                *(rects->tl + ii) = 0;
                haar_feature_extend(rects, self->_htype, self->n_rect, ii, *(delta + ii), *(h_axis + ii));
                if (!ii) {
                    has_next = 0;
                    break;
                }
            }
        }

        valid = 1;
        delta_flag = 0;
        for (ii = 0; ii < ndim; ii++) {
            if ((*(rects->tl + ii) + (*(h_axis + ii) * *(delta + ii)) > *(inp_dims + ii))) {
                valid = 0;
                delta_flag = 1;
                break;
            }
        }
    }

    exit:
        free(f_dims);
        free(h_axis);
        free(points);
        free(rects);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

void CHaarFeaturesPy_dealloc(CHaarFeatures *self)
{
    Py_TYPE(self->htype)->tp_free((PyObject *)(self->htype));
    Py_TYPE(self->axis)->tp_free((PyObject *)(self->axis));
    Py_TYPE(self->feature_dims)->tp_free((PyObject *)(self->feature_dims));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *CHaarFeaturesPy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CHaarFeatures *self;
    self = (CHaarFeatures *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->ndim = 0;
        self->n_rect = 0;
        self->_htype = 0;
        self->n_features = 0;

        self->htype = NULL;
        self->axis = NULL;
        self->feature_dims = NULL;

    }

    return (PyObject *)self;
}

int CHaarFeaturesPy_init(CHaarFeatures *self, PyObject *args)
{
    int *axis, *feature_dims, h_axis[HAAR_MAX_NDIM];

    if (!PyArg_ParseTuple(args, "iiO&O&", &(self->_htype), &(self->ndim),
                          dims_convert_to_int, &feature_dims, dims_convert_to_int, &axis)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid args");
        return -1;
    }

    self->n_rect = haar_feature_n_rectangles(self->_htype);

    haar_feature_get_h_axis(self->_htype, self->ndim, h_axis, axis);

    self->axis = PyTuple_New((Py_ssize_t)self->ndim);
    self->feature_dims = PyTuple_New((Py_ssize_t)self->ndim);

    for (int ii = 0; ii < self->ndim; ii++) {
        if (PyTuple_SetItem(self->axis, (Py_ssize_t)ii, Py_BuildValue("i", *(h_axis + ii))) ||
            PyTuple_SetItem(self->feature_dims, (Py_ssize_t)ii, Py_BuildValue("i", *(feature_dims + ii)))) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyTuple_SetItem args");
            free(feature_dims);
            free(axis);
            return -1;
        }
    }

    haar_feature_htype_to_srt(self->_htype, self->ndim, h_axis, &(self->htype));

    free(feature_dims);
    free(axis);

    if (!dims_convert_to_int(self->feature_dims, &feature_dims) ||
        !dims_convert_to_int(self->axis, &axis)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_IntpConverter");
        return -1;
    }

    self->n_features = haar_feature_n_features(self->ndim, axis, feature_dims, feature_dims, NULL);
    free(feature_dims);
    free(axis);
    return 0;
}

PyObject *CHaarFeaturesPy_coordinates(CHaarFeatures *self, PyObject *args, PyObject *kw)
{
    static char* kwlist[] = {"dims", NULL};
    int not_occurred = 0, *dims;
    PyArrayObject *output;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|O&", kwlist, dims_convert_to_int, &dims)) {
        PyErr_SetString(PyExc_RuntimeError, "invalid args or keywords");
        goto exit;
    }

    not_occurred = haar_feature_coordinates(self, dims, &output);

    exit:
        free(dims);
        return not_occurred ? (PyObject *)output : Py_BuildValue("");
}

PyObject *CHaarFeaturesPy_like_features(CHaarFeatures *self, PyObject *args, PyObject *kw)
{
    static char* kwlist[] = {"", "top_left", NULL};
    int not_occurred = 0, *top_left = NULL;
    PyArrayObject *output, *integral;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O&|O&", kwlist,
                                     convert_input_array, &integral,
                                     dims_convert_to_int, &top_left)) {
        PyErr_SetString(PyExc_RuntimeError, "invalid args or keywords");
        goto exit;
    }

    not_occurred = haar_feature_like(self, integral, top_left, &output);

    exit:
        if (top_left) {
            free(top_left);
        }
        return not_occurred ? (PyObject *)output : Py_BuildValue("");
}

// #####################################################################################################################

















