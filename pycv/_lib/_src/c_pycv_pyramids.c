#include "c_pycv_base.h"
#include "c_pycv_pyramids.h"

// ################################################ IO #################################################################
// *********************************************************************************************************************

#define DEFAULT_TYPE NPY_DOUBLE

#define DEFAULT_ITEMSIZE NPY_SIZEOF_DOUBLE

static int IO_dims_from_array(PyArrayObject *array, int **dims)
{
    int ndim = PyArray_NDIM(array);
    UTILS_MALLOC(ndim, sizeof(int), *dims);
    if (!ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: UTILS_MALLOC");
        return 0;
    }
    for (int ii = 0; ii < ndim; ii++) {
        *(*dims + ii) = (int)PyArray_DIM(array, ii);
    }
    return 1;
}

// *********************************************************************************************************************

static int IO_parse_ndim(PyObject *object, Layer **layer)
{
    int ndim;
     if (!IO_object_parse(object, "i", ndim)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: IO_object_parse");
        return 0;
    }
    if (ndim <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: ndim must be positive integer");
        return 0;
    }
    if (!Layer_new(layer, ndim, DEFAULT_ITEMSIZE, DEFAULT_TYPE)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Layer_new");
        return 0;
    }
    return 1;
}

static int IO_parse_scalespace(PyObject *object, Layer **layer)
{
    double *scalespace = NULL, *scales = NULL;
    int size, nscales, ndim = (*layer)->ndim;

    if (!IO_object_check(List, object)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: scalespace must be list of tuples");
        return 0;
    }
    nscales = IO_object_size(List, object);
    if (!nscales) {
        PyErr_SetString(PyExc_RuntimeError, "Error: scalespace must be non empty list of tuples");
        return 0;
    }
    size = ndim * nscales;
    UTILS_MALLOC(size, sizeof(double), scalespace);
    if (!size) {
        PyErr_SetString(PyExc_RuntimeError, "Error: UTILS_MALLOC");
        return 0;
    }
    scales = scalespace;
    for (int ii = 0; ii < nscales; ii++) {
        PyObject *tup = IO_object_get_item(List, object, ii);
        if (tup == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Error: IO_object_get_item");
            free(scalespace);
            return 0;
        }
        if (IO_object_size(Tuple, tup) != ndim) {
            PyErr_SetString(PyExc_RuntimeError, "Error: scale size need to be equal to ndim");
            free(scalespace);
            return 0;
        }
        for (int jj = 0; jj < ndim; jj++) {
            if (!IO_object_parse(IO_object_get_item(Tuple, tup, jj), "d", *scales)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: IO_object_parse");
                free(scalespace);
                return 0;
            }
            scales += 1;
        }
    }

    int valid = Layer_set_scalespace(*layer, nscales, scalespace);
    free(scalespace);
    if (valid && (*layer)->iterator != NULL)
        valid = Layer_update_offsets(*layer);
    return valid;
}

static int IO_parse_factors(PyObject *object, Layer **layer)
{
    double *factors = NULL, *factor = NULL;
    int ndim = (*layer)->ndim;

    if (!IO_object_check(Tuple, object)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: factors must be tuple");
        return 0;
    }
    if (IO_object_size(Tuple, object) != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: factors size need to be equal to ndim");
        return 0;
    }

    UTILS_MALLOC(ndim, sizeof(double), factors);
    if (!ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: UTILS_MALLOC");
        return 0;
    }
    factor = factors;
    for (int jj = 0; jj < ndim; jj++) {
        if (!IO_object_parse(IO_object_get_item(Tuple, object, jj), "d", *factor)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: IO_object_parse");
            free(factors);
            return 0;
        }
        factor += 1;
    }
    int valid = Layer_set_factors(*layer, factors);
    free(factors);
    if (valid && (*layer)->iterator != NULL)
        valid = Layer_update_offsets(*layer);
    return valid;
}

static int IO_parse_input_dims(PyObject *object, Layer **layer)
{
    int *dims = NULL, *dim = NULL;
    int ndim = (*layer)->ndim;

    if (!IO_object_check(Tuple, object)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: dims must be tuple");
        return 0;
    }
    if (IO_object_size(Tuple, object) != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: dims size need to be equal to ndim");
        return 0;
    }

    UTILS_MALLOC(ndim, sizeof(int), dims);
    if (!ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: UTILS_MALLOC");
        return 0;
    }
    dim = dims;
    for (int jj = 0; jj < ndim; jj++) {
        if (!IO_object_parse(IO_object_get_item(Tuple, object, jj), "i", *dim)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: IO_object_parse");
            free(dims);
            return 0;
        }
        dim += 1;
    }
    int valid = Layer_set_input_dim(*layer, dims);
    free(dims);
    return valid;
}

static int IO_parse_order(PyObject *object, Layer **layer)
{
    int order;
    if (!IO_object_parse(object, "i", order))
        return 0;
    if (order < 0 || order >= 4)
        return 0;
    return Layer_set_order(*layer, order);
}

static int IO_parse_cval(PyObject *object, Layer **layer)
{
    int cval;
    if (!IO_object_parse(object, "d", cval))
        return 0;
    (*layer)->cval = cval;
    return 1;
}

static int IO_parse_extend_mode(PyObject *object, Layer **layer)
{
    int mode;
    if (!IO_object_parse(object, "i", mode))
        return 0;
    if (mode < 3 || mode > 7)
        return 0;
    Layer_set_extend_mode(*layer, mode);
    return 1;
}

static int IO_parse_descr(PyObject *object, Layer **layer)
{
    PyArray_Descr *descr = (PyArray_Descr *)object;
    Layer_set_itemsize(*layer, (int)descr->elsize, (int)descr->type_num);
    return 1;
}

static int IO_parse_array(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_C_CONTIGUOUS;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

// *********************************************************************************************************************

static void IO_get_scalespace(Layer *layer, PyObject **output)
{
    int ndim = layer->ndim;
    Gaussian1D *node = layer->scalespace;
    PyObject *scale = NULL;

    *output = IO_object_new(List, 0);

    while (node != NULL) {

        if (!node->axis)
            scale = IO_object_new(Tuple, ndim);

        IO_object_SET_ITEM(Tuple, scale, node->axis, IO_object_build("d", node->kernel->sigma));
        node = node->next;

        if (node == NULL || node->axis == 0) {
            PyList_Append(*output, scale);
            Py_XDECREF(scale);
        }
    }
}

static void IO_get_factors(Layer *layer, PyObject **output)
{
    int ndim = layer->ndim;
    Rescale1D *node = layer->factors;

    *output = IO_object_new(Tuple, ndim);
    for (int jj = 0; jj < ndim; jj++) {
        if (node == NULL) {
            IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("d", 1.0));
            continue;
        }
        IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("d", node->factor));
        node = node->next;
    }
}

static void IO_get_anti_alias_scales(Layer *layer, PyObject **output)
{
    int ndim = layer->ndim;
    Rescale1D *node = layer->factors;

    *output = IO_object_new(Tuple, ndim);
    for (int jj = 0; jj < ndim; jj++) {
        if (node == NULL || node->kernel == NULL) {
            IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("d", 0.0));
            if (node != NULL)
                node = node->next;
            continue;
        }
        IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("d", node->kernel->sigma));
        node = node->next;
    }
}

static void IO_get_input_dims(Layer *layer, PyObject **output)
{
    int ndim = layer->ndim;
    Iterator1D *node = layer->iterator;

    *output = IO_object_new(Tuple, ndim);
    for (int jj = 0; jj < ndim; jj++) {
        if (node == NULL) {
            IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("i", 0));
            continue;
        }
        IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("i", node->dim));
        node = node->next;
    }
}

static void IO_get_output_dims(Layer *layer, PyObject **output)
{
    int ndim = layer->ndim;
    Iterator1D *iter_node = layer->iterator;
    Rescale1D *factors = layer->factors;

    *output = IO_object_new(Tuple, ndim);
    for (int jj = 0; jj < ndim; jj++) {
        if (iter_node == NULL) {
            IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("i", 0));
            continue;
        } else if (factors == NULL) {
            IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("i", iter_node->dim));
            continue;
        }
        int out = (int)((double)iter_node->dim * factors->factor + 0.5);
        IO_object_SET_ITEM(Tuple, *output, jj, IO_object_build("i", out));
        iter_node = iter_node->next;
        factors = factors->next;
    }
}

// #####################################################################################################################

static int layer_iterator_adapt(Layer *self, PyArrayObject *input)
{
    int ndim = self->ndim, *dims = NULL;

    if (self->ndim != PyArray_NDIM(input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid input ndim");
        return 0;
    }

    if (self->iterator != NULL) {
        Iterator1D *node = self->iterator;
        while (node != NULL) {
            if (node->dim != (int)PyArray_DIM(input, node->axis))
                break;
            node = node->next;
        }
        if (node == NULL) {
            Layer_set_itemsize(self, (int)PyArray_ITEMSIZE(input), (int)PyArray_TYPE(input));
            if (self->iterator->offsets == NULL && !Layer_update_offsets(self)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: Layer_update_offsets");
                return 0;
            }
            return 1;
        }
    }
    if (!IO_dims_from_array(input, &dims)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: IO_dims_from_array");
        return 0;
    }

    self->itemsize = (int)PyArray_ITEMSIZE(input);
    self->numtype = (int)PyArray_TYPE(input);

    if (!Layer_set_input_dim(self, dims)) {
        free(dims);
        PyErr_SetString(PyExc_RuntimeError, "Error: Layer_set_input_dim");
        return 0;
    }

    free(dims);
    if (self->iterator->offsets == NULL && !Layer_update_offsets(self)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Layer_update_offsets");
        return 0;
    }
    return 1;
}

// *********************************************************************************************************************

#define case_type_copy(_ntype, _dtype, _from, _to, _n, _stride)                                                        \
case NPY_##_ntype:                                                                                                     \
{                                                                                                                      \
    char *_pi = _from, *_po = _to;                                                                                     \
    for (int _ii = 0; _ii < _n; _ii++) {                                                                               \
        *(_dtype *)_po = *(_dtype *)_pi;                                                                               \
        _po += _stride;                                                                                                \
        _pi += _stride;                                                                                                \
    }                                                                                                                  \
}                                                                                                                      \
break

#define type_array_copy(_ntype, _from, _to, _n, _stride)                                                               \
{                                                                                                                      \
    switch (_ntype) {                                                                                                  \
        case_type_copy(BOOL, npy_bool, _from, _to, _n, _stride);                                                       \
        case_type_copy(UBYTE, npy_ubyte, _from, _to, _n, _stride);                                                     \
        case_type_copy(USHORT, npy_ushort, _from, _to, _n, _stride);                                                   \
        case_type_copy(UINT, npy_uint, _from, _to, _n, _stride);                                                       \
        case_type_copy(ULONG, npy_ulong, _from, _to, _n, _stride);                                                     \
        case_type_copy(ULONGLONG, npy_ulonglong, _from, _to, _n, _stride);                                             \
        case_type_copy(BYTE, npy_byte, _from, _to, _n, _stride);                                                       \
        case_type_copy(SHORT, npy_short, _from, _to, _n, _stride);                                                     \
        case_type_copy(INT, npy_int, _from, _to, _n, _stride);                                                         \
        case_type_copy(LONG, npy_long, _from, _to, _n, _stride);                                                       \
        case_type_copy(LONGLONG, npy_longlong, _from, _to, _n, _stride);                                               \
        case_type_copy(FLOAT, npy_float, _from, _to, _n, _stride);                                                     \
        case_type_copy(DOUBLE, npy_double, _from, _to, _n, _stride);                                                   \
    }                                                                                                                  \
}

// ************* Pyramid convolve **************

#define case_type_convolve(_ntype, _dtype, _x, _h, _n, _ofs, _fl, _cv, _o)                                             \
case NPY_##_ntype:                                                                                                     \
{                                                                                                                      \
    _o = 0;                                                                                                            \
    for (int _ii = 0; _ii < _n; _ii++) {                                                                               \
        if (*(_ofs + _ii) == _fl)                                                                                      \
            _o += _cv * *(_h + _ii);                                                                                   \
        else                                                                                                           \
            _o += (double)(*(_dtype *)(_x + *(_ofs + _ii))) * *(_h + _ii);                                             \
    }                                                                                                                  \
}                                                                                                                      \
break

#define type_convolve(_ntype, _x, _h, _n, _ofs, _fl, _cv, _o)                                                          \
{                                                                                                                      \
    switch (_ntype) {                                                                                                  \
        case_type_convolve(BOOL, npy_bool, _x, _h, _n, _ofs, _fl, _cv, _o);                                            \
        case_type_convolve(UBYTE, npy_ubyte, _x, _h, _n, _ofs, _fl, _cv, _o);                                          \
        case_type_convolve(USHORT, npy_ushort, _x, _h, _n, _ofs, _fl, _cv, _o);                                        \
        case_type_convolve(UINT, npy_uint, _x, _h, _n, _ofs, _fl, _cv, _o);                                            \
        case_type_convolve(ULONG, npy_ulong, _x, _h, _n, _ofs, _fl, _cv, _o);                                          \
        case_type_convolve(ULONGLONG, npy_ulonglong, _x, _h, _n, _ofs, _fl, _cv, _o);                                  \
        case_type_convolve(BYTE, npy_byte, _x, _h, _n, _ofs, _fl, _cv, _o);                                            \
        case_type_convolve(SHORT, npy_short, _x, _h, _n, _ofs, _fl, _cv, _o);                                          \
        case_type_convolve(INT, npy_int, _x, _h, _n, _ofs, _fl, _cv, _o);                                              \
        case_type_convolve(LONG, npy_long, _x, _h, _n, _ofs, _fl, _cv, _o);                                            \
        case_type_convolve(LONGLONG, npy_longlong, _x, _h, _n, _ofs, _fl, _cv, _o);                                    \
        case_type_convolve(FLOAT, npy_float, _x, _h, _n, _ofs, _fl, _cv, _o);                                          \
        case_type_convolve(DOUBLE, npy_double, _x, _h, _n, _ofs, _fl, _cv, _o);                                        \
    }                                                                                                                  \
}

static void convolve1D(Iterator1D *iterator, int numtype, double *h, char *input, char *output, double cval)
{
    int *ofs = NULL, n, flag;
    char *inp = input, *out = output;
    double x;
    Iterator1D *iter = iterator;
    Offsets1D *offsets = NULL;

    UTILS_NODE_GOTO_LAST(iter);
    offsets = iter->offsets;

    ofs = offsets->offsets + offsets->init_pos;
    n = offsets->dim;
    flag = offsets->flag;

    do {
        type_convolve(numtype, inp, h, n, ofs, flag, cval, x);
        PYCV_SET_VALUE_F2A(numtype, out, x);

        iter = iterator;
        IteratorND_offsets_next2(iter, inp, out, ofs);

    } while (iter != NULL);
}

// ************* Pyramid scale *****************

static int scaleND(Iterator1D **iterator, Gaussian1D *scales, int numtype, char *input, char *output, char *mem,
                   int extend_mode, double cval)
{
    Iterator1D *iter_node = *iterator;
    int stride, stride_o = 0, nn = 0, size = 1, ndim = (*iterator)->axis + 1;
    char *ptro_base = NULL, *ptri = input, *ptro = NULL;
    Gaussian1D *scale = scales;

    while (scale != NULL && iter_node != NULL) {
        if (scale->kernel->sigma != 0)
            nn++;
        scale = scale->next;
        size *= iter_node->dim;
        stride = iter_node->dim * iter_node->stride;
        iter_node = iter_node->next;
    }

    if (!nn) {
        type_array_copy(numtype, input, output, size, (*iterator)->stride);
        return 1;
    }

    scale = scales;

    while (ndim--) {
        iter_node = *iterator;
        UTILS_NODE_GOTO_LAST(iter_node);

        GaussianMem *kernel = scale->kernel;

        if (kernel->sigma != 0.0) {
            nn--;
            if (!nn) {
                ptro = output;
            } else {
                ptro = mem + stride_o;
            }

            if (!Iterator1D_update_offset_dim(iter_node, kernel->len, extend_mode)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: Iterator1D_update_offset_dim");
                return 0;
            }

            convolve1D(*iterator, numtype, kernel->h, ptri, ptro, cval);

            ptri = ptro;
            stride_o = stride_o ? 0 : stride;
        }

        IteratorND_next_axis(iterator);
        scale = scale->next;
    }
    return 1;
}


static int layer_scale(Layer *self, PyArrayObject *input, PyArrayObject **output)
{
    npy_intp dims[NPY_MAXDIMS];
    Gaussian1D *scales = self->scalespace;
    PyArrayObject *extra_mem = NULL;
    int stride, size = 1;
    char *inp_ptr = NULL, *out_ptr = NULL, *ex_ptr = NULL;

    if (!layer_iterator_adapt(self, input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: layer_iterator_adapt");
        return 0;
    }

    Iterator1D *iter_node = self->iterator;
    *dims = (npy_intp)self->nscales;
    *dims += 1;

    while (iter_node != NULL) {
        *(dims + iter_node->axis + 1) = (npy_intp)(iter_node->dim);
        size *= iter_node->dim;
        iter_node = iter_node->next;
    }

    *output = (PyArrayObject *)PyArray_EMPTY(self->ndim + 1, dims, self->numtype, 0);
    if (*output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return 0;
    }

    stride = (int)PyArray_STRIDE(*output, 0);

    if (self->ndim > 1 && self->scalespace != NULL) {
        *dims = 2;
        extra_mem = (PyArrayObject *)PyArray_EMPTY(self->ndim + 1, dims, self->numtype, 0);
        if (extra_mem == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
            return 0;
        }
        ex_ptr = (void *)PyArray_DATA(extra_mem);
    }

    inp_ptr = (void *)PyArray_DATA(input);
    out_ptr = (void *)PyArray_DATA(*output);

    type_array_copy(self->numtype, inp_ptr, out_ptr, size, self->itemsize);
    inp_ptr = out_ptr;
    out_ptr += stride;

    while (scales != NULL) {
        if (!scaleND(&(self->iterator), scales, self->numtype, inp_ptr, out_ptr, ex_ptr, self->extend_mode, self->cval)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
            Py_XDECREF(extra_mem);
            return 0;
        }
        inp_ptr = out_ptr;
        out_ptr += stride;
        scales = scales->next;
        while (scales != NULL && scales->axis)
            scales = scales->next;
    }

    Py_XDECREF(extra_mem);
    return 1;
}

// ************* Pyramid rescale ****************

static void rescale1D(Iterator1D *iterator_i, Iterator1D *iterator_o, Offsets1D *offsets,
                              double factor, int numtype, char *input, char *output, double cval)
{
    int *ofs = offsets->offsets + offsets->init_pos, n = offsets->dim, flag = offsets->flag, stride = 0, order = offsets->dim - 1;
    char *inp = input, *out = output, *pi = NULL;
    double x, pos, nodes[4], delta = 0;
    Iterator1D *iter_i = NULL, *iter_o = NULL;

    do {
        pi = inp + stride;
        for (int ii = 0; ii < n; ii++) {
            if (*(ofs + ii) == flag) {
                *(nodes + ii) = cval;
            } else {
                PYCV_GET_VALUE(numtype, double, (pi + *(ofs + ii)), *(nodes + ii));
            }
        }
        x = pycv_interpolate(order, nodes, delta);
        PYCV_SET_VALUE_F2A(numtype, out, x);

        iter_i = iterator_i;
        iter_o = iterator_o;

        while (iter_o != NULL) {
            if (iter_o->coordinate < iter_o->dim - 1) {
                iter_o->coordinate += 1;
                out += iter_o->stride;
                if (iter_o->next != NULL) {
                    iter_i->coordinate += 1;
                    inp += iter_i->stride;
                } else {
                    pos = ((double)(iter_o->coordinate)) * factor;
                    if (order & 1) {
                        iter_i->coordinate = (int)floor(pos);
                        delta = pos - floor(pos);
                    } else {
                        iter_i->coordinate = (int)floor(pos + 0.5);
                        delta = pos - floor(pos + 0.5);
                    }
                    stride = iter_i->coordinate * iter_i->stride;
                    OFFSETS_GOTO(offsets, ofs, iter_i->coordinate);
                }
                break;
            } else {
                iter_o->coordinate = 0;
                out -= iter_o->stride_back;

                iter_i->coordinate = 0;
                inp -= iter_i->stride_back;

                iter_o = iter_o->next;
                iter_i = iter_i->next;
            }
        }

    } while (iter_o != NULL);
}

static int layer_rescale(Layer *self, PyArrayObject *input, PyArrayObject **output)
{
    npy_intp dims[NPY_MAXDIMS];
    Rescale1D *factor = self->factors;
    int ofs_dim, size = 1, mem_i = 0;
    double f;
    PyArrayObject *mem1 = NULL, *mem2 = NULL, *curr_output = NULL;
    char *inp_ptr = NULL, *out_ptr = NULL;
    RescaleIterator *iterator = NULL;
    Offsets1D *offsets = NULL;
    GaussianMem *kernel = NULL;
    Iterator1D *iter_node = NULL, *iter_i = NULL;

    if (!layer_iterator_adapt(self, input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: layer_iterator_adapt");
        return 0;
    }

    iter_node = self->iterator;

    while (iter_node != NULL) {
        *(dims + iter_node->axis) = (npy_intp)(iter_node->dim);
        size *= iter_node->dim;
        iter_node = iter_node->next;
    }

    inp_ptr = (void *)PyArray_DATA(input);

    if (factor == NULL) {
        *output = (PyArrayObject *)PyArray_EMPTY(self->ndim, dims, self->numtype, 0);
        if (*output == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
            return 0;
        }
        out_ptr = (void *)PyArray_DATA(*output);
        type_array_copy(self->numtype, inp_ptr, out_ptr, size, self->iterator->stride);
        return 1;
    }

    if (!RescaleIterator_new(&iterator, &(self->iterator), self->extend_mode)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: RescaleIterator_new");
        return 0;
    }

    ofs_dim = self->order == 0 ? 1 : 3 + 2 * ((self->order - 1) / 2);

    while (factor != NULL) {
        iter_node = self->iterator;
        UTILS_NODE_GOTO_LAST(iter_node);

        if (factor->kernel != NULL) {
            kernel = factor->kernel;

            if (!Iterator1D_update_offset_dim(iter_node, kernel->len, self->extend_mode)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: Iterator1D_update_offset_dim");
                RescaleIterator_free(&iterator);
                Py_XDECREF(mem1);
                Py_XDECREF(mem2);
                return 0;
            }

            iter_i = iterator->input;
            UTILS_NODE_GOTO_LAST(iter_i);
            iter_i->offsets = iter_node->offsets;

             if (!mem_i) {
                Py_XDECREF(mem1);
                mem1 = (PyArrayObject *)PyArray_EMPTY(self->ndim, dims, self->numtype, 0);
                curr_output = mem1;
                mem_i = 1;
            } else {
                Py_XDECREF(mem2);
                mem2 = (PyArrayObject *)PyArray_EMPTY(self->ndim, dims, self->numtype, 0);
                curr_output = mem2;
                mem_i = 0;
            }

            if (curr_output == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
                RescaleIterator_free(&iterator);
                Py_XDECREF(mem1);
                Py_XDECREF(mem2);
                return 0;
            }

            out_ptr = (void *)PyArray_DATA(curr_output);

            convolve1D(iterator->input, self->numtype, kernel->h, inp_ptr, out_ptr, self->cval);

            iter_i->offsets = NULL;
            inp_ptr = out_ptr;
        }

        if (!Iterator1D_update_offset_dim(iter_node, ofs_dim, self->extend_mode)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: Iterator1D_update_offset_dim");
            RescaleIterator_free(&iterator);
            Py_XDECREF(mem1);
            Py_XDECREF(mem2);
            return 0;
        }

        Offsets1D *offsets = iter_node->offsets;
        offsets->init_pos = (offsets->stride / 2) - self->order / 2;
        offsets->dim = self->order + 1;

        *(dims + iter_node->axis) = (npy_intp)((double)iter_node->dim * factor->factor + 0.5);
        f = ((double)(iter_node->dim - 1) / (double)(*(dims + iter_node->axis) - 1));

        RescaleIterator_update_output(iterator, (int)*(dims + iter_node->axis));

        if (factor->next == NULL) {
            *output = (PyArrayObject *)PyArray_EMPTY(self->ndim, dims, self->numtype, 0);
            curr_output = *output;
        } else if (!mem_i) {
            Py_XDECREF(mem1);
            mem1 = (PyArrayObject *)PyArray_EMPTY(self->ndim, dims, self->numtype, 0);
            curr_output = mem1;
            mem_i = 1;
        } else {
            Py_XDECREF(mem2);
            mem2 = (PyArrayObject *)PyArray_EMPTY(self->ndim, dims, self->numtype, 0);
            curr_output = mem2;
            mem_i = 0;
        }

        if (curr_output == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
            RescaleIterator_free(&iterator);
            Py_XDECREF(mem1);
            Py_XDECREF(mem2);
            return 0;
        }

        out_ptr = (void *)PyArray_DATA(curr_output);
        rescale1D(iterator->input, iterator->output, offsets, f, self->numtype, inp_ptr, out_ptr, self->cval);

        inp_ptr = out_ptr;
        RescaleIterator_update_input(iterator);
        factor = factor->next;

        IteratorND_next_axis(&(self->iterator));

    }

    Py_XDECREF(mem1);
    Py_XDECREF(mem2);
    RescaleIterator_free(&iterator);
    return 1;
}

// #####################################################################################################################
// ********************************************** build ****************************************************************

void CLayerPy_dealloc(CLayer *self)
{
    Layer_free(&(self->layer));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *CLayerPy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CLayer *self;
    self = (CLayer *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->layer = NULL;
        if (!Layer_new(&(self->layer), 0, DEFAULT_ITEMSIZE, DEFAULT_TYPE)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: Layer_new");
            CLayerPy_dealloc(self);
            self = NULL;
        }
    }
    return (PyObject *)self;
}

int CLayerPy_init(CLayer *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"", "order", "dtype", "scalespace", "factors", "padding", "cval" , NULL};
    Layer_free(&(self->layer));
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O&|O&O&O&O&O&O&", kwlist,
                                     IO_parse_ndim, &(self->layer),
                                     IO_parse_order, &(self->layer),
                                     IO_parse_descr, &(self->layer),
                                     IO_parse_scalespace, &(self->layer),
                                     IO_parse_factors, &(self->layer),
                                     IO_parse_extend_mode, &(self->layer),
                                     IO_parse_cval, &(self->layer))) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArg_ParseTupleAndKeywords");
        return -1;
    }
    return 1;
}

// ********************************************* setter ****************************************************************

int CLayer_set_input_dtype(CLayer *self, PyObject *descr)
{return IO_parse_descr(descr, &(self->layer)) ? 0 : -1;}

int CLayer_set_padding_mode(CLayer *self, PyObject *extend_mode)
{return IO_parse_extend_mode(extend_mode, &(self->layer)) ? 0 : -1;}

int CLayer_set_order(CLayer *self, PyObject *order)
{return IO_parse_order(order, &(self->layer)) ? 0 : -1;}

int CLayer_set_scalespace(CLayer *self, PyObject *scalespace)
{return IO_parse_scalespace(scalespace, &(self->layer)) ? 0 : -1;}

int CLayer_set_factors(CLayer *self, PyObject *factors)
{return IO_parse_factors(factors, &(self->layer)) ? 0 : -1;}

int CLayer_set_input_dims(CLayer *self, PyObject *dims)
{return IO_parse_input_dims(dims, &(self->layer)) ? 0 : -1;}

int CLayer_set_cval(CLayer *self, PyObject *cval)
{return IO_parse_cval(cval, &(self->layer)) ? 0 : -1;}

// ********************************************* getter ****************************************************************

PyObject *CLayer_get_ndim(CLayer *self)
{return IO_object_build("i", self->layer->ndim);}

PyObject *CLayer_get_input_dtype(CLayer *self)
{return (PyObject *)PyArray_DescrFromType(self->layer->numtype);}

PyObject *CLayer_get_nscales(CLayer *self)
{return IO_object_build("i", self->layer->nscales);}

PyObject *CLayer_get_padding_mode(CLayer *self)
{return IO_object_build("i", self->layer->extend_mode);}

PyObject *CLayer_get_order(CLayer *self)
{return IO_object_build("i", self->layer->order);}

PyObject *CLayer_get_cval(CLayer *self)
{return IO_object_build("d", self->layer->cval);}

PyObject *CLayer_get_scalespace(CLayer *self)
{
    PyObject *output = NULL;
    IO_get_scalespace(self->layer, &output);
    return output;
}

PyObject *CLayer_get_factors(CLayer *self)
{
    PyObject *output = NULL;
    IO_get_factors(self->layer, &output);
    return output;
}

PyObject *CLayer_get_anti_alias_scales(CLayer *self)
{
    PyObject *output = NULL;
    IO_get_anti_alias_scales(self->layer, &output);
    return output;
}

PyObject *CLayer_get_input_dims(CLayer *self)
{
    PyObject *output = NULL;
    IO_get_input_dims(self->layer, &output);
    return output;
}

PyObject *CLayer_get_output_dims(CLayer *self)
{
    PyObject *output = NULL;
    IO_get_output_dims(self->layer, &output);
    return output;
}

// ********************************************* methods ***************************************************************

PyObject *CLayer_scale(CLayer *self, PyObject *args)
{
    PyArrayObject *input = NULL;
    PyArrayObject *output = NULL;

    if (!PyArg_ParseTuple(args, "O&", IO_parse_array, &input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArg_ParseTuple");
        goto exit;
    }

    if (!layer_scale(self->layer, input, &output)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: layer_scale \n");
        goto exit;
    }

    exit:
        Py_XDECREF(input);
        return output == NULL ? Py_BuildValue("") : (PyObject *)output;
}

PyObject *CLayer_rescale(CLayer *self, PyObject *args)
{
    PyArrayObject *input = NULL;
    PyArrayObject *output = NULL;

    if (!PyArg_ParseTuple(args, "O&", IO_parse_array, &input)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArg_ParseTuple");
        goto exit;
    }

    if (!layer_rescale(self->layer, input, &output)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: layer_rescale \n");
        goto exit;
    }

    exit:
        Py_XDECREF(input);
        return output == NULL ? Py_BuildValue("") : (PyObject *)output;
}

PyObject *CLayer_reduce(CLayer *self)
{
    Layer_reduce(self->layer);
    return Py_BuildValue("");
}

// #####################################################################################################################