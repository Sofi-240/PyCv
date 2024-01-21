#include "ops_base.h"

// #####################################################################################################################

int valid_dtype(int dtype_num)
{
    switch (dtype_num) {
        case NPY_BOOL:
            goto exit;
        case NPY_UBYTE:
            goto exit;
        case NPY_USHORT:
            goto exit;
        case NPY_UINT:
            goto exit;
        case NPY_ULONG:
            goto exit;
        case NPY_ULONGLONG:
            goto exit;
        case NPY_BYTE:
            goto exit;
        case NPY_SHORT:
            goto exit;
        case NPY_INT:
            goto exit;
        case NPY_LONG:
            goto exit;
        case NPY_LONGLONG:
            goto exit;
        case NPY_FLOAT:
            goto exit;
        case NPY_DOUBLE:
            goto exit;
        default:
            PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
            goto exit;
    }
    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

void ArrayIterInit(PyArrayObject *array, ArrayIter *iterator)
{
    int ii;
    iterator->nd_m1 = PyArray_NDIM(array) - 1;
    for(ii = 0; ii < PyArray_NDIM(array); ii++) {
        iterator->dims_m1[ii] = PyArray_DIM(array, ii) - 1;
        iterator->coordinates[ii] = 0;
        iterator->strides[ii] = PyArray_STRIDE(array, ii);
        iterator->backstrides[ii] = PyArray_STRIDE(array, ii) * iterator->dims_m1[ii];
    }
}

void CoordinatesIterInit(npy_intp nd, npy_intp *shape, CoordinatesIter *iterator)
{
    npy_intp ii;
    iterator->nd_m1 = nd - 1;
    for(ii = 0; ii < nd; ii++) {
        iterator->dims_m1[ii] = shape[ii] - 1;
        iterator->coordinates[ii] = 0;
    }
}

// #####################################################################################################################

int init_uint8_binary_table(unsigned int **binary_table)
{
    const unsigned int change_point[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    unsigned int *bt_run;
    unsigned int value, counter;
    int ii, jj;

    *binary_table = calloc(256 * 8, sizeof(unsigned int));
    if (!*binary_table) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < 8; ii++) {
        value = 0;
        counter = change_point[ii];
        bt_run = *binary_table + ii;
        for (jj = 0; jj < 256; jj++) {
            bt_run[0] = value;
            counter -= 1;
            if (counter == 0) {
                counter = change_point[ii];
                value = value ? 0 : 1;
            }
            bt_run += 8;
        }
    }

    exit:
        if (PyErr_Occurred()) {
            free(*binary_table);
            return 0;
        }
       return 1;
}

// #####################################################################################################################

int array_to_footprint(PyArrayObject *array, npy_bool **footprint, int *non_zeros)
{
    ArrayIter iter;
    npy_intp array_size, ii, nz = 0;
    npy_bool tmp = NPY_FALSE, *fpo;
    char *po = NULL;
    int num_type;

    array_size = PyArray_SIZE(array);
    ArrayIterInit(array, &iter);

    po = (void *)PyArray_DATA(array);

    *footprint = malloc(array_size * sizeof(npy_bool));
    if (!*footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    fpo = *footprint;
    num_type = PyArray_TYPE(array);
    for (ii = 0; ii < array_size; ii++) {
        GET_VALUE_AS(num_type, npy_bool, po, tmp);
        if (tmp) {
            nz++;
        }
        *fpo++ = tmp;
        ARRAY_ITER_NEXT(iter, po);
    }
    *non_zeros = nz;

    exit:
        if (PyErr_Occurred()) {
            free(*footprint);
            return 0;
        } else {
            return 1;
        }
}

int footprint_as_con(npy_intp nd, int connectivity, npy_bool **footprint, int *non_zeros, int hole)
{
    npy_intp ii, jj, tmp, footprint_size, shape[NPY_MAXDIMS];
    CoordinatesIter iter;
    npy_bool *fpo;
    int nz;

    footprint_size = 1;
    for (ii = 0; ii < nd; ii++) {
        footprint_size *= 3;
        shape[ii] = 3;
    }

    CoordinatesIterInit(nd, shape, &iter);

    *footprint = malloc(footprint_size * sizeof(npy_bool));
    if (!*footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    fpo = *footprint;
    nz = 0;
    for (ii = 0; ii < footprint_size; ii++) {
        tmp = 0;
        for (jj = 0; jj < nd; jj++) {
            tmp += abs(iter.coordinates[jj] - 1);
        }
        if (tmp <= connectivity && (!hole || tmp != 0)) {
            *fpo++ = NPY_TRUE;
            nz++;
        } else {
            *fpo++ = NPY_FALSE;
        }
        COORDINATES_ITER_NEXT(iter);
    }
    *non_zeros = nz;

    exit:
        if (PyErr_Occurred()) {
            free(*footprint);
            return 0;
        } else {
            return 1;
        }
}

int footprint_for_cc(npy_intp nd, int connectivity, npy_bool **footprint, int *non_zeros)
{
    npy_intp ii, jj, tmp, mid, footprint_size, shape[NPY_MAXDIMS];
    CoordinatesIter iter;
    npy_bool *fpo;

    footprint_size = 1;
    for (ii = 0; ii < nd; ii++) {
        footprint_size *= 3;
        shape[ii] = 3;
    }
    mid = footprint_size / 2;

    CoordinatesIterInit(nd, shape, &iter);

    *footprint = malloc(footprint_size * sizeof(npy_bool));
    if (!*footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    fpo = *footprint;

    for (ii = 0; ii < footprint_size; ii++) {
        tmp = 0;
        for (jj = 0; jj < nd; jj++) {
            tmp += abs(iter.coordinates[jj] - 1);
        }
        if (tmp <= connectivity && ii <= mid) {
            *fpo++ = NPY_TRUE;
        } else {
            *fpo++ = NPY_FALSE;
        }
        COORDINATES_ITER_NEXT(iter);
    }
    *non_zeros = mid;

    exit:
        if (PyErr_Occurred()) {
            free(*footprint);
            return 0;
        } else {
            return 1;
        }
}

int copy_data_as_double(PyArrayObject *array, double **line, npy_bool *footprint)
{
    ArrayIter iter;
    npy_intp array_size, line_size = 0, ii;
    char *po = NULL;
    double tmp = 0.0, *lpo;
    int num_type;

    num_type = PyArray_TYPE(array);
    array_size = PyArray_SIZE(array);
    ArrayIterInit(array, &iter);

    if (!footprint) {
        line_size = array_size;
    } else {
        for (ii = 0; ii < array_size; ii++) {
            if (footprint[ii]) {
                line_size++;
            }
        }
    }

    po = (void *)PyArray_DATA(array);

    *line = malloc(array_size * sizeof(double));
    if (!*line) {
        PyErr_NoMemory();
        goto exit;
    }
    lpo = *line;

    for (ii = 0; ii < array_size; ii++) {
        if (footprint[ii]) {
            GET_VALUE_AS(num_type, double, po, tmp);
            *lpo++ = tmp;
        }
        ARRAY_ITER_NEXT(iter, po);
    }

    exit:
        if (PyErr_Occurred()) {
            free(*line);
            return 0;
        } else {
            return 1;
        }
}

// #####################################################################################################################

int init_offsets_ravel(PyArrayObject *array,
                       npy_intp *kernel_shape,
                       npy_intp *kernel_origins,
                       npy_bool *footprint,
                       npy_intp **offsets)
{
    CoordinatesIter iter;
    npy_intp ii, jj, nd, kernel_size, non_zeros, index_ravel;
    npy_intp array_strides[NPY_MAXDIMS], origins[NPY_MAXDIMS], position;
    npy_intp *of_po;

    nd = PyArray_NDIM(array);
    CoordinatesIterInit(nd, kernel_shape, &iter);
    kernel_size = 1;

    for (ii = 0; ii < nd; ii++) {
        array_strides[ii] = PyArray_STRIDE(array, ii);

        kernel_size *= kernel_shape[ii];
        origins[ii] = kernel_origins ? *kernel_origins++ : kernel_shape[ii] / 2;
    }

    if (!footprint) {
        non_zeros = kernel_size;
    } else {
        non_zeros = 0;
        for (ii = 0; ii < kernel_size; ii++) {
            if (footprint[ii]) {
                non_zeros++;
            }
        }
    }

    *offsets = malloc(non_zeros * sizeof(npy_intp));
    if (!*offsets) {
        PyErr_NoMemory();
        goto exit;
    }
    of_po = *offsets;

    for (ii = 0; ii < kernel_size; ii++) {
        if (!footprint || footprint[ii]) {
            index_ravel = 0;
            for(jj = 0; jj < nd; jj++) {
                position = iter.coordinates[jj] - origins[jj];
                index_ravel += position * array_strides[jj];
            }
            *of_po++ = index_ravel;
        }
        COORDINATES_ITER_NEXT(iter);
    }
    exit:
        if (PyErr_Occurred()) {
            free(*offsets);
            return 0;
        } else {
            return 1;
        }
}

int init_offsets_coordinates(npy_intp nd,
                             npy_intp *kernel_shape,
                             npy_intp *kernel_origins,
                             npy_bool *footprint,
                             npy_intp **offsets)
{
    CoordinatesIter iter;
    npy_intp ii, jj, kernel_size, non_zeros;
    npy_intp origins[NPY_MAXDIMS];
    npy_intp *of_po;

    CoordinatesIterInit(nd, kernel_shape, &iter);
    kernel_size = 1;

    for (ii = 0; ii < nd; ii++) {
        kernel_size *= kernel_shape[ii];
        origins[ii] = kernel_origins ? *kernel_origins++ : kernel_shape[ii] / 2;
    }

    if (!footprint) {
        non_zeros = kernel_size;
    } else {
        non_zeros = 0;
        for (ii = 0; ii < kernel_size; ii++) {
            if (footprint[ii]) {
                non_zeros++;
            }
        }
    }

    *offsets = malloc(non_zeros * nd * sizeof(npy_intp));
    if (!*offsets) {
        PyErr_NoMemory();
        goto exit;
    }
    of_po = *offsets;

    for (ii = 0; ii < kernel_size; ii++) {
        if (!footprint || footprint[ii]) {
            for (jj = 0; jj < nd; jj++) {
                *of_po++ = iter.coordinates[jj] - origins[jj];
            }
        }
        COORDINATES_ITER_NEXT(iter);
    }
    exit:
        if (PyErr_Occurred()) {
            free(*offsets);
            return 0;
        } else {
            return 1;
        }
}

int init_borders_lut(npy_intp nd,
                     npy_intp *array_shape,
                     npy_intp *kernel_shape,
                     npy_intp *kernel_origins,
                     npy_bool **borders_lookup)
{
    CoordinatesIter iter;
    npy_intp ii, jj, array_size;
    npy_intp origins[NPY_MAXDIMS], k_shape[NPY_MAXDIMS], a_shape[NPY_MAXDIMS];
    npy_bool *lut;
    int is_border;

    CoordinatesIterInit(nd, array_shape, &iter);

    array_size = 1;

    for (ii = 0; ii < nd; ii++) {
        a_shape[ii] = *array_shape++;
        array_size *= a_shape[ii];

        k_shape[ii] = *kernel_shape++;
        origins[ii] = kernel_origins ? *kernel_origins++ : k_shape[ii] / 2;
    }

    *borders_lookup = malloc(array_size * sizeof(npy_bool));
    if (!*borders_lookup) {
        PyErr_NoMemory();
        goto exit;
    }
    lut = *borders_lookup;

    for (ii = 0; ii < array_size; ii++) {
        is_border = 0;
        for (jj = 0; jj < nd; jj++) {
            if (iter.coordinates[jj] < origins[jj] || iter.coordinates[jj] > a_shape[jj] - k_shape[jj] + origins[jj]) {
                is_border = 1;
                break;
            }
        }
        *lut++ = is_border ? NPY_TRUE : NPY_FALSE;
        COORDINATES_ITER_NEXT(iter);
    }
    exit:
        if (PyErr_Occurred()) {
            free(*borders_lookup);
            return 0;
        } else {
            return 1;
        }
}

int array_offsets_to_list_offsets(PyArrayObject *array, npy_intp *offsets, npy_intp **list_offsets)
{
    npy_intp itemsize, ii, *lof;
    int offsets_size;

    itemsize = PyArray_ITEMSIZE(array);
    offsets_size = sizeof(offsets) / sizeof(npy_intp);

    *list_offsets = malloc(offsets_size * sizeof(npy_intp));
    if (!*list_offsets) {
        PyErr_NoMemory();
        goto exit;
    }
    lof = *list_offsets;

    for (ii = 0; ii < offsets_size; ii++) {
        *lof++ = offsets[ii] / itemsize;
    }

    exit:
        if (PyErr_Occurred()) {
            free(*list_offsets);
            return 0;
        } else {
            return 1;
        }
}

#define RAVEL_COORDINATE(_nd, _position, _strides, _valid_offset)                             \
{                                                                                             \
    int _ii;                                                                                  \
    _valid_offset = 0;                                                                        \
    for (_ii = 0; _ii < _nd; _ii++) {                                                         \
        _valid_offset += _position[_ii] * _strides[_ii];                                      \
    }                                                                                         \
}

int init_offsets_lut(PyArrayObject *array,
                     npy_intp *kernel_shape,
                     npy_intp *kernel_origins,
                     npy_bool *footprint,
                     npy_intp **offsets_lookup,
                     npy_intp *offsets_stride,
                     npy_intp *offsets_flag,
                     BordersMode mode)
{
    CoordinatesIter a_iter, k_iter;
    npy_intp ii, jj, kk, nd, array_size, kernel_size, offsets_size, itemsize;
    npy_intp flag, stride_pos, max_dims = 0, max_stride = 0;
    npy_intp origins[NPY_MAXDIMS], k_shape[NPY_MAXDIMS], a_shape[NPY_MAXDIMS], a_stride[NPY_MAXDIMS], pos, position[NPY_MAXDIMS];
    npy_intp valid_offset = 0, *lut = NULL;
    int is_valid;
    int atype = 0;

    nd = PyArray_NDIM(array);
    itemsize = PyArray_ITEMSIZE(array);
    array_size = PyArray_SIZE(array);
    kernel_size = 1;

    for (ii = 0; ii < nd; ii++) {
        a_shape[ii] = PyArray_DIM(array, ii);
        a_stride[ii] = PyArray_STRIDE(array, ii);

        k_shape[ii] = *kernel_shape++;
        origins[ii] = kernel_origins ? *kernel_origins++ : k_shape[ii] / 2;
        kernel_size *= k_shape[ii];

        max_dims = max_dims < a_shape[ii] ? a_shape[ii] : max_dims;
        stride_pos = a_stride[ii] < 0 ? -a_stride[ii] : a_stride[ii];
        max_stride = max_stride < stride_pos ? stride_pos : max_stride;

        position[ii] = 0;
    }

    *offsets_flag = flag = max_stride * max_stride + 1;

    if (mode > (int)BORDER_BUFFER_ATYPE) {
        atype = 1;
    }

    if (!footprint) {
        offsets_size = kernel_size;
    } else {
        offsets_size = 0;
        for (ii = 0; ii < kernel_size; ii++) {
            if (footprint[ii]) {
                offsets_size++;
            }
        }
    }

    *offsets_stride = offsets_size;
    CoordinatesIterInit(nd, a_shape, &a_iter);
    CoordinatesIterInit(nd, k_shape, &k_iter);

    *offsets_lookup = malloc(array_size * offsets_size * sizeof(npy_intp));
    if (!*offsets_lookup) {
        PyErr_NoMemory();
        goto exit;
    }
    lut = *offsets_lookup;

    for (ii = 0; ii < array_size; ii++) {
        for (jj = 0; jj < kernel_size; jj++) {
            if (!footprint || footprint[jj]) {
                switch (mode) {
                    case BORDER_FLAG:
                    case BORDER_ATYPE_FLAG:
                    case BORDER_VALID:
                    case BORDER_ATYPE_VALID:
                    case BORDER_CONSTANT:
                    case BORDER_ATYPE_CONSTANT:
                        is_valid = 1;
                        for (kk = 0; kk < nd; kk++) {
                            pos = k_iter.coordinates[kk] - origins[kk] + a_iter.coordinates[kk];
                            if (pos < 0 || pos >= a_shape[kk]) {
                                is_valid = 0;
                                break;
                            }
                            position[kk] = pos - a_iter.coordinates[kk];
                        }
                        if (is_valid) {
                            RAVEL_COORDINATE(nd, position, a_stride, valid_offset);
                        } else {
                            valid_offset = flag;
                        }
                        break;
                    case BORDER_REFLECT:
                    case BORDER_ATYPE_REFLECT:
                        for (kk = 0; kk < nd; kk++) {
                            pos = k_iter.coordinates[kk] - origins[kk] + a_iter.coordinates[kk];
                            if (pos < 0) {
                                pos = abs(pos);
                                if (pos >= a_shape[kk]) {
                                    pos = a_shape[kk] - 1;
                                }
                            } else if (pos >= a_shape[kk]) {
                                pos = 2 * a_shape[kk] - pos - 2;
                                if (pos < 0) {
                                    pos = 0;
                                }
                            }
                            position[kk] = pos - a_iter.coordinates[kk];
                        }
                        RAVEL_COORDINATE(nd, position, a_stride, valid_offset);
                        break;
                    case BORDER_SYMMETRIC:
                    case BORDER_ATYPE_SYMMETRIC:
                        for (kk = 0; kk < nd; kk++) {
                            pos = k_iter.coordinates[kk] - origins[kk] + a_iter.coordinates[kk];
                            if (pos < 0) {
                                pos = abs(pos) - 1;
                                if (pos >= a_shape[kk]) {
                                    pos = a_shape[kk] - 1;
                                }
                            } else if (pos >= a_shape[kk]) {
                                pos = 2 * a_shape[kk] - pos - 1;
                                if (pos < 0) {
                                    pos = 0;
                                }
                            }
                            position[kk] = pos - a_iter.coordinates[kk];
                        }
                        RAVEL_COORDINATE(nd, position, a_stride, valid_offset);
                        break;
                    case BORDER_WRAP:
                    case BORDER_ATYPE_WRAP:
                        for (kk = 0; kk < nd; kk++) {
                            pos = k_iter.coordinates[kk] - origins[kk] + a_iter.coordinates[kk];
                            if (pos < 0) {
                                pos = a_shape[kk] + pos;
                            } else if (pos >= a_shape[kk]) {
                                pos -= a_shape[kk];
                            }
                            position[kk] = pos - a_iter.coordinates[kk];
                        }
                        RAVEL_COORDINATE(nd, position, a_stride, valid_offset);
                        break;
                    case BORDER_EDGE:
                    case BORDER_ATYPE_EDGE:
                        for (kk = 0; kk < nd; kk++) {
                            pos = k_iter.coordinates[kk] - origins[kk] + a_iter.coordinates[kk];
                            if (pos < 0) {
                                pos = 0;
                            } else if (pos >= a_shape[kk]) {
                                pos = a_shape[kk] - 1;
                            }
                            position[kk] = pos - a_iter.coordinates[kk];
                        }
                        RAVEL_COORDINATE(nd, position, a_stride, valid_offset);
                        break;
                }
                if (atype) {
                    valid_offset /= itemsize;
                }
                *lut++ = valid_offset;
            }
            COORDINATES_ITER_NEXT(k_iter);
        }
        COORDINATES_ITER_RESET(k_iter);
        COORDINATES_ITER_NEXT(a_iter);
    }

    exit:
        if (PyErr_Occurred()) {
            free(*offsets_lookup);
            return 0;
        } else {
            return 1;
        }

}

// #####################################################################################################################


