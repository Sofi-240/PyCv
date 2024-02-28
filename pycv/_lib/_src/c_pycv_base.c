#include "c_pycv_base.h"

// #####################################################################################################################

int PYCV_valid_dtype(int num_type)
{
    switch (num_type) {
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
        case NPY_FLOAT:
        case NPY_DOUBLE:
            break;
        default:
            return 0;
    }
    return 1;
}

// #####################################################################################################################

void PYCV_ArrayIteratorInit(PyArrayObject *array, PYCV_ArrayIterator *iterator)
{
    int ii;
    iterator->nd_m1 = (npy_intp)PyArray_NDIM(array) - 1;
    for (ii = 0; ii < PyArray_NDIM(array); ii++) {
        iterator->dims_m1[ii] = PyArray_DIM(array, ii) - 1;
        iterator->strides[ii] = PyArray_STRIDE(array, ii);
        iterator->strides_back[ii] = iterator->strides[ii] * iterator->dims_m1[ii];
        iterator->coordinates[ii] = 0;
    }
}

void PYCV_CoordinatesIteratorInit(npy_intp ndim, npy_intp *dims, PYCV_CoordinatesIterator *iterator)
{
    npy_intp ii;
    iterator->nd_m1 = ndim - 1;
    for (ii = 0; ii < ndim; ii++) {
        iterator->dims_m1[ii] = dims[ii] - 1;
        iterator->coordinates[ii] = 0;
    }
}

// #####################################################################################################################

npy_intp PYCV_FitCoordinate(npy_intp coordinate, npy_intp dim, npy_intp flag, PYCV_ExtendBorder mode)
{
    npy_intp out = coordinate, dim2;
    if (coordinate >= 0 && coordinate < dim) {
        return out;
    }
    switch (mode) {
        case PYCV_EXTEND_FLAG:
        case PYCV_EXTEND_VALID:
        case PYCV_EXTEND_CONSTANT:
            out = flag;
            break;
        case PYCV_EXTEND_EDGE:
            out = coordinate < 0 ? 0 : dim - 1;
            break;
        case PYCV_EXTEND_WRAP:
            out = coordinate;
            if (coordinate < 0) {
                out += dim * (npy_intp)(-out / dim);
                if (out < 0) {
                    out += dim;
                }
            } else {
                out -= dim * (npy_intp)(out / dim);
            }
            break;
        case PYCV_EXTEND_SYMMETRIC:
            dim2 = 2 * dim;
            out = coordinate;
            if (out < 0) {
                if (out < -dim2) {
                    out += dim2 * (npy_intp)(-out / dim2);
                }
                if (out < -dim) {
                    out += dim2;
                } else {
                    out = -out - 1;
                }
            } else {
                out -= dim2 * (npy_intp)(out / dim2);
                if (out >= dim) {
                    out = dim2 - out - 1;
                }
            }
            break;
        case PYCV_EXTEND_REFLECT:
            dim2 = 2 * dim - 2;
            out = coordinate;
            if (out < 0) {
                out += dim2 * (npy_intp)(-out / dim2);
                if (out <= 1 - dim) {
                    out += dim2;
                } else {
                    out = -out;
                }
            } else {
                out -= dim2 * (npy_intp)(out / dim2);
                if (out >= dim) {
                    out = dim2 - out;
                }
            }
            break;
        default:
            out = -1; // Invalid mode
    }
    return out;
}


int PYCV_InitOffsets(PyArrayObject *array,
                     npy_intp *shape,
                     npy_intp *center,
                     npy_bool *footprint,
                     npy_intp **ravel_offsets,
                     npy_intp **unravel_offsets)
{
    PYCV_CoordinatesIterator iter;
    npy_intp array_strides[NPY_MAXDIMS], ndim;
    npy_intp k_size = 1, f_size, k_center[NPY_MAXDIMS];
    npy_intp ii, jj, ravel, position[NPY_MAXDIMS];
    npy_intp *ro = NULL, *uro = NULL;

    ndim = (npy_intp)PyArray_NDIM(array);

    for (ii = 0; ii < ndim; ii++) {
        array_strides[ii] = (npy_intp)PyArray_STRIDE(array, (int)ii);

        k_center[ii] = center ? *center++ : shape[ii] / 2;
        k_size *= shape[ii];
        position[ii] = 0;
    }

    PYCV_FOOTPRINT_NONZERO(footprint, k_size, f_size);

    if (ravel_offsets) {
        *ravel_offsets = malloc(f_size * sizeof(npy_intp));
        if (!*ravel_offsets) {
            PyErr_NoMemory();
            goto exit;
        }
        ro = *ravel_offsets;
    }

    if (unravel_offsets) {
        *unravel_offsets = malloc(f_size * ndim * sizeof(npy_intp));
        if (!*unravel_offsets) {
            PyErr_NoMemory();
            goto exit;
        }
        uro = *unravel_offsets;
    }

    PYCV_CoordinatesIteratorInit(ndim, shape, &iter);

    for (ii = 0; ii < k_size; ii++) {
        if (!footprint || footprint[ii]) {
            for (jj = 0; jj < ndim; jj++) {
                position[jj] = iter.coordinates[jj] - k_center[jj];
                if (uro) {
                    *uro++ = position[jj];
                }
            }
            if (ro) {
                PYCV_RAVEL_COORDINATE(position, ndim, array_strides, ravel);
                *ro++ = ravel;
            }
        }
        PYCV_COORDINATES_ITERATOR_NEXT(iter);
    }

    exit:
        if (PyErr_Occurred()) {
            if (ro) {
                free(*ravel_offsets);
            }
            if (uro) {
                free(*unravel_offsets);
            }
            return 0;
        } else {
            return 1;
        }
}

// #####################################################################################################################

void PYCV_NeighborhoodIteratorInit(PyArrayObject *array,
                                   npy_intp *shape,
                                   npy_intp *center,
                                   npy_intp n,
                                   NeighborhoodIterator *iterator)
{
    npy_intp ii, k_center[NPY_MAXDIMS], array_dims[NPY_MAXDIMS], sz, ndim;

    ndim = (npy_intp)PyArray_NDIM(array);
    iterator->nd_m1 = ndim - 1;

    for (ii = 0; ii < ndim; ii++) {
        array_dims[ii] = PyArray_DIM(array, (int)ii);
        iterator->dims_m1[ii] = array_dims[ii] - 1;
        iterator->strides[ii] = PyArray_STRIDE(array, (int)ii);
        iterator->strides_back[ii] = iterator->strides[ii] * iterator->dims_m1[ii];
        iterator->coordinates[ii] = 0;

        k_center[ii] = center ? *center++ : shape[ii] / 2;
    }

    if (ndim > 0) {
        iterator->nn_strides[ndim - 1] = n;
        for (ii = ndim - 2; ii >= 0; ii--) {
            sz = array_dims[ii + 1] < shape[ii + 1] ? array_dims[ii + 1] : shape[ii + 1];
            iterator->nn_strides[ii] = iterator->nn_strides[ii + 1] * sz;
        }
    }

    for (ii = 0; ii < ndim; ii++) {
        sz = array_dims[ii] < shape[ii] ? array_dims[ii] : shape[ii];
        iterator->nn_strides_back[ii] = iterator->nn_strides[ii] * (sz - 1);
        iterator->boundary_low[ii] = k_center[ii];
        iterator->boundary_high[ii] = array_dims[ii] - shape[ii] + k_center[ii];
    }
}


// #####################################################################################################################

int PYCV_InitNeighborhoodOffsets(PyArrayObject *array,
                                 npy_intp *shape,
                                 npy_intp *center,
                                 npy_bool *footprint,
                                 npy_intp **ravel_offsets,
                                 npy_intp **unravel_offsets,
                                 npy_intp *flag,
                                 PYCV_ExtendBorder mode)
{
    PYCV_CoordinatesIterator iter;
    npy_intp array_strides[NPY_MAXDIMS], array_dims[NPY_MAXDIMS], ndim;
    npy_intp max_dim = 0, max_stride = 0, pos_stride;
    npy_intp k_size = 1, f_size, k_center[NPY_MAXDIMS];
    npy_intp position[NPY_MAXDIMS], array_position[NPY_MAXDIMS], max_index_p1;
    npy_intp ii, jj, kk, ravel, n_offsets = 1, pos;
    npy_intp *ro = NULL, *uro = NULL;
    int valid;

    ndim = (npy_intp)PyArray_NDIM(array);

    for (ii = 0; ii < ndim; ii++) {
        array_dims[ii] = PyArray_DIM(array, (int)ii);
        array_strides[ii] = PyArray_STRIDE(array, (int)ii);

        k_center[ii] = center ? *center++ : shape[ii] / 2;
        k_size *= shape[ii];

        pos_stride = array_strides[ii] < 0 ? -array_strides[ii] : array_strides[ii];
        max_stride = pos_stride > max_stride ? pos_stride : max_stride;
        max_dim = array_dims[ii] > max_dim ? array_dims[ii] : max_dim;

        position[ii] = 0;
        array_position[ii] = 0;

        n_offsets *= (array_dims[ii] < shape[ii] ? array_dims[ii] : shape[ii]);
    }

    PYCV_FOOTPRINT_NONZERO(footprint, k_size, f_size);

    *flag = max_index_p1 = max_dim * max_stride + 1;

    if (ravel_offsets) {
        *ravel_offsets = malloc(n_offsets * f_size * sizeof(npy_intp));
        if (!*ravel_offsets) {
            PyErr_NoMemory();
            goto exit;
        }
        ro = *ravel_offsets;
    }

    if (unravel_offsets) {
        *unravel_offsets = malloc(n_offsets * f_size * ndim * sizeof(npy_intp));
        if (!*unravel_offsets) {
            PyErr_NoMemory();
            goto exit;
        }
        uro = *unravel_offsets;
    }

    PYCV_CoordinatesIteratorInit(ndim, shape, &iter);

    if (PYCV_FitCoordinate(-1, array_dims[0], max_index_p1, mode) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "invalid extend border mode");
        goto exit;
    }

    for (ii = 0; ii < n_offsets; ii++) {
        for (jj = 0; jj < k_size; jj++) {
            if (!footprint || footprint[jj]) {
                valid = 1;
                for (kk = 0; kk < ndim; kk++) {
                    pos = iter.coordinates[kk] - k_center[kk] + array_position[kk];
                    pos = PYCV_FitCoordinate(pos, array_dims[kk], max_index_p1, mode);
                    position[kk] = pos - array_position[kk];
                    if (pos == max_index_p1) {
                        valid = 0;
                    }
                }
                if (ro) {
                    if (valid) {
                        PYCV_RAVEL_COORDINATE(position, ndim, array_strides, ravel);
                    } else {
                        ravel = max_index_p1;
                    }
                    *ro++ = ravel;
                }
                if (uro) {
                    for (kk = 0; kk < ndim; kk++) {
                        *uro++ = valid ? position[kk] : max_index_p1;
                    }
                }
            }
            PYCV_COORDINATES_ITERATOR_NEXT(iter);
        }
        PYCV_COORDINATES_ITERATOR_RESET(iter);

        for (jj = ndim - 1; jj >= 0; jj--) {
            if (array_position[jj] == k_center[jj]) {
                array_position[jj] += array_dims[jj] - shape[jj] + 1;
                if (array_position[jj] <= k_center[jj]) {
                    array_position[jj] = k_center[jj] + 1;
                }
            } else {
                array_position[jj]++;
            }
            if (array_position[jj] < array_dims[jj]) {
                break;
            } else {
                array_position[jj] = 0;
            }
        }
    }
    exit:
        if (PyErr_Occurred()) {
            if (uro) {
                free(*unravel_offsets);
            }
            if (ro) {
                free(*ravel_offsets);
            }
            return 0;
        } else {
            return 1;
        }
}

// #####################################################################################################################


int PYCV_AllocateToFootprint(PyArrayObject *array, npy_bool **footprint, npy_intp *nonzero, int flip)
{
    PYCV_ArrayIterator iter;
    npy_intp ndim, array_size, n = 0, ii;
    npy_bool *fpo = NULL, out;
    char *pi = NULL;
    int num_type;

    ndim = PyArray_NDIM(array);
    array_size = PyArray_SIZE(array);
    num_type = PyArray_TYPE(array);

    PYCV_ArrayIteratorInit(array, &iter);

    pi = (void *)PyArray_DATA(array);

    *footprint = malloc(array_size * sizeof(npy_bool));
    if (!*footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    fpo = *footprint;

    for (ii = array_size - 1; ii >= 0; ii--) {
        PYCV_GET_VALUE(num_type, npy_bool, pi, out);
        n += (out ? 1 : 0);
        if (flip) {
            fpo[ii] = out;
        } else {
            *fpo++ = out;
        }
        PYCV_ARRAY_ITERATOR_NEXT(iter, pi);
    }
    *nonzero = n;

    exit:
        if (PyErr_Occurred()) {
            if (fpo) {
                free(*footprint);
            }
            return 0;
        } else {
            return 1;
        }
}


int PYCV_AllocateKernelFlip(PyArrayObject *kernel, npy_bool **footprint, npy_double **h)
{
    PYCV_ArrayIterator iter;
    npy_intp k_size, ii, jj, f_size = 0;
    char *pk_base = NULL, *pk = NULL;
    npy_bool *ff = NULL;
    npy_double *hh = NULL, tmp = 0.0;
    int num_type;

    k_size = PyArray_SIZE(kernel);
    num_type = PyArray_TYPE(kernel);

    PYCV_ArrayIteratorInit(kernel, &iter);
    pk_base = pk = (void *)PyArray_DATA(kernel);

    if (footprint) {
        *footprint = malloc(k_size * sizeof(npy_bool));
        if (!*footprint) {
            PyErr_NoMemory();
            goto exit;
        }
        ff = *footprint;

        for (ii = k_size - 1; ii >= 0; ii--) {
            PYCV_GET_VALUE(num_type, npy_double, pk, tmp);
            if (fabs(tmp) > DBL_EPSILON) {
                *ff++ = 1;
                f_size += 1;
            } else {
                *ff++ = 0;
            }
            PYCV_ARRAY_ITERATOR_NEXT(iter, pk);
        }
        PYCV_ARRAY_ITERATOR_RESET(iter);
    } else {
        f_size = k_size;
    }

    pk = pk_base;

    if (h) {
        *h = malloc(f_size * sizeof(npy_double));
        if (!*h) {
            PyErr_NoMemory();
            goto exit;
        }
        jj = f_size - 1;
        hh = *h;
        ff = *footprint;
        for (ii = k_size - 1; ii >= 0; ii--) {
            if (!footprint || ff[ii]) {
                PYCV_GET_VALUE(num_type, npy_double, pk, hh[jj]);
                jj -= 1;
            }
            PYCV_ARRAY_ITERATOR_NEXT(iter, pk);
        }
    }

    exit:
        if (PyErr_Occurred()) {
            free(*footprint);
            free(*h);
            return 0;
        } else {
            return 1;
        }
}


int PYCV_DefaultFootprint(npy_intp ndim,
                          npy_intp connectivity,
                          npy_bool **footprint,
                          npy_intp *nonzero,
                          unsigned int one_side)
{
    npy_intp ii, jj, tmp, mid, n = 1, shape[NPY_MAXDIMS], sz = 0;
    PYCV_CoordinatesIterator iter;
    npy_bool *fpo = NULL;

    for (ii = 0; ii < ndim; ii++) {
        n *= 3;
        shape[ii] = 3;
    }
    mid = n / 2;

    PYCV_CoordinatesIteratorInit(ndim, shape, &iter);

    *footprint = malloc(n * sizeof(npy_bool));
    if (!*footprint) {
        PyErr_NoMemory();
        goto exit;
    }
    fpo = *footprint;

    for (ii = 0; ii < n; ii++) {
        tmp = 0;
        for (jj = 0; jj < ndim; jj++) {
            tmp += (npy_intp)abs((int)iter.coordinates[jj] - 1);
        }
        if (tmp <= connectivity && (!one_side || ii < mid)) {
            *fpo++ = 1;
            sz++;
        } else {
            *fpo++ = 0;
        }
        PYCV_COORDINATES_ITERATOR_NEXT(iter);
    }
    *nonzero = sz;

    exit:
        if (PyErr_Occurred()) {
            if (fpo) {
                free(*footprint);
            }
            return 0;
        } else {
            return 1;
        }
}


// #####################################################################################################################

int PYCV_CoordinatesListInit(npy_intp ndim, npy_intp max_size, PYCV_CoordinatesList *object)
{
    npy_intp ii;
    object->ndim = ndim;
    object->max_size = max_size;
    object->coordinates = malloc(max_size * sizeof(npy_intp*));
    if (!object->coordinates) {
        object->coordinates_size = -1;
        return 0;
    }
    for (ii = 0; ii < max_size; ii++) {
        object->coordinates[ii] = malloc(ndim * sizeof(npy_intp));
        if (!object->coordinates[ii]) {
            object->coordinates_size = -1;
            return 0;
        }
    }
    object->coordinates_size = 0;
    return 1;
}

int PYCV_CoordinatesListFree(PYCV_CoordinatesList *object)
{
    npy_intp ii;
    for (ii = 0; ii < object->max_size; ii++) {
        free(object->coordinates[ii]);
    }
    free(object->coordinates);
    object->coordinates_size = 0;
    return 1;
}

// #####################################################################################################################
