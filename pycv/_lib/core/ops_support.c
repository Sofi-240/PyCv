#include "ops_support.h"

// #####################################################################################################################

npy_intp RAVEL_INDEX(npy_intp *index, npy_intp *array_shape, npy_intp nd_m1)
{
    npy_intp r_index, a_offsets, ii;

    r_index = index[nd_m1];
    a_offsets = 1;

    for (ii = nd_m1 - 1; ii <= 0; ii--) {
        a_offsets *= array_shape[ii + 1];
        r_index += a_offsets * index[ii];
    }
    return r_index;
}

npy_intp UNRAVEL_INDEX(npy_intp index, npy_intp *array_shape, npy_intp nd_m1)
{
    npy_intp u_index[NPY_MAXDIMS], a_offsets[NPY_MAXDIMS];
    npy_intp ii;

    a_offsets[nd_m1] = 1;

    for (ii = nd_m1 - 1; ii <= 0; ii--) {
        a_offsets[ii] = a_offsets[ii + 1] * array_shape[ii + 1];
    }

    for (ii = 0; ii <= nd_m1; ii++) {
        u_index[ii] = index / a_offsets[ii];
        index -= u_index[ii] * a_offsets[ii];
    }
    return u_index;
}

// #####################################################################################################################

int INIT_Base_Iterator(PyArrayObject *array, Base_Iterator *iterator)
{
    int ii;

    iterator->nd_m1 = PyArray_NDIM(array) - 1;
    for(ii = 0; ii < PyArray_NDIM(array); ii++) {
        iterator->dims_m1[ii] = PyArray_DIM(array, ii) - 1;
        iterator->coordinates[ii] = 0;
        iterator->strides[ii] = PyArray_STRIDE(array, ii);
        iterator->backstrides[ii] = PyArray_STRIDE(array, ii) * iterator->dims_m1[ii];
    }
    return 1;
}

// #####################################################################################################################

int INIT_FOOTPRINT(PyArrayObject *kernel, npy_bool **footprint, int *footprint_size)
{
    Base_Iterator dptr_k;
    char *ki;
    npy_intp ii, filter_size;
    npy_bool val, *fo = NULL;

    if (!INIT_Base_Iterator(kernel, &dptr_k)){
        goto exit;
    }

    filter_size = PyArray_SIZE(kernel);
    *footprint = malloc(filter_size * sizeof(npy_bool));

    if (!*footprint) {
        PyErr_NoMemory();
        goto exit;
    }

    fo = *footprint;
    ki = (void *)PyArray_DATA(kernel);
    *footprint_size = 0;

    for (ii = 0; ii < filter_size; ii++) {
        val = *(double *)ki != 0 ? NPY_TRUE : NPY_FALSE;
        *fo++ = val;
        if (val) {
            *footprint_size += 1;
        }
        BASE_ITERATOR_NEXT(dptr_k, ki);

    }

    exit:
        if (PyErr_Occurred()) {
            free(*footprint);
            return 0;
        } else {
            return 1;
        }
}

int INIT_OFFSETS(PyArrayObject *array,
                 npy_intp *kernel_shape,
                 npy_intp *kernel_origins,
                 npy_bool *footprint,
                 npy_intp **offsets,
                 npy_bool **borders_lookup)
{
    npy_intp nd_m1, ii, jj, kernel_size, footprint_size, array_size, index_ravel;
    npy_intp array_dims[NPY_MAXDIMS], array_strides[NPY_MAXDIMS];
    npy_intp filter_dims[NPY_MAXDIMS], origins[NPY_MAXDIMS];
    npy_intp position[NPY_MAXDIMS], kernel_coordinates[NPY_MAXDIMS], array_coordinates[NPY_MAXDIMS];
    npy_intp *of;
    npy_bool *borders;
    int is_border;

    nd_m1 = PyArray_NDIM(array) - 1;
    array_size = PyArray_SIZE(array);
    kernel_size = 1;

    for (ii = 0; ii <= nd_m1; ii++) {
        array_dims[ii] = PyArray_DIM(array, ii);
        array_strides[ii] = PyArray_STRIDE(array, ii);

        filter_dims[ii] = *kernel_shape++;
        origins[ii] = kernel_origins ? *kernel_origins++ : filter_dims[ii] / 2;
        kernel_size *= filter_dims[ii];

        position[ii] = 0;
        kernel_coordinates[ii] = 0;
        array_coordinates[ii] = 0;
    }


    if (!footprint) {
        footprint_size = kernel_size;
    } else {
        footprint_size = 0;
        for (ii = 0; ii < kernel_size; ii++) {
            if (footprint[ii]) {
                footprint_size += 1;
            }
        }
    }

    *offsets = malloc(footprint_size * sizeof(npy_intp));
    if (!*offsets) {
        PyErr_NoMemory();
        goto exit;
    }
    of = *offsets;

    for (ii = 0; ii < kernel_size; ii++) {
        if (!footprint || footprint[ii]) {
            for(jj = 0; jj <= nd_m1; jj++) {
                position[jj] = kernel_coordinates[jj] - origins[jj];
            }

            index_ravel = position[nd_m1] * array_strides[nd_m1];

            for(jj = nd_m1 - 1; jj >= 0; jj--) {
                index_ravel += position[jj] * array_strides[jj];
            }
            *of++ = index_ravel;
        }
        for (jj = nd_m1; jj >= 0; jj--) {
            if (kernel_coordinates[jj] < filter_dims[jj] - 1) {
                kernel_coordinates[jj]++;
                break;
            }
            else {
                kernel_coordinates[jj] = 0;
            }
        }
    }

    *borders_lookup = malloc(array_size * sizeof(npy_bool));
    if (!*borders_lookup) {
        PyErr_NoMemory();
        goto exit;
    }
    borders = *borders_lookup;
    is_border = 1;

    for (ii = 0; ii < array_size; ii++) {

        *borders++ = is_border ? NPY_TRUE : NPY_FALSE;

        for (jj = nd_m1; jj >= 0; jj--) {
            if (array_coordinates[jj] < array_dims[jj] - 1) {
                array_coordinates[jj]++;
                break;
            }
            else {
                array_coordinates[jj] = 0;
            }
        }
        is_border = 0;
        for (jj = 0; jj <= nd_m1; jj++) {
            if (array_coordinates[jj] < origins[jj] || array_coordinates[jj] > array_dims[jj] - filter_dims[jj] + origins[jj]) {
                is_border = 1;
                break;
            }
        }
    }

    exit:
        if (PyErr_Occurred()) {
            free(*offsets);
            free(*borders_lookup);
            return 0;
        } else {
            return 1;
        }

}

int INIT_OFFSETS_ARRAY(PyArrayObject *array,
                       npy_intp *kernel_shape,
                       npy_intp *kernel_origins,
                       npy_bool *footprint,
                       npy_intp **offsets,
                       npy_intp *offsets_flag,
                       npy_intp *offsets_stride)
{

    npy_intp nd_m1, ii, jj, kk, kernel_size, footprint_size, array_size, offsets_size;
    npy_intp flag, stride, max_dim = 0, max_stride = 0;
    npy_intp array_dims[NPY_MAXDIMS], array_strides[NPY_MAXDIMS];
    npy_intp filter_dims[NPY_MAXDIMS], origins[NPY_MAXDIMS];
    npy_intp position[NPY_MAXDIMS], kernel_coordinates[NPY_MAXDIMS], array_coordinates[NPY_MAXDIMS];
    npy_intp *kernel_offsets, *of;
    int is_border, is_valid;

    nd_m1 = PyArray_NDIM(array) - 1;
    array_size = PyArray_SIZE(array);
    kernel_size = 1;

    for (ii = 0; ii <= nd_m1; ii++) {
        array_dims[ii] = PyArray_DIM(array, ii);
        array_strides[ii] = PyArray_STRIDE(array, ii);

        if (array_dims[ii] > max_dim) {
            max_dim = array_dims[ii];
        }
        stride = array_strides[ii] > 0 ? array_strides[ii] : -array_strides[ii];
        if (array_strides[ii] > max_stride) {
            max_stride = array_strides[ii];
        }

        filter_dims[ii] = *kernel_shape++;
        origins[ii] = kernel_origins ? *kernel_origins++ : filter_dims[ii] / 2;
        kernel_size *= filter_dims[ii];

        position[ii] = 0;
        kernel_coordinates[ii] = 0;
        array_coordinates[ii] = 0;
    }


    if (!footprint) {
        footprint_size = kernel_size;
    } else {
        footprint_size = 0;
        for (ii = 0; ii < kernel_size; ii++) {
            if (footprint[ii]) {
                footprint_size += 1;
            }
        }
    }

    *offsets_stride = footprint_size;

    kernel_offsets = (npy_intp *)malloc(kernel_size * sizeof(npy_intp));
    if (!kernel_offsets) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < kernel_size; ii++) {
        for(jj = 0; jj <= nd_m1; jj++) {
            position[jj] = kernel_coordinates[jj] - origins[jj];
        }

        kernel_offsets[ii] = position[nd_m1] * array_strides[nd_m1];

        for(jj = nd_m1 - 1; jj >= 0; jj--) {
            kernel_offsets[ii] += position[jj] * array_strides[jj];
        }

        for (jj = nd_m1; jj >= 0; jj--) {
            if (kernel_coordinates[jj] < filter_dims[jj] - 1) {
                kernel_coordinates[jj]++;
                break;
            }
            else {
                kernel_coordinates[jj] = 0;
            }
        }
    }

    offsets_size = array_size * footprint_size;

    *offsets = malloc(offsets_size * sizeof(npy_intp));
    if (!*offsets) {
        PyErr_NoMemory();
        goto exit;
    }
    of = *offsets;
    flag = max_dim * max_stride + 1;
    *offsets_flag = flag;
    is_border = 1;

    for (ii = 0; ii < array_size; ii++) {
        for (jj = 0; jj < kernel_size; jj++) {
            if (!footprint || footprint[jj]) {
                if (is_border) {
                    is_valid = 1;
                    for(kk = 0; kk <= nd_m1; kk++) {
                        position[kk] = kernel_coordinates[kk] - origins[kk] + array_coordinates[kk];
                        if (position[kk] < 0 || position[kk] >= array_dims[kk]) {
                            is_valid = 0;
                            break;
                        }
                    }
                    *of++ = is_valid ? kernel_offsets[jj] : flag;
                } else {
                    *of++ = kernel_offsets[jj];
                }
            }
            for (kk = nd_m1; kk >= 0; kk--) {
                if (kernel_coordinates[kk] < filter_dims[kk] - 1) {
                    kernel_coordinates[kk]++;
                    break;
                }
                else {
                    kernel_coordinates[kk] = 0;
                }
            }
        }
        for (jj = nd_m1; jj >= 0; jj--) {
            if (array_coordinates[jj] < array_dims[jj] - 1) {
                array_coordinates[jj]++;
                break;
            }
            else {
                array_coordinates[jj] = 0;
            }
        }
        is_border = 0;
        for (jj = 0; jj <= nd_m1; jj++) {
            if (array_coordinates[jj] < origins[jj] || array_coordinates[jj] > array_dims[jj] - filter_dims[jj] + origins[jj]) {
                is_border = 1;
                break;
            }
        }
    }


    exit:
        if (PyErr_Occurred()) {
            free(*offsets);
            return 0;
        } else {
            return 1;
        }

}

// #####################################################################################################################

int INIT_Neighborhood_Iterator(npy_intp *neighborhood_size, npy_intp *array_size, Neighborhood_Iterator *iterator)
{
    iterator->strides = neighborhood_size;
    iterator->ptr = 0;
    iterator->bound = array_size;
    return 1;
}


// #####################################################################################################################
