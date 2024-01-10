#include "ops_support.h"
#include "morphology.h"


// #####################################################################################################################



// #####################################################################################################################

int ops_binary_erosion(PyArrayObject *input,
                       PyArrayObject *strel,
                       PyArrayObject *output,
                       npy_intp *origins,
                       int iterations,
                       PyArrayObject *mask,
                       int invert)
{
    Base_Iterator dptr_o, dptr_i, dptr_s, dptr_m;
    Neighborhood_Iterator dptr_of;
    char *pi = NULL, *po = NULL, *si = NULL, *ma = NULL;
    npy_bool *footprint;
    npy_intp nd, ii, jj, offsets_stride, offsets_flag, *of, *offsets;
    npy_bool buffer, _true, _false;

    NPY_BEGIN_THREADS_DEF;
    nd = PyArray_NDIM(input);
    if (!INIT_Base_Iterator(strel, &dptr_s)){
        goto exit;
    }
    si = (void *)PyArray_DATA(strel);

    footprint = (npy_bool *)malloc(PyArray_SIZE(strel) * sizeof(npy_bool));

    if (!footprint) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < PyArray_SIZE(strel); ii++) {
        footprint[ii] = *(npy_bool *)si ? NPY_TRUE : NPY_FALSE;
        BASE_ITERATOR_NEXT(dptr_s, si);
    }

    INIT_OFFSETS_ARRAY(input, PyArray_DIMS(strel), origins, footprint, &offsets, &offsets_flag, &offsets_stride);

    if (!INIT_Neighborhood_Iterator(offsets_stride, PyArray_SIZE(input), &dptr_of)){
        goto exit;
    }

    if (!INIT_Base_Iterator(input, &dptr_i)){
        goto exit;
    }

    if (!INIT_Base_Iterator(output, &dptr_o)){
        goto exit;
    }

    if (mask) {
        if (!INIT_Base_Iterator(mask, &dptr_m)){
            goto exit;
        }
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    if (mask) {
        ma = (void *)PyArray_DATA(mask);
    }

    of = offsets;
    if (invert) {
        _true = NPY_FALSE;
        _false = NPY_TRUE;
    } else {
        _true = NPY_TRUE;
        _false = NPY_FALSE;
    }

    for (ii = 0; ii < PyArray_SIZE(input); ii++) {
        if (!mask || *(npy_bool *)(ma)) {
            buffer = _true;
            for (jj = 0; jj < offsets_stride; jj++) {
                if (of[jj] < offsets_flag) {
                    buffer = *(npy_bool *)(pi + of[jj]) == _true ? _true : _false;
                }
                if (buffer == _false) {
                    break;
                }
            }
        } else {
            buffer = *(npy_bool *)pi;
        }
        *(npy_bool *)po = buffer;

        if (mask) {
            BASE_ITERATOR_NEXT3(dptr_i, pi, dptr_o, po, dptr_m, ma)    ;
        } else {
            BASE_ITERATOR_NEXT2(dptr_i, pi, dptr_o, po);
        }
        NEIGHBORHOOD_ITERATOR_NEXT(dptr_of, of);
    }

    NPY_END_THREADS;

    exit:
        free(offsets);
        return PyErr_Occurred() ? 0 : 1;
}