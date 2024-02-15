#include "c_pycv_base.h"
#include "c_pycv_features.h"

// #####################################################################################################################

#define PYCV_F_AXIS_ITERATOR_NEXT(_iterator, _pointer, _axis)                                                          \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    if ((_iterator).coordinates[_axis] < (_iterator).dims_m1[_axis]) {                                                 \
        (_iterator).coordinates[_axis]++;                                                                              \
        _pointer += (_iterator).strides[_axis];                                                                        \
    } else {                                                                                                           \
        (_iterator).coordinates[_axis] = 1;                                                                            \
        _pointer -= (_iterator).strides_back[_axis];                                                                   \
        _pointer += (_iterator).strides[_axis];                                                                        \
        for (_ii = (_iterator).nd_m1; _ii >= 0; _ii--) {                                                               \
            if (_ii == _axis){                                                                                         \
                continue;                                                                                              \
            }                                                                                                          \
            if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                                             \
                (_iterator).coordinates[_ii]++;                                                                        \
                _pointer += (_iterator).strides[_ii];                                                                  \
                break;                                                                                                 \
            } else {                                                                                                   \
                (_iterator).coordinates[_ii] = 0;                                                                      \
                _pointer -= (_iterator).strides_back[_ii];                                                             \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_F_CASE_INTEGRAL_ADD(_NTYPE, _dtype, _pointer, _stride)                                                    \
case NPY_##_NTYPE:                                                                                                     \
    *(_dtype *)_pointer += *(_dtype *)(_pointer - _stride);                                                            \
    break

#define PYCV_F_INTEGRAL_ADD(_NTYPE, _pointer, _stride)                                                                 \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_F_CASE_INTEGRAL_ADD(BOOL, npy_bool, _pointer, _stride);                                                   \
        PYCV_F_CASE_INTEGRAL_ADD(UBYTE, npy_ubyte, _pointer, _stride);                                                 \
        PYCV_F_CASE_INTEGRAL_ADD(USHORT, npy_ushort, _pointer, _stride);                                               \
        PYCV_F_CASE_INTEGRAL_ADD(UINT, npy_uint, _pointer, _stride);                                                   \
        PYCV_F_CASE_INTEGRAL_ADD(ULONG, npy_ulong, _pointer, _stride);                                                 \
        PYCV_F_CASE_INTEGRAL_ADD(ULONGLONG, npy_ulonglong, _pointer, _stride);                                         \
        PYCV_F_CASE_INTEGRAL_ADD(BYTE, npy_byte, _pointer, _stride);                                                   \
        PYCV_F_CASE_INTEGRAL_ADD(SHORT, npy_short, _pointer, _stride);                                                 \
        PYCV_F_CASE_INTEGRAL_ADD(INT, npy_int, _pointer, _stride);                                                     \
        PYCV_F_CASE_INTEGRAL_ADD(LONG, npy_long, _pointer, _stride);                                                   \
        PYCV_F_CASE_INTEGRAL_ADD(LONGLONG, npy_longlong, _pointer, _stride);                                           \
        PYCV_F_CASE_INTEGRAL_ADD(FLOAT, npy_float, _pointer, _stride);                                                 \
        PYCV_F_CASE_INTEGRAL_ADD(DOUBLE, npy_double, _pointer, _stride);                                               \
    }                                                                                                                  \
}

int PYCV_integral_image(PyArrayObject *output)
{
    npy_intp array_size;
    int num_type;
    npy_intp jj, aa, integral_size[NPY_MAXDIMS];

    PYCV_ArrayIterator iter;
    char *po = NULL, *po_base = NULL;

    NPY_BEGIN_THREADS_DEF;

    array_size = PyArray_SIZE(output);

    PYCV_ArrayIteratorInit(output, &iter);

    num_type = PyArray_TYPE(output);

    for (aa = 0; aa <= iter.nd_m1; aa++) {
        integral_size[aa] = 1;
        for (jj = 0; jj <= iter.nd_m1; jj++) {
            integral_size[aa] *= (jj == aa ? iter.dims_m1[jj] : (iter.dims_m1[jj] + 1));
        }
    }

    NPY_BEGIN_THREADS;

    po_base = po = (void *)PyArray_DATA(output);

    for (aa = 0; aa <= iter.nd_m1; aa++) {
        po = po_base;
        PYCV_F_AXIS_ITERATOR_NEXT(iter, po, aa);

        for (jj = 0; jj < integral_size[aa]; jj++) {
             PYCV_F_INTEGRAL_ADD(num_type, po, iter.strides[aa]);
             PYCV_F_AXIS_ITERATOR_NEXT(iter, po, aa);
        }
        PYCV_ARRAY_ITERATOR_RESET(iter);
    }

    NPY_END_THREADS;

    exit:
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################
