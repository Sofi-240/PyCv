#ifndef OPS_SUPPORT_H
#define OPS_SUPPORT_H

#define NO_IMPORT_ARRAY
#include "ops.h"
#undef NO_IMPORT_ARRAY

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <limits.h>
#include <assert.h>


// #####################################################################################################################
/*
Dtype operations
*/

int check_dtype(int dtype);

#define TYPE_CASE_VALUE_OUT(_TYPE, _type, _po, _val)                                             \
case _TYPE:                                                                                      \
    *(_type *)_po = (_type)_val;                                                                 \
    break

#define TYPE_CASE_VALUE_OUT_F2U(_TYPE, _type, _po, _val)                                         \
case _TYPE:                                                                                      \
    *(_type *)_po = (_val) > -1. ? (_type)(_val) : -(_type)(-_val);                              \
    break

#define TYPE_CASE_GET_VALUE_DOUBLE(_TYPE, _type, _pi, _out)                                      \
case _TYPE:                                                                                      \
{                                                                                                \
    _out = (double)(*((_type *)_pi));                                                            \
}                                                                                                \
break

#define TYPE_CASE_GET_VALUE_BOOL(_TYPE, _type, _pi, _out)                                        \
case _TYPE:                                                                                      \
{                                                                                                \
    _out = *(_type *)_pi ? NPY_TRUE : NPY_FALSE;                                                 \
}                                                                                                \
break

#define GET_VALUE_AS_DOUBLE(dtype, pointer, value)                                               \
{                                                                                                \
    switch (dtype) {                                                                             \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_BOOL, npy_bool, pointer, value);                          \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_UBYTE, npy_ubyte, pointer, value);                        \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_USHORT, npy_ushort, pointer, value);                      \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_UINT, npy_uint, pointer, value);                          \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_ULONG, npy_ulong, pointer, value);                        \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_ULONGLONG, npy_ulonglong, pointer, value);                \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_BYTE, npy_byte, pointer, value);                          \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_SHORT, npy_short, pointer, value);                        \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_INT, npy_int, pointer, value);                            \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_LONG, npy_long, pointer, value);                          \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_LONGLONG, npy_longlong, pointer, value);                  \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_FLOAT, npy_float, pointer, value);                        \
        TYPE_CASE_GET_VALUE_DOUBLE(NPY_DOUBLE, npy_double, pointer, value);                      \
        default:                                                                                 \
            PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");                    \
    }                                                                                            \
}

#define GET_VALUE_AS_BOOL(dtype, pointer, value)                                                 \
{                                                                                                \
    switch (dtype) {                                                                             \
        TYPE_CASE_GET_VALUE_BOOL(NPY_BOOL, npy_bool, pointer, value);                            \
        TYPE_CASE_GET_VALUE_BOOL(NPY_UBYTE, npy_ubyte, pointer, value);                          \
        TYPE_CASE_GET_VALUE_BOOL(NPY_USHORT, npy_ushort, pointer, value);                        \
        TYPE_CASE_GET_VALUE_BOOL(NPY_UINT, npy_uint, pointer, value);                            \
        TYPE_CASE_GET_VALUE_BOOL(NPY_ULONG, npy_ulong, pointer, value);                          \
        TYPE_CASE_GET_VALUE_BOOL(NPY_ULONGLONG, npy_ulonglong, pointer, value);                  \
        TYPE_CASE_GET_VALUE_BOOL(NPY_BYTE, npy_byte, pointer, value);                            \
        TYPE_CASE_GET_VALUE_BOOL(NPY_SHORT, npy_short, pointer, value);                          \
        TYPE_CASE_GET_VALUE_BOOL(NPY_INT, npy_int, pointer, value);                              \
        TYPE_CASE_GET_VALUE_BOOL(NPY_LONG, npy_long, pointer, value);                            \
        TYPE_CASE_GET_VALUE_BOOL(NPY_LONGLONG, npy_longlong, pointer, value);                    \
        TYPE_CASE_GET_VALUE_BOOL(NPY_FLOAT, npy_float, pointer, value);                          \
        TYPE_CASE_GET_VALUE_BOOL(NPY_DOUBLE, npy_double, pointer, value);                        \
        default:                                                                                 \
            PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");                    \
    }                                                                                            \
}

#define SET_VALUE_FROM_DOUBLE(dtype, pointer, value)                                             \
{                                                                                                \
    switch (dtype) {                                                                             \
        TYPE_CASE_VALUE_OUT_F2U(NPY_BOOL, npy_bool, pointer, value);                             \
        TYPE_CASE_VALUE_OUT_F2U(NPY_UBYTE, npy_ubyte, pointer, value);                           \
        TYPE_CASE_VALUE_OUT_F2U(NPY_USHORT, npy_ushort, pointer, value);                         \
        TYPE_CASE_VALUE_OUT_F2U(NPY_UINT, npy_uint, pointer, value);                             \
        TYPE_CASE_VALUE_OUT_F2U(NPY_ULONG, npy_ulong, pointer, value);                           \
        TYPE_CASE_VALUE_OUT_F2U(NPY_ULONGLONG, npy_ulonglong, pointer, value);                   \
        TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, pointer, value);                                 \
        TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, pointer, value);                               \
        TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, pointer, value);                                   \
        TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, pointer, value);                                 \
        TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, pointer, value);                         \
        TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, pointer, value);                               \
        TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, pointer, value);                             \
        default:                                                                                 \
            PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");                   \
    }                                                                                            \
}

// #####################################################################################################################

npy_intp RAVEL_INDEX(npy_intp *index, npy_intp *array_shape, npy_intp nd_m1);
npy_intp UNRAVEL_INDEX(npy_intp index, npy_intp *array_shape, npy_intp nd_m1);

// #####################################################################################################################

typedef struct {
    int nd_m1;
    npy_intp dims_m1[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp backstrides[NPY_MAXDIMS];
} Base_Iterator;

int INIT_Base_Iterator(PyArrayObject *array, Base_Iterator *iterator);

#define BASE_ITERATOR_NEXT(iterator, pointer)                                                 \
{                                                                                             \
    int _ii;                                                                                  \
    for(_ii = (iterator).nd_m1; _ii >= 0; _ii--)                                              \
        if ((iterator).coordinates[_ii] < (iterator).dims_m1[_ii]) {                          \
            (iterator).coordinates[_ii]++;                                                    \
            pointer += (iterator).strides[_ii];                                               \
            break;                                                                            \
        } else {                                                                              \
            (iterator).coordinates[_ii] = 0;                                                  \
            pointer -= (iterator).backstrides[_ii];                                           \
        }                                                                                     \
}

#define BASE_ITERATOR_NEXT2(iterator1, pointer1, iterator2, pointer2)                         \
{                                                                                             \
    int _ii;                                                                                  \
    for(_ii = (iterator1).nd_m1; _ii >= 0; _ii--)                                             \
        if ((iterator1).coordinates[_ii] < (iterator1).dims_m1[_ii]) {                        \
            (iterator1).coordinates[_ii]++;                                                   \
            pointer1 += (iterator1).strides[_ii];                                             \
            pointer2 += (iterator2).strides[_ii];                                             \
            break;                                                                            \
        } else {                                                                              \
            (iterator1).coordinates[_ii] = 0;                                                 \
            pointer1 -= (iterator1).backstrides[_ii];                                         \
            pointer2 -= (iterator2).backstrides[_ii];                                         \
        }                                                                                     \
}

#define BASE_ITERATOR_NEXT3(iterator1, pointer1, iterator2, pointer2, iterator3, pointer3)    \
{                                                                                             \
    int _ii;                                                                                  \
    for(_ii = (iterator1).nd_m1; _ii >= 0; _ii--)                                             \
        if ((iterator1).coordinates[_ii] < (iterator1).dims_m1[_ii]) {                        \
            (iterator1).coordinates[_ii]++;                                                   \
            pointer1 += (iterator1).strides[_ii];                                             \
            pointer2 += (iterator2).strides[_ii];                                             \
            pointer3 += (iterator3).strides[_ii];                                             \
            break;                                                                            \
        } else {                                                                              \
            (iterator1).coordinates[_ii] = 0;                                                 \
            pointer1 -= (iterator1).backstrides[_ii];                                         \
            pointer2 -= (iterator2).backstrides[_ii];                                         \
            pointer3 -= (iterator3).backstrides[_ii];                                         \
        }                                                                                     \
}

#define BASE_ITERATOR_RESET(iterator)                                                         \
{                                                                                             \
    int _ii;                                                                                  \
    for (_ii = 0; _ii <= (iterator).nd_m1; _ii++) {                                           \
        (iterator).coordinates[_ii] = 0;                                                      \
    }                                                                                         \
}

#define BASE_ITERATOR_GOTO(iterator, destination, base, pointer)                              \
{                                                                                             \
    int _ii;                                                                                  \
    pointer = base;                                                                           \
    for(_ii = (iterator).nd_m1; _ii >= 0; _ii--) {                                            \
        pointer += destination[_ii] * (iterator).strides[_ii];                                \
        (iterator).coordinates[_ii] = destination[_ii];                                       \
    }                                                                                         \
}

// #####################################################################################################################

int INIT_FOOTPRINT(PyArrayObject *kernel, npy_bool **footprint, int *footprint_size);

int COPY_DATA_TO_DOUBLE(PyArrayObject *array, double **line, npy_bool *footprint);

int INIT_OFFSETS(PyArrayObject *array,
                 npy_intp *kernel_shape,
                 npy_intp *kernel_origins,
                 npy_bool *footprint,
                 npy_intp **offsets);

int INIT_OFFSETS_AS_COORDINATES(npy_intp nd,
                                npy_intp *kernel_shape,
                                npy_intp *kernel_origins,
                                npy_bool *footprint,
                                npy_intp **offsets);

int INIT_OFFSETS_WITH_BORDERS(PyArrayObject *array,
                              npy_intp *kernel_shape,
                              npy_intp *kernel_origins,
                              npy_bool *footprint,
                              npy_intp **offsets,
                              npy_bool **borders_lookup);

int INIT_OFFSETS_ARRAY(PyArrayObject *array,
                       npy_intp *kernel_shape,
                       npy_intp *kernel_origins,
                       npy_bool *footprint,
                       npy_intp **offsets,
                       npy_intp *offsets_flag,
                       npy_intp *offsets_stride) ;

// #####################################################################################################################

typedef struct {
    npy_intp strides;
    npy_intp ptr;
    npy_intp bound;
} Neighborhood_Iterator;

int INIT_Neighborhood_Iterator(npy_intp *neighborhood_size, npy_intp *array_size, Neighborhood_Iterator *iterator);

#define NEIGHBORHOOD_ITERATOR_NEXT(iterator, pointer)                                           \
{                                                                                               \
    if ((iterator).ptr < (iterator).bound) {                                                    \
        pointer += (iterator).strides;                                                          \
        (iterator).ptr++;                                                                       \
    }                                                                                           \
}

//#define NEIGHBORHOOD_ITERATOR_RESET(iterator, pointer)                         \
//{                         \
//    pointer -= (iterator).strides * (iterator).ptr;                         \
//    (iterator).ptr = 0;                         \
//}
//
//#define NEIGHBORHOOD_ITERATOR_GOTO(iterator, pointer, index)                         \
//{                         \
//    if (index != (iterator).ptr && index < (iterator).bound) {                         \
//        pointer += (iterator).strides * (index - (iterator).ptr);                         \
//        (iterator).ptr = index;                         \
//    }                         \
//}

// #####################################################################################################################

#endif
























