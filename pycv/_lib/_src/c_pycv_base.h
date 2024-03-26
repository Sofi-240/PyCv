#ifndef C_PYCV_BASE_H
#define C_PYCV_BASE_H

#define NO_IMPORT_ARRAY
#include "c_pycv.h"
#undef NO_IMPORT_ARRAY

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// #####################################################################################################################

int PYCV_valid_dtype(int num_type);

#define PYCV_CASE_SET_VALUE_2U_SAFE(_NTYPE, _dtype, _pointer, _val)                                                    \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    *(_dtype *)_pointer = _val > -1 ? (_dtype)_val : -(_dtype)(-_val);                                                 \
}                                                                                                                      \
break


#define PYCV_CASE_SET_VALUE_F2U(_NTYPE, _dtype, _pointer, _val)                                                        \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    _val = _val > 0 ? _val + 0.5 : 0;                                                                                  \
    _val = _val > NPY_MAX_##_NTYPE ? NPY_MAX_##_NTYPE : _val;                                                          \
    *(_dtype *)_pointer = (_dtype)_val;                                                                                \
}                                                                                                                      \
break


#define PYCV_CASE_SET_VALUE_F2I(_NTYPE, _dtype, _pointer, _val)                                                        \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    _val = _val > 0 ? _val + 0.5 : _val - 0.5;                                                                         \
    _val = _val > NPY_MAX_##_NTYPE ? NPY_MAX_##_NTYPE : _val;                                                          \
    _val = _val < NPY_MIN_##_NTYPE ? NPY_MIN_##_NTYPE : _val;                                                          \
    *(_dtype *)_pointer = (_dtype)_val;                                                                                \
}                                                                                                                      \
break


#define PYCV_CASE_SET_VALUE(_NTYPE, _dtype, _pointer, _val)                                                            \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    *(_dtype *)_pointer = (_dtype)_val;                                                                                \
}                                                                                                                      \
break

#define PYCV_SET_VALUE_F2A(_NTYPE, _pointer, _val)                                                                     \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_CASE_SET_VALUE(BOOL, npy_bool, _pointer, _val);                                                           \
        PYCV_CASE_SET_VALUE_F2U(UBYTE, npy_ubyte, _pointer, _val);                                                     \
        PYCV_CASE_SET_VALUE_F2U(USHORT, npy_ushort, _pointer, _val);                                                   \
        PYCV_CASE_SET_VALUE_F2U(UINT, npy_uint, _pointer, _val);                                                       \
        PYCV_CASE_SET_VALUE_F2U(ULONG, npy_ulong, _pointer, _val);                                                     \
        PYCV_CASE_SET_VALUE_F2U(ULONGLONG, npy_ulonglong, _pointer, _val);                                             \
        PYCV_CASE_SET_VALUE_F2I(BYTE, npy_byte, _pointer, _val);                                                       \
        PYCV_CASE_SET_VALUE_F2I(SHORT, npy_short, _pointer, _val);                                                     \
        PYCV_CASE_SET_VALUE_F2I(INT, npy_int, _pointer, _val);                                                         \
        PYCV_CASE_SET_VALUE_F2I(LONG, npy_long, _pointer, _val);                                                       \
        PYCV_CASE_SET_VALUE_F2I(LONGLONG, npy_longlong, _pointer, _val);                                               \
        PYCV_CASE_SET_VALUE(FLOAT, npy_float, _pointer, _val);                                                         \
        PYCV_CASE_SET_VALUE(DOUBLE, npy_double, _pointer, _val);                                                       \
    }                                                                                                                  \
}


#define PYCV_SET_VALUE(_NTYPE, _pointer, _val)                                                                         \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_CASE_SET_VALUE(BOOL, npy_bool, _pointer, _val);                                                           \
        PYCV_CASE_SET_VALUE_2U_SAFE(UBYTE, npy_ubyte, _pointer, _val);                                                 \
        PYCV_CASE_SET_VALUE_2U_SAFE(USHORT, npy_ushort, _pointer, _val);                                               \
        PYCV_CASE_SET_VALUE_2U_SAFE(UINT, npy_uint, _pointer, _val);                                                   \
        PYCV_CASE_SET_VALUE_2U_SAFE(ULONG, npy_ulong, _pointer, _val);                                                 \
        PYCV_CASE_SET_VALUE_2U_SAFE(ULONGLONG, npy_ulonglong, _pointer, _val);                                         \
        PYCV_CASE_SET_VALUE(BYTE, npy_byte, _pointer, _val);                                                           \
        PYCV_CASE_SET_VALUE(SHORT, npy_short, _pointer, _val);                                                         \
        PYCV_CASE_SET_VALUE(INT, npy_int, _pointer, _val);                                                             \
        PYCV_CASE_SET_VALUE(LONG, npy_long, _pointer, _val);                                                           \
        PYCV_CASE_SET_VALUE(LONGLONG, npy_longlong, _pointer, _val);                                                   \
        PYCV_CASE_SET_VALUE(FLOAT, npy_float, _pointer, _val);                                                         \
        PYCV_CASE_SET_VALUE(DOUBLE, npy_double, _pointer, _val);                                                       \
    }                                                                                                                  \
}


#define PYCV_CASE_GET_VALUE_AS(_NTYPE, _dtype, _dtype_as, _pointer, _out)                                              \
case _NTYPE:                                                                                                           \
{                                                                                                                      \
    _out = (_dtype_as)(*((_dtype *)_pointer));                                                                         \
}                                                                                                                      \
break


#define PYCV_GET_VALUE(_NTYPE, _dtype_as, _pointer, _out)                                                              \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_CASE_GET_VALUE_AS(NPY_BOOL, npy_bool, _dtype_as, _pointer, _out);                                         \
        PYCV_CASE_GET_VALUE_AS(NPY_UBYTE, npy_ubyte, _dtype_as, _pointer, _out);                                       \
        PYCV_CASE_GET_VALUE_AS(NPY_USHORT, npy_ushort, _dtype_as, _pointer, _out);                                     \
        PYCV_CASE_GET_VALUE_AS(NPY_UINT, npy_uint, _dtype_as, _pointer, _out);                                         \
        PYCV_CASE_GET_VALUE_AS(NPY_ULONG, npy_ulong, _dtype_as, _pointer, _out);                                       \
        PYCV_CASE_GET_VALUE_AS(NPY_ULONGLONG, npy_ulonglong, _dtype_as, _pointer, _out);                               \
        PYCV_CASE_GET_VALUE_AS(NPY_BYTE, npy_byte, _dtype_as, _pointer, _out);                                         \
        PYCV_CASE_GET_VALUE_AS(NPY_SHORT, npy_short, _dtype_as, _pointer, _out);                                       \
        PYCV_CASE_GET_VALUE_AS(NPY_INT, npy_int, _dtype_as, _pointer, _out);                                           \
        PYCV_CASE_GET_VALUE_AS(NPY_LONG, npy_long, _dtype_as, _pointer, _out);                                         \
        PYCV_CASE_GET_VALUE_AS(NPY_LONGLONG, npy_longlong, _dtype_as, _pointer, _out);                                 \
        PYCV_CASE_GET_VALUE_AS(NPY_FLOAT, npy_float, _dtype_as, _pointer, _out);                                       \
        PYCV_CASE_GET_VALUE_AS(NPY_DOUBLE, npy_double, _dtype_as, _pointer, _out);                                     \
    }                                                                                                                  \
}

// #####################################################################################################################

typedef struct {
    npy_intp nd_m1;
    npy_intp dims_m1[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp strides_back[NPY_MAXDIMS];
    int numtype;
} PYCV_ArrayIterator;


void PYCV_ArrayIteratorInit(PyArrayObject *array, PYCV_ArrayIterator *iterator);


#define PYCV_ARRAY_ITERATOR_NEXT(_iterator, _pointer)                                                                  \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator).nd_m1; _ii >= 0; _ii--) {                                                                   \
        if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                                                 \
            (_iterator).coordinates[_ii]++;                                                                            \
            _pointer += (_iterator).strides[_ii];                                                                      \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator).coordinates[_ii] = 0;                                                                          \
            _pointer -= (_iterator).strides_back[_ii];                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_ARRAY_ITERATOR_NEXT2(_iterator1, _pointer1, _iterator2, _pointer2)                                        \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator1).nd_m1; _ii >= 0; _ii--) {                                                                  \
        if ((_iterator1).coordinates[_ii] < (_iterator1).dims_m1[_ii]) {                                               \
            (_iterator1).coordinates[_ii]++;                                                                           \
            _pointer1 += (_iterator1).strides[_ii];                                                                    \
            (_iterator2).coordinates[_ii]++;                                                                           \
            _pointer2 += (_iterator2).strides[_ii];                                                                    \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator1).coordinates[_ii] = 0;                                                                         \
            _pointer1 -= (_iterator1).strides_back[_ii];                                                               \
            (_iterator2).coordinates[_ii] = 0;                                                                         \
            _pointer2 -= (_iterator2).strides_back[_ii];                                                               \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_ARRAY_ITERATOR_NEXT3(_iterator1, _pointer1, _iterator2, _pointer2, _iterator3, _pointer3)                 \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator1).nd_m1; _ii >= 0; _ii--) {                                                                  \
        if ((_iterator1).coordinates[_ii] < (_iterator1).dims_m1[_ii]) {                                               \
            (_iterator1).coordinates[_ii]++;                                                                           \
            _pointer1 += (_iterator1).strides[_ii];                                                                    \
            (_iterator2).coordinates[_ii]++;                                                                           \
            _pointer2 += (_iterator2).strides[_ii];                                                                    \
            (_iterator3).coordinates[_ii]++;                                                                           \
            _pointer3 += (_iterator3).strides[_ii];                                                                    \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator1).coordinates[_ii] = 0;                                                                         \
            _pointer1 -= (_iterator1).strides_back[_ii];                                                               \
            (_iterator2).coordinates[_ii] = 0;                                                                         \
            _pointer2 -= (_iterator2).strides_back[_ii];                                                               \
            (_iterator3).coordinates[_ii] = 0;                                                                         \
            _pointer3 -= (_iterator3).strides_back[_ii];                                                               \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_ARRAY_ITERATOR_RESET(_iterator)                                                                           \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        (_iterator).coordinates[_ii] = 0;                                                                              \
    }                                                                                                                  \
}

#define PYCV_ARRAY_ITERATOR_GOTO(_iterator, _pointer_base, _pointer, _coordinates)                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _pointer = _pointer_base;                                                                                          \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        _pointer += _coordinates[_ii] * (_iterator).strides[_ii];                                                      \
        (_iterator).coordinates[_ii] = _coordinates[_ii];                                                              \
    }                                                                                                                  \
}

#define PYCV_ARRAY_ITERATOR_GOTO_RAVEL(_iterator, _pointer_base, _pointer, _index)                                     \
{                                                                                                                      \
    npy_intp _ii, _ind;                                                                                                \
    _pointer = _pointer_base;                                                                                          \
    _ind = _index;                                                                                                     \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        (_iterator).coordinates[_ii] = _ind / (_iterator).strides[_ii];                                                \
        _pointer += (_iterator).coordinates[_ii] * (_iterator).strides[_ii];                                           \
        _ind -= (_iterator).coordinates[_ii] * (_iterator).strides[_ii];                                               \
    }                                                                                                                  \
}

typedef struct {
    npy_intp nd_m1;
    npy_intp dims_m1[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
} PYCV_CoordinatesIterator;


void PYCV_CoordinatesIteratorInit(npy_intp ndim, npy_intp *dims, PYCV_CoordinatesIterator *iterator);


#define PYCV_COORDINATES_ITERATOR_NEXT(_iterator)                                                                      \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator).nd_m1; _ii >= 0; _ii--) {                                                                   \
        if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                                                 \
            (_iterator).coordinates[_ii]++;                                                                            \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator).coordinates[_ii] = 0;                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_COORDINATES_ITERATOR_RESET(_iterator)                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        (_iterator).coordinates[_ii] = 0;                                                                              \
    }                                                                                                                  \
}

#define PYCV_COORDINATES_ITERATOR_GOTO(_iterator, _coordinates)                                                        \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        (_iterator).coordinates[_ii] = _coordinates[_ii];                                                              \
    }                                                                                                                  \
}

// #####################################################################################################################

/*
    same as numpy pad
                          pad   |  image  |  pad
       pos           : -4-3-2-1 | 0 1 2 3 | 4 5 6 7
    ___________________________________________________
    3. REFLECT       :  3 4 3 2 | 1 2 3 4 | 3 2 1 2
    4. CONSTANT(c=0) :  0 0 0 0 | 1 2 3 4 | 0 0 0 0
    5. SYMMETRIC     :  4 3 2 1 | 1 2 3 4 | 4 3 2 1
    6. WRAP          :  1 2 3 4 | 1 2 3 4 | 1 2 3 4
    7. EDGE          :  1 1 1 1 | 1 2 3 4 | 4 4 4 4
*/

typedef enum {
    PYCV_EXTEND_FLAG = 1,
    PYCV_EXTEND_VALID = 2,
    PYCV_EXTEND_REFLECT = 3,
    PYCV_EXTEND_CONSTANT = 4,
    PYCV_EXTEND_SYMMETRIC = 5,
    PYCV_EXTEND_WRAP = 6,
    PYCV_EXTEND_EDGE = 7,
} PYCV_ExtendBorder;


#define PYCV_RAVEL_COORDINATE(_coordinate, _ndim, _strides, _out)                                                      \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _out = 0;                                                                                                          \
    for (_ii = 0; _ii < _ndim; _ii++) {                                                                                \
        _out += _coordinate[_ii] * _strides[_ii];                                                                      \
    }                                                                                                                  \
}

#define PYCV_FOOTPRINT_NONZERO(_footprint, _size, _nonzero)                                                            \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _nonzero = _size;                                                                                                  \
    if (_footprint) {                                                                                                  \
        for (_ii = 0; _ii < _size; _ii++) {                                                                            \
            if (!_footprint[_ii]) {                                                                                    \
                    _nonzero--;                                                                                        \
                }                                                                                                      \
            }                                                                                                          \
    }                                                                                                                  \
}


npy_intp PYCV_FitCoordinate(npy_intp coordinate, npy_intp dim, npy_intp flag, PYCV_ExtendBorder mode);


int PYCV_InitOffsets(PyArrayObject *array,
                     npy_intp *shape,
                     npy_intp *center,
                     npy_bool *footprint,
                     npy_intp **ravel_offsets,
                     npy_intp **unravel_offsets);

// #####################################################################################################################


typedef struct {
    npy_intp nd_m1;
    npy_intp dims_m1[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp strides_back[NPY_MAXDIMS];
    npy_intp nn_strides[NPY_MAXDIMS];
    npy_intp nn_strides_back[NPY_MAXDIMS];
    npy_intp boundary_low[NPY_MAXDIMS];
    npy_intp boundary_high[NPY_MAXDIMS];
    int numtype;
} NeighborhoodIterator;


void PYCV_NeighborhoodIteratorInit(PyArrayObject *array,
                                   npy_intp *shape,
                                   npy_intp *center,
                                   npy_intp n,
                                   NeighborhoodIterator *iterator);


#define PYCV_NEIGHBORHOOD_ITERATOR_NEXT(_iterator, _pointer, _offsets)                                                 \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator).nd_m1; _ii >= 0; _ii--) {                                                                   \
        if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                                                 \
            if ((_iterator).coordinates[_ii] < (_iterator).boundary_low[_ii] ||                                        \
                    (_iterator).coordinates[_ii] >= (_iterator).boundary_high[_ii]) {                                  \
                _offsets += (_iterator).nn_strides[_ii];                                                               \
            }                                                                                                          \
            (_iterator).coordinates[_ii]++;                                                                            \
            _pointer += (_iterator).strides[_ii];                                                                      \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator).coordinates[_ii] = 0;                                                                          \
            _pointer -= (_iterator).strides_back[_ii];                                                                 \
            _offsets -= (_iterator).nn_strides_back[_ii];                                                              \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_NEXT2(_iterator1, _pointer1, _iterator2, _pointer2, _offsets)                       \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator1).nd_m1; _ii >= 0; _ii--) {                                                                  \
        if ((_iterator1).coordinates[_ii] < (_iterator1).dims_m1[_ii]) {                                               \
            if ((_iterator1).coordinates[_ii] < (_iterator1).boundary_low[_ii] ||                                      \
                    (_iterator1).coordinates[_ii] >= (_iterator1).boundary_high[_ii]) {                                \
                _offsets += (_iterator1).nn_strides[_ii];                                                              \
            }                                                                                                          \
            (_iterator1).coordinates[_ii]++;                                                                           \
            _pointer1 += (_iterator1).strides[_ii];                                                                    \
            (_iterator2).coordinates[_ii]++;                                                                           \
            _pointer2 += (_iterator2).strides[_ii];                                                                    \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator1).coordinates[_ii] = 0;                                                                         \
            _pointer1 -= (_iterator1).strides_back[_ii];                                                               \
            (_iterator2).coordinates[_ii] = 0;                                                                         \
            _pointer2 -= (_iterator2).strides_back[_ii];                                                               \
            _offsets -= (_iterator1).nn_strides_back[_ii];                                                             \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_NEXT3(_iterator1, _pointer1, _iterator2, _pointer2, _iterator3, _pointer3, _offsets)\
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = (_iterator1).nd_m1; _ii >= 0; _ii--) {                                                                  \
        if ((_iterator1).coordinates[_ii] < (_iterator1).dims_m1[_ii]) {                                               \
            if ((_iterator1).coordinates[_ii] < (_iterator1).boundary_low[_ii] ||                                      \
                    (_iterator1).coordinates[_ii] >= (_iterator1).boundary_high[_ii]) {                                \
                _offsets += (_iterator1).nn_strides[_ii];                                                              \
            }                                                                                                          \
            (_iterator1).coordinates[_ii]++;                                                                           \
            _pointer1 += (_iterator1).strides[_ii];                                                                    \
            (_iterator2).coordinates[_ii]++;                                                                           \
            _pointer2 += (_iterator2).strides[_ii];                                                                    \
            (_iterator3).coordinates[_ii]++;                                                                           \
            _pointer3 += (_iterator3).strides[_ii];                                                                    \
            break;                                                                                                     \
        } else {                                                                                                       \
            (_iterator1).coordinates[_ii] = 0;                                                                         \
            _pointer1 -= (_iterator1).strides_back[_ii];                                                               \
            (_iterator2).coordinates[_ii] = 0;                                                                         \
            _pointer2 -= (_iterator2).strides_back[_ii];                                                               \
            (_iterator3).coordinates[_ii] = 0;                                                                         \
            _pointer3 -= (_iterator3).strides_back[_ii];                                                               \
            _offsets -= (_iterator1).nn_strides_back[_ii];                                                             \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_RESET(_iterator)                                                                    \
{                                                                                                                      \
    PYCV_ARRAY_ITERATOR_RESET(_iterator);                                                                              \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_GOTO(_iterator, _pointer_base, _pointer, _offsets_base, _offsets, _coordinates)     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _pointer = _pointer_base;                                                                                          \
    _offsets = _offsets_base;                                                                                          \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        _pointer += _coordinates[_ii] * (_iterator).strides[_ii];                                                      \
        (_iterator).coordinates[_ii] = _coordinates[_ii];                                                              \
        if ((_iterator).coordinates[_ii] < (_iterator).boundary_low[_ii]) {                                            \
            _offsets += (_iterator).nn_strides[_ii] * _coordinates[_ii];                                               \
        } else if ((_iterator).coordinates[_ii] > (_iterator).boundary_high[_ii] &&                                    \
                   (_iterator).boundary_high[_ii] >= (_iterator).boundary_low[_ii]) {                                  \
            _offsets += (_iterator).nn_strides[_ii] *                                                                  \
                        (_coordinates[_ii] + (_iterator).boundary_low[_ii] - (_iterator).boundary_high[_ii]);          \
        } else {                                                                                                       \
            _offsets += (_iterator).nn_strides[_ii] * (_iterator).boundary_low[_ii];                                   \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_GOTO2(_iterator1, _pointer1_base, _pointer1,                                        \
                                         _iterator2, _pointer2_base, _pointer2,                                        \
                                         _offsets_base, _offsets, _coordinates)                                        \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _pointer1 = _pointer1_base;                                                                                        \
    _pointer2 = _pointer2_base;                                                                                        \
    _offsets = _offsets_base;                                                                                          \
    for (_ii = 0; _ii <= (_iterator1.nd_m1); _ii++) {                                                                  \
        _pointer1 += _coordinates[_ii] * (_iterator1).strides[_ii];                                                    \
        (_iterator1).coordinates[_ii] = _coordinates[_ii];                                                             \
        _pointer2 += _coordinates[_ii] * (_iterator2).strides[_ii];                                                    \
        (_iterator2).coordinates[_ii] = _coordinates[_ii];                                                             \
        if ((_iterator1).coordinates[_ii] < (_iterator1).boundary_low[_ii]) {                                          \
            _offsets += (_iterator1).nn_strides[_ii] * _coordinates[_ii];                                              \
        } else if ((_iterator1).coordinates[_ii] > (_iterator1).boundary_high[_ii] &&                                  \
                   (_iterator1).boundary_high[_ii] >= (_iterator1).boundary_low[_ii]) {                                \
            _offsets += (_iterator1).nn_strides[_ii] *                                                                 \
                        (_coordinates[_ii] + (_iterator1).boundary_low[_ii] - (_iterator1).boundary_high[_ii]);        \
        } else {                                                                                                       \
            _offsets += (_iterator1).nn_strides[_ii] * (_iterator1).boundary_low[_ii];                                 \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_GOTO3(_iterator1, _pointer1_base, _pointer1,                                        \
                                         _iterator2, _pointer2_base, _pointer2,                                        \
                                         _iterator3, _pointer3_base, _pointer3,                                        \
                                         _offsets_base, _offsets, _coordinates)                                        \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    _pointer1 = _pointer1_base;                                                                                        \
    _pointer2 = _pointer2_base;                                                                                        \
    _pointer3 = _pointer3_base;                                                                                        \
    _offsets = _offsets_base;                                                                                          \
    for (_ii = 0; _ii <= (_iterator1.nd_m1); _ii++) {                                                                  \
        _pointer1 += _coordinates[_ii] * (_iterator1).strides[_ii];                                                    \
        (_iterator1).coordinates[_ii] = _coordinates[_ii];                                                             \
        _pointer2 += _coordinates[_ii] * (_iterator2).strides[_ii];                                                    \
        (_iterator2).coordinates[_ii] = _coordinates[_ii];                                                             \
        _pointer3 += _coordinates[_ii] * (_iterator3).strides[_ii];                                                    \
        (_iterator3).coordinates[_ii] = _coordinates[_ii];                                                             \
        if ((_iterator1).coordinates[_ii] < (_iterator1).boundary_low[_ii]) {                                          \
            _offsets += (_iterator1).nn_strides[_ii] * _coordinates[_ii];                                              \
        } else if ((_iterator1).coordinates[_ii] > (_iterator1).boundary_high[_ii] &&                                  \
                   (_iterator1).boundary_high[_ii] >= (_iterator1).boundary_low[_ii]) {                                \
            _offsets += (_iterator1).nn_strides[_ii] *                                                                 \
                        (_coordinates[_ii] + (_iterator1).boundary_low[_ii] - (_iterator1).boundary_high[_ii]);        \
        } else {                                                                                                       \
            _offsets += (_iterator1).nn_strides[_ii] * (_iterator1).boundary_low[_ii];                                 \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_GOTO_RAVEL(_iterator, _pointer_base, _pointer, _offsets_base, _offsets, _index)     \
{                                                                                                                      \
    npy_intp _ii, _ind;                                                                                                \
    _pointer = _pointer_base;                                                                                          \
    _offsets = _offsets_base;                                                                                          \
    _ind = _index;                                                                                                     \
    for (_ii = 0; _ii <= (_iterator.nd_m1); _ii++) {                                                                   \
        (_iterator).coordinates[_ii] = _ind / (_iterator).strides[_ii];                                                \
        _pointer += (_iterator).coordinates[_ii] * (_iterator).strides[_ii];                                           \
        _ind -= (_iterator).coordinates[_ii] * (_iterator).strides[_ii];                                               \
        if ((_iterator).coordinates[_ii] < (_iterator).boundary_low[_ii]) {                                            \
            _offsets += (_iterator).nn_strides[_ii] * (_iterator).coordinates[_ii];                                    \
        } else if ((_iterator).coordinates[_ii] > (_iterator).boundary_high[_ii] &&                                    \
                   (_iterator).boundary_high[_ii] >= (_iterator).boundary_low[_ii]) {                                  \
            _offsets += (_iterator).nn_strides[_ii] *                                                                  \
                        ((_iterator).coordinates[_ii] + (_iterator).boundary_low[_ii] - (_iterator).boundary_high[_ii]);\
        } else {                                                                                                       \
            _offsets += (_iterator).nn_strides[_ii] * (_iterator).boundary_low[_ii];                                   \
        }                                                                                                              \
    }                                                                                                                  \
}

#define PYCV_NEIGHBORHOOD_ITERATOR_GOTO2_RAVEL(_iterator1, _pointer1_base, _pointer1,                                  \
                                               _iterator2, _pointer2_base, _pointer2,                                  \
                                               _offsets_base, _offsets, _index)                                        \
{                                                                                                                      \
    npy_intp _ii, _ind = _index;                                                                                       \
    _pointer1 = _pointer1_base;                                                                                        \
    _pointer2 = _pointer2_base;                                                                                        \
    _offsets = _offsets_base;                                                                                          \
    for (_ii = 0; _ii <= (_iterator1.nd_m1); _ii++) {                                                                  \
        (_iterator1).coordinates[_ii] = _ind / (_iterator1).strides[_ii];                                              \
        _ind -= (_iterator1).coordinates[_ii] * (_iterator1).strides[_ii];                                             \
        _pointer1 += (_iterator1).coordinates[_ii] * (_iterator1).strides[_ii];                                        \
        _pointer2 += (_iterator1).coordinates[_ii] * (_iterator2).strides[_ii];                                        \
        (_iterator2).coordinates[_ii] = (_iterator1).coordinates[_ii];                                                 \
        if ((_iterator1).coordinates[_ii] < (_iterator1).boundary_low[_ii]) {                                          \
            _offsets += (_iterator1).nn_strides[_ii] * (_iterator1).coordinates[_ii];                                  \
        } else if ((_iterator1).coordinates[_ii] > (_iterator1).boundary_high[_ii] &&                                  \
                   (_iterator1).boundary_high[_ii] >= (_iterator1).boundary_low[_ii]) {                                \
            _offsets += (_iterator1).nn_strides[_ii] *                                                                 \
                        ((_iterator1).coordinates[_ii] + (_iterator1).boundary_low[_ii] - (_iterator1).boundary_high[_ii]);\
        } else {                                                                                                       \
            _offsets += (_iterator1).nn_strides[_ii] * (_iterator1).boundary_low[_ii];                                 \
        }                                                                                                              \
    }                                                                                                                  \
}


// #####################################################################################################################

int PYCV_InitNeighborhoodOffsets(PyArrayObject *array,
                                 npy_intp *shape,
                                 npy_intp *center,
                                 npy_bool *footprint,
                                 npy_intp **ravel_offsets,
                                 npy_intp **unravel_offsets,
                                 npy_intp *flag,
                                 PYCV_ExtendBorder mode);


// #####################################################################################################################

int PYCV_AllocateToFootprint(PyArrayObject *array, npy_bool **footprint, npy_intp *nonzero, int flip);

int PYCV_AllocateKernelFlip(PyArrayObject *kernel, npy_bool **footprint, npy_double **h);

int PYCV_DefaultFootprint(npy_intp ndim,
                          npy_intp connectivity,
                          npy_bool **footprint,
                          npy_intp *nonzero,
                          unsigned int one_side);

// #####################################################################################################################

typedef struct {
    npy_intp ndim;
    npy_intp max_size;
    npy_intp coordinates_size;
    npy_intp **coordinates;
} PYCV_CoordinatesList;

int PYCV_CoordinatesListInit(npy_intp ndim, npy_intp max_size, PYCV_CoordinatesList *object);

#define PYCV_COORDINATES_LIST_APPEND(_object, _coordinate)                                                             \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    if ((_object).max_size > (_object).coordinates_size) {                                                             \
        for (_ii = 0; _ii < (_object).ndim; _ii++) {                                                                   \
            (_object).coordinates[(_object).coordinates_size][_ii] = _coordinate[_ii];                                 \
        }                                                                                                              \
    }                                                                                                                  \
    (_object).coordinates_size++;                                                                                      \
}

#define PYCV_COORDINATES_LIST_SET(_object, _coordinate, _index)                                                        \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    if ((_object).coordinates_size > _index) {                                                                         \
        for (_ii = 0; _ii < (_object).ndim; _ii++) {                                                                   \
            (_object).coordinates[_index][_ii] = _coordinate[_ii];                                                     \
        }                                                                                                              \
    }                                                                                                                  \
}

int PYCV_CoordinatesListFree(PYCV_CoordinatesList *object);


// #####################################################################################################################

int PYCV_CopyArrayTo(PyArrayObject *to, PyArrayObject *from);

// #####################################################################################################################


#endif