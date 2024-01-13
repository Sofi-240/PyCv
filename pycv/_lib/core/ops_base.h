
// #####################################################################################################################

#define TYPE_CASE_SET_VALUE_F2U(_NUM_TYPE, _dtype, _pointer, _val)                                    \
case _NUM_TYPE:                                                                                       \
    *(_dtype *)_pointer = (_val) > -1. ? (_dtype)(_val) : -(_dtype)(-_val);                           \
    break

#define TYPE_CASE_SET_VALUE(_NUM_TYPE, _dtype, _pointer, _val)                                        \
case _NUM_TYPE:                                                                                       \
    *(_dtype *)_pointer = (_dtype)_val;                                                               \
    break

#define TYPE_CASE_GET_VALUE_AS(_NUM_TYPE, _dtype, _dtype_as, _pointer, _out)                          \
case _NUM_TYPE:                                                                                       \
    _out = (_dtype_as)(*((_dtype *)_pointer));                                                        \
    break

int valid_dtype(int dtype_num);


#define SET_VALUE_TO(_NUM_TYPE, _pointer, _val)                                                      \
{                                                                                                    \
    int _safe_cast = sizeof(_val) == sizeof(double) ? 1 : 0;                                         \
    if (_safe_cast) {                                                                                \
        switch (_NUM_TYPE) {                                                                         \
            TYPE_CASE_VALUE_OUT_F2U(NPY_BOOL, npy_bool, _pointer, _val);                             \
            TYPE_CASE_VALUE_OUT_F2U(NPY_UBYTE, npy_ubyte, _pointer, _val);                           \
            TYPE_CASE_VALUE_OUT_F2U(NPY_USHORT, npy_ushort, _pointer, _val);                         \
            TYPE_CASE_VALUE_OUT_F2U(NPY_UINT, npy_uint, _pointer, _val);                             \
            TYPE_CASE_VALUE_OUT_F2U(NPY_ULONG, npy_ulong, _pointer, _val);                           \
            TYPE_CASE_VALUE_OUT_F2U(NPY_ULONGLONG, npy_ulonglong, _pointer, _val);                   \
            TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, _pointer, _val);                                 \
            TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, _pointer, _val);                               \
            TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, _pointer, _val);                                   \
            TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, _pointer, _val);                                 \
            TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, _pointer, _val);                         \
            TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, _pointer, _val);                               \
            TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, _pointer, _val);                             \
            default:                                                                                 \
                PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");                   \
        }                                                                                            \
    } else {                                                                                         \
        switch (_NUM_TYPE) {                                                                         \
            TYPE_CASE_VALUE_OUT(NPY_BOOL, npy_bool, _pointer, _val);                                 \
            TYPE_CASE_VALUE_OUT(NPY_UBYTE, npy_ubyte, _pointer, _val);                               \
            TYPE_CASE_VALUE_OUT(NPY_USHORT, npy_ushort, _pointer, _val);                             \
            TYPE_CASE_VALUE_OUT(NPY_UINT, npy_uint, _pointer, _val);                                 \
            TYPE_CASE_VALUE_OUT(NPY_ULONG, npy_ulong, _pointer, _val);                               \
            TYPE_CASE_VALUE_OUT(NPY_ULONGLONG, npy_ulonglong, _pointer, _val);                       \
            TYPE_CASE_VALUE_OUT(NPY_BYTE, npy_byte, _pointer, _val);                                 \
            TYPE_CASE_VALUE_OUT(NPY_SHORT, npy_short, _pointer, _val);                               \
            TYPE_CASE_VALUE_OUT(NPY_INT, npy_int, _pointer, _val);                                   \
            TYPE_CASE_VALUE_OUT(NPY_LONG, npy_long, _pointer, _val);                                 \
            TYPE_CASE_VALUE_OUT(NPY_LONGLONG, npy_longlong, _pointer, _val);                         \
            TYPE_CASE_VALUE_OUT(NPY_FLOAT, npy_float, _pointer, _val);                               \
            TYPE_CASE_VALUE_OUT(NPY_DOUBLE, npy_double, _pointer, _val);                             \
            default:                                                                                 \
                PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");                   \
        }                                                                                            \
    }                                                                                                \
}

#define GET_VALUE_AS(_NUM_TYPE, _dtype, _pointer, _val)                                              \
{                                                                                                    \
    switch (dtype) {                                                                                 \
        TYPE_CASE_GET_VALUE_AS(NPY_BOOL, npy_bool, _dtype, pointer, value);                          \
        TYPE_CASE_GET_VALUE_AS(NPY_UBYTE, npy_ubyte, _dtype, pointer, value);                        \
        TYPE_CASE_GET_VALUE_AS(NPY_USHORT, npy_ushort, _dtype, pointer, value);                      \
        TYPE_CASE_GET_VALUE_AS(NPY_UINT, npy_uint, _dtype, pointer, value);                          \
        TYPE_CASE_GET_VALUE_AS(NPY_ULONG, npy_ulong, _dtype, pointer, value);                        \
        TYPE_CASE_GET_VALUE_AS(NPY_ULONGLONG, npy_ulonglong, _dtype, pointer, value);                \
        TYPE_CASE_GET_VALUE_AS(NPY_BYTE, npy_byte, _dtype, pointer, value);                          \
        TYPE_CASE_GET_VALUE_AS(NPY_SHORT, npy_short, _dtype, pointer, value);                        \
        TYPE_CASE_GET_VALUE_AS(NPY_INT, npy_int, _dtype, pointer, value);                            \
        TYPE_CASE_GET_VALUE_AS(NPY_LONG, npy_long, _dtype, pointer, value);                          \
        TYPE_CASE_GET_VALUE_AS(NPY_LONGLONG, npy_longlong, _dtype, pointer, value);                  \
        TYPE_CASE_GET_VALUE_AS(NPY_FLOAT, npy_float, _dtype, pointer, value);                        \
        TYPE_CASE_GET_VALUE_AS(NPY_DOUBLE, npy_double, _dtype, pointer, value);                      \
        default:                                                                                     \
            PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");                        \
    }                                                                                                \
}

// #####################################################################################################################

typedef struct {
    int nd_m1;
    npy_intp dims_m1[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp backstrides[NPY_MAXDIMS];
} ArrayIter;

void ArrayIterInit(PyArrayObject *array, ArrayIter *iterator);

#define ARRAY_ITER_NEXT_NO_POINTER(_iterator)                                                   \
{                                                                                               \
    int _ii;                                                                                    \
    for(_ii = (_iterator).nd_m1; _ii >= 0; _ii--)                                               \
        if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                          \
            (_iterator).coordinates[_ii]++;                                                     \
            break;                                                                              \
        } else {                                                                                \
            (_iterator).coordinates[_ii] = 0;                                                   \
        }                                                                                       \
}

#define ARRAY_ITER_NEXT(_iterator, _pointer)                                                    \
{                                                                                               \
    int _ii;                                                                                    \
    for(_ii = (_iterator).nd_m1; _ii >= 0; _ii--)                                               \
        if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                          \
            (_iterator).coordinates[_ii]++;                                                     \
            _pointer += (_iterator).strides[_ii];                                               \
            break;                                                                              \
        } else {                                                                                \
            (_iterator).coordinates[_ii] = 0;                                                   \
            _pointer -= (_iterator).backstrides[_ii];                                           \
        }                                                                                       \
}

#define ARRAY_ITER_NEXT2(_iterator1, _pointer1, _iterator2, _pointer2)                         \
{                                                                                              \
    int _ii;                                                                                   \
    for(_ii = (_iterator1).nd_m1; _ii >= 0; _ii--)                                             \
        if ((_iterator1).coordinates[_ii] < (_iterator1).dims_m1[_ii]) {                       \
            (_iterator1).coordinates[_ii]++;                                                   \
            _pointer1 += (_iterator1).strides[_ii];                                            \
            _pointer2 += (_iterator2).strides[_ii];                                            \
            break;                                                                             \
        } else {                                                                               \
            (_iterator1).coordinates[_ii] = 0;                                                 \
            _pointer1 -= (_iterator1).backstrides[_ii];                                        \
            _pointer2 -= (_iterator2).backstrides[_ii];                                        \
        }                                                                                      \
}

#define ARRAY_ITER_NEXT3(_iterator1, _pointer1, _iterator2, _pointer2, _iterator3, _pointer3)    \
{                                                                                                \
    int _ii;                                                                                     \
    for(_ii = (_iterator1).nd_m1; _ii >= 0; _ii--)                                               \
        if ((_iterator1).coordinates[_ii] < (_iterator1).dims_m1[_ii]) {                         \
            (_iterator1).coordinates[_ii]++;                                                     \
            _pointer1 += (_iterator1).strides[_ii];                                              \
            _pointer2 += (_iterator2).strides[_ii];                                              \
            _pointer3 += (_iterator3).strides[_ii];                                              \
            break;                                                                               \
        } else {                                                                                 \
            (_iterator1).coordinates[_ii] = 0;                                                   \
            _pointer1 -= (_iterator1).backstrides[_ii];                                          \
            _pointer2 -= (_iterator2).backstrides[_ii];                                          \
            _pointer3 -= (_iterator3).backstrides[_ii];                                          \
        }                                                                                        \
}

#define ARRAY_ITER_RESET(_iterator)                                                           \
{                                                                                             \
    int _ii;                                                                                  \
    for (_ii = 0; _ii <= (_iterator).nd_m1; _ii++) {                                          \
        (_iterator).coordinates[_ii] = 0;                                                     \
    }                                                                                         \
}

#define ARRAY_ITER_GOTO(_iterator, _coordinates, _base, _pointer)                              \
{                                                                                              \
    int _ii;                                                                                   \
    _pointer = _base;                                                                          \
    for(_ii = (_iterator).nd_m1; _ii >= 0; _ii--) {                                            \
        _pointer += _coordinates[_ii] * (_iterator).strides[_ii];                              \
        (_iterator).coordinates[_ii] = _coordinates[_ii];                                      \
    }                                                                                          \
}

typedef struct {
    int nd_m1;
    npy_intp dims_m1[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
} CoordinatesIter;

void CoordinatesIterInit(npy_intp nd, npy_intp *shape, CoordinatesIter *iterator);

#define COORDINATES_ITER_NEXT(_iterator)                                                        \
{                                                                                               \
    int _ii;                                                                                    \
    for(_ii = (_iterator).nd_m1; _ii >= 0; _ii--)                                               \
        if ((_iterator).coordinates[_ii] < (_iterator).dims_m1[_ii]) {                          \
            (_iterator).coordinates[_ii]++;                                                     \
            break;                                                                              \
        } else {                                                                                \
            (_iterator).coordinates[_ii] = 0;                                                   \
        }                                                                                       \
}

#define COORDINATES_ITER_RESET(_iterator)                                                     \
{                                                                                             \
    int _ii;                                                                                  \
    for (_ii = 0; _ii <= (_iterator).nd_m1; _ii++) {                                          \
        (_iterator).coordinates[_ii] = 0;                                                     \
    }                                                                                         \
}


// #####################################################################################################################

int array_to_footprint(PyArrayObject *array, npy_bool **footprint, int *non_zeros);

int copy_data_as_double(PyArrayObject *array, double **line, npy_bool *footprint);

// #####################################################################################################################

int init_offsets_ravel(PyArrayObject *array,
                       npy_intp *kernel_shape,
                       npy_intp *kernel_origins,
                       npy_bool *footprint,
                       npy_intp **offsets);

int init_offsets_coordinates(npy_intp nd,
                             npy_intp *kernel_shape,
                             npy_intp *kernel_origins,
                             npy_bool *footprint,
                             npy_intp **offsets);

int init_borders_lut(npy_intp nd,
                     npy_intp *array_shape,
                     npy_intp *kernel_shape,
                     npy_intp *kernel_origins,
                     npy_bool **borders_lookup);

int init_offsets_lut(PyArrayObject *array,
                     npy_intp *kernel_shape,
                     npy_intp *kernel_origins,
                     npy_bool *footprint,
                     npy_intp **offsets_lookup,
                     npy_intp *offsets_flag,
                     npy_intp *offsets_stride);

// #####################################################################################################################