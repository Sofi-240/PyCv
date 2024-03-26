

#define CASE_FILL_IF_ZERO(_NTYPE, _dtype, _pi, _pos, _of, _flag, _dfs, _n)                                             \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    if (_of != _flag && !(double)(*(_dtype *)(_pi + _of))) {                                                           \
        *(_dtype *)(_pi + _of) = 1;                                                                                    \
        *(_dfs + _n) = _pos + _of;                                                                                     \
        _n++;                                                                                                          \
    }                                                                                                                  \
}                                                                                                                      \
return

#define FILL_IF_ZERO(_NTYPE, _pi, _pos, _of, _flag, _dfs, _n)                                                          \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_FILL_IF_ZERO(BOOL, npy_bool, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(UBYTE, npy_ubyte, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(USHORT, npy_ushort, _pi, _pos, _of, _flag, _dfs, _n);                                        \
        CASE_FILL_IF_ZERO(UINT, npy_uint, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(ULONG, npy_ulong, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(ULONGLONG, npy_ulonglong, _pi, _pos, _of, _flag, _dfs, _n);                                  \
        CASE_FILL_IF_ZERO(BYTE, npy_byte, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(SHORT, npy_short, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(INT, npy_int, _pi, _pos, _of, _flag, _dfs, _n);                                              \
        CASE_FILL_IF_ZERO(LONG, npy_long, _pi, _pos, _of, _flag, _dfs, _n);                                            \
        CASE_FILL_IF_ZERO(LONGLONG, npy_longlong, _pi, _pos, _of, _flag, _dfs, _n);                                    \
        CASE_FILL_IF_ZERO(FLOAT, npy_float, _pi, _pos, _of, _flag, _dfs, _n);                                          \
        CASE_FILL_IF_ZERO(DOUBLE, npy_double, _pi, _pos, _of, _flag, _dfs, _n);                                        \
    }                                                                                                                  \
}

int PYCV_binary_region_fill(PyArrayObject *output,
                            npy_intp *seed_point,
                            PyArrayObject *strel,
                            npy_intp *center,
                            npy_double fill)
{
    PYCV_ArrayIterator iter_o;
    char *po = NULL, *po_base = NULL;
    npy_bool *footprint;
    npy_intp array_size, ii, f_size, *offsets, *ff;
    npy_intp *dfs = NULL, n = 1, ni = 0, ii;

    array_size = PyArray_SIZE(output);

    if (!PYCV_AllocateToFootprint(strel, &footprint, &f_size, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_AllocateToFootprint \n");
        goto exit;
    }

    if (!PYCV_InitNeighborhoodOffsets(output, PyArray_DIMS(strel), center, footprint,
                                      &offsets, NULL, &flag, PYCV_EXTEND_CONSTANT)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        goto exit;
    }
    PYCV_NeighborhoodIteratorInit(output, PyArray_DIMS(strel), center, f_size, &iter_o);

    dfs = malloc(array_size * sizeof(npy_intp));

    if (!dfs) {
        PyErr_NoMemory();
        goto exit;
    }

    PYCV_RAVEL_COORDINATE(seed_point, iter_o.nd_m1 + 1, iter_o.strides, *dfs);

    po_base = po = (void *)PyArray_DATA(output);

    while (ni < n) {
        p = *(dfs + ni);
        PYCV_NEIGHBORHOOD_ITERATOR_GOTO_RAVEL(iter_o, po_base, po, offsets, ff, p);

        for (ii = 0; ii < f_size; ii++) {
            FILL_IF_ZERO(iter_o.numtype, po, p, *(ff + ii), flag, dfs, n);
        }
        ni++
    }

    exit:
        free(dfs);
        free(offsets);
        free(footprint);
        return PyErr_Occurred() ? 0 : 1;
}
