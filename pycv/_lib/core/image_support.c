#include "ops_base.h"
#include "image_support.h"

// #####################################################################################################################

/*
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


is_up and is_right:
gy >= 0 and gx >= 0

  |
   __ ---->  / ---> \ gradient direction

is_down and is_left:
gy <= 0 and gx <= 0

  __     ---> / ---> \ gradient direction
     |

is_down and is_right:
gy <= 0 and gx >= 0

    __   ---> \---> / gradient direction
   |

is_up and is_left:
gy >= 0 and gx <= 0

    |  ---> \---> / gradient direction
  __

[0, 1, 2],
[3, 4, 5]
[6, 7, 8]

90 - 135                    135 - 180
[22, 21, 0],                [22,  0,  0],
[0,  1,  0],      or        [21,  1, 11],
[0, 11, 12]                 [ 0,  0, 12],

|gy| > |gx|                 |gy| <= |gx|
e = |gx| / |gy|             e = |gy| / |gx|

0 - 45                     45 - 90
[ 0,  0, 12],              [ 0, 11, 12],
[21,  1, 11],      or      [ 0,  1,  0],
[22,  0,  0]               [22, 21,  0],

|gx| > |gy|                |gx| <= |gy|
e = |gy| / |gx|            e = |gx| / |gy|

upper = (12 * w + 11 * (1.0 - w))
lower = (22 * w + 21 * (1.0 - w))
*/

#define CASE_TYPE_CANNY_INTERPOLATE(_NUM_TYPE, _type, _offsets, _th, _mag_po, _gy, _gx, _out_val)                      \
case _NUM_TYPE:                                                                                                        \
{                                                                                                                      \
    unsigned int _grad_up, _grad_down, _grad_left, _grad_right, _diag_pos, _diag_neg;                                  \
    double _abs_gy, _abs_gx, _est, _mag, _p11, _p12, _p21, _p22;                                                       \
    _mag = (double)(*((_type *)_mag_po));                                                                              \
    _out_val = 0.0;                                                                                                    \
    if (_mag >= _th) {                                                                                                 \
        _grad_up = _gy >= 0 ? 1 : 0;                                                                                   \
        _grad_down = _gy <= 0 ? 1 : 0;                                                                                 \
        _grad_right = _gx >= 0 ? 1 : 0;                                                                                \
        _grad_left = _gx <= 0 ? 1 : 0;                                                                                 \
        _diag_neg = (_grad_up && _grad_right) || (_grad_down && _grad_left) ? 1 : 0;                                   \
        _diag_pos = (_grad_down && _grad_right) || (_grad_up && _grad_left) ? 1 : 0;                                   \
        if (_diag_neg || _diag_pos) {                                                                                  \
            _abs_gy = fabs(_gy);                                                                                       \
            _abs_gx = fabs(_gx);                                                                                       \
            if (_diag_neg && _abs_gy > _abs_gx) {                                                                      \
                _est = _abs_gx / _abs_gy;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[7])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[8])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[1])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[0])));                                                  \
            } else if (_diag_neg) {                                                                                    \
                _est = _abs_gy / _abs_gx;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[5])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[8])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[3])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[0])));                                                  \
            } else if (_diag_pos && _abs_gy < _abs_gx) {                                                               \
                _est = _abs_gy / _abs_gx;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[5])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[2])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[3])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[6])));                                                  \
            } else {                                                                                                   \
                _est = _abs_gx / _abs_gy;                                                                              \
                _p11 = (double)(*((_type *)(_mag_po + _offsets[1])));                                                  \
                _p12 = (double)(*((_type *)(_mag_po + _offsets[2])));                                                  \
                _p21 = (double)(*((_type *)(_mag_po + _offsets[7])));                                                  \
                _p22 = (double)(*((_type *)(_mag_po + _offsets[6])));                                                  \
            }                                                                                                          \
            if ((_p12 * _est + _p11 * (1 - _est)) <= _mag && (_p22 * _est + _p21 * (1 - _est)) <= _mag) {              \
                _out_val = _mag;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define EX_CANNY_INTERPOLATE(_NUM_TYPE, _offsets, _th, _mag_po, _gy, _gx, _out_val)                                    \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_BOOL, npy_bool, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_UBYTE, npy_ubyte, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_USHORT, npy_ushort, _offsets, _th, _mag_po, _gy, _gx, _out_val);               \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_UINT, npy_uint, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_ULONG, npy_ulong, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_ULONGLONG, npy_ulonglong, _offsets, _th, _mag_po, _gy, _gx, _out_val);         \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_BYTE, npy_byte, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_SHORT, npy_short, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_INT, npy_int, _offsets, _th, _mag_po, _gy, _gx, _out_val);                     \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_LONG, npy_long, _offsets, _th, _mag_po, _gy, _gx, _out_val);                   \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_LONGLONG, npy_longlong, _offsets, _th, _mag_po, _gy, _gx, _out_val);           \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_FLOAT, npy_float, _offsets, _th, _mag_po, _gy, _gx, _out_val);                 \
        CASE_TYPE_CANNY_INTERPOLATE(NPY_DOUBLE, npy_double, _offsets, _th, _mag_po, _gy, _gx, _out_val);               \
    }                                                                                                                  \
}

int ops_canny_nonmaximum_suppression(PyArrayObject *magnitude,
                                     PyArrayObject *grad_y,
                                     PyArrayObject *grad_x,
                                     double threshold,
                                     PyArrayObject *mask,
                                     PyArrayObject *output)
{
    const npy_intp shape[2] = {3, 3};

    npy_intp array_size;
    npy_intp *offsets;
    npy_bool *borders_lookup;
    npy_intp ii;

    ArrayIter iter_o, iter_ma, iter_m, iter_y, iter_x;
    char *po = NULL, *pma = NULL, *pm = NULL, *py = NULL, *px = NULL;
    int num_type_o, num_type_ma, num_type_m, num_type_y, num_type_x;

    double gy = 0.0, gx = 0.0, mag;
    int ma;

    NPY_BEGIN_THREADS_DEF;

    if (!init_offsets_ravel(magnitude, shape, NULL, NULL, &offsets)) {
        goto exit;
    }
    if (!init_borders_lut(PyArray_NDIM(magnitude), PyArray_DIMS(magnitude), shape, NULL, &borders_lookup)) {
        goto exit;
    }

    array_size = PyArray_SIZE(magnitude);

    num_type_o = PyArray_TYPE(output);
    if (mask) {
        num_type_ma = PyArray_TYPE(mask);
    }
    num_type_m = PyArray_TYPE(magnitude);
    num_type_y = PyArray_TYPE(grad_y);
    num_type_x = PyArray_TYPE(grad_x);

    ArrayIterInit(output, &iter_o);
    if (mask) {
        ArrayIterInit(mask, &iter_ma);
    }
    ArrayIterInit(magnitude, &iter_m);
    ArrayIterInit(grad_y, &iter_y);
    ArrayIterInit(grad_x, &iter_x);

    NPY_BEGIN_THREADS;

    po = (void *)PyArray_DATA(output);
    if (mask) {
        pma = (void *)PyArray_DATA(mask);
    }
    pm = (void *)PyArray_DATA(magnitude);
    py = (void *)PyArray_DATA(grad_y);
    px = (void *)PyArray_DATA(grad_x);

    for (ii = 0; ii < array_size; ii++) {
        mag = 0.0;
        ma = 0;
        if (!borders_lookup[ii]) {
            ma = 1;
            if (mask) {
                GET_VALUE_AS(num_type_ma, int, pma, ma);
            }
        }
        if (ma) {
            GET_VALUE_AS(num_type_y, double, py, gy);
            GET_VALUE_AS(num_type_x, double, px, gx);
            EX_CANNY_INTERPOLATE(num_type_m, offsets, threshold, pm, gy, gx, mag);
        }
        SET_VALUE_TO(num_type_o, po, mag);

        ARRAY_ITER_NEXT3(iter_m, pm, iter_y, py, iter_x, px);
        if (mask) {
            ARRAY_ITER_NEXT2(iter_ma, pma, iter_o, po);
        } else {
            ARRAY_ITER_NEXT(iter_o, po);
        }
    }
    NPY_END_THREADS;
    exit:
        free(offsets);
        free(borders_lookup);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

//int ops_build_max_tree(PyArrayObject *input,
//                       PyArrayObject *traverser,
//                       PyArrayObject *parent,
//                       int connectivity,
//                       PyArrayObject *values_map)
//{
//
//}


int ops_build_max_tree(PyArrayObject *input, PyArrayObject *traverser, PyArrayObject *parent)
{
    int contiguous_i, contiguous_t, contiguous_p;
    int num_type_i, num_type_t, num_type_p;
    npy_intp itemsize_i, itemsize_t, itemsize_p, size, nd, *strides, *dims;
    char *pi = NULL, *pt = NULL, *pp = NULL, *pi_base = NULL, *pp_base = NULL;

    npy_intp *trav, ii;
    PyArray_ArgSortFunc *argsort_func;

    npy_bool *footprint = NULL;
    npy_intp *offsets, *offsets_cc;
    int footprint_size;
    npy_intp shape[NPY_MAXDIMS];

    npy_intp *nodes, jj;
    npy_intp idx, cc, cci, outside = 0;
    npy_intp ind, ind_n, node, root, q, qq;
    int undef = -1;
    double val1 = 0, val2 = 0;

    contiguous_i = PyArray_ISCONTIGUOUS(input);
    contiguous_t = PyArray_ISCONTIGUOUS(traverser);
    contiguous_p = PyArray_ISCONTIGUOUS(parent);

    if (!contiguous_i || !contiguous_t || !contiguous_p) {
        PyErr_SetString(PyExc_RuntimeError, "all the inputs need to be contiguous\n");
        return 0;
    }

    size = PyArray_SIZE(input);
    nd = PyArray_NDIM(input);

    itemsize_i = PyArray_ITEMSIZE(input);
    itemsize_t = PyArray_ITEMSIZE(traverser);
    itemsize_p = PyArray_ITEMSIZE(parent);

    num_type_i = PyArray_TYPE(input);
    num_type_t = PyArray_TYPE(traverser);
    num_type_p = PyArray_TYPE(parent);

    strides = PyArray_STRIDES(input);
    dims = PyArray_DIMS(input);

    trav = (npy_intp*)malloc(size * sizeof(npy_intp));
    if (!trav) {
        PyErr_NoMemory();
        return 0;
    }

    pi_base = pi = (void *)PyArray_DATA(input);
    pt = (void *)PyArray_DATA(traverser);
    pp_base = pp = (void *)PyArray_DATA(parent);

    nodes = (npy_intp*)malloc(size * sizeof(npy_intp));
    if (!nodes) {
        PyErr_NoMemory();
        return 0;
    }

    for (ii = 0; ii < size; ii++) {
        SET_VALUE_TO(num_type_p, pp, undef);
        pp += itemsize_p;
        nodes[ii] = undef;
        trav[ii] = ii;
    }

    argsort_func = PyArray_DESCR(input)->f->argsort[NPY_MERGESORT];
    if (argsort_func == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "argsort_func not supported\n");
        goto exit;
    }

    if (argsort_func(pi, trav, size, input) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Couldn't perform argsort.\n");
        goto exit;
    }

    if (!footprint_as_con(nd, 2, &footprint, &footprint_size, 1)) {
        goto exit;
    }
    for (ii = 0; ii < nd; ii++) {
        shape[ii] = 3;
    }

    if (!init_offsets_ravel(input, shape, NULL, footprint, &offsets)){
        goto exit;
    }

    for (ii = 0; ii < footprint_size; ii++) {
        offsets[ii] /= itemsize_i;
    }

    if (!init_offsets_coordinates(nd, shape, NULL, footprint, &offsets_cc)){
        goto exit;
    }

    for (ii = size - 1; ii >= 0; ii--) {
        ind = trav[ii];

        pp = pp_base + ind * itemsize_p;
        SET_VALUE_TO(num_type_p, pp, ind);

        nodes[ind] = ind;

        for (jj = 0; jj < footprint_size; jj++) {

            /* check if the neighbor is within the extent of the array: */
            idx = ind * itemsize_i;
            outside = 0;
            for (cci = 0; cci < nd; cci++) {
                cc = idx / strides[cci];
                cc += offsets_cc[jj * nd + cci];
                if (cc < 0 || cc >= dims[cci]) {
                    outside = 1;
                    break;
                }
                idx -= (cc - offsets_cc[jj * nd + cci]) * strides[cci];
            }
            ind_n = ind + offsets[jj];

            if (!outside) {

                if (nodes[ind_n] != undef) {
                    node = ind_n;

                    while (nodes[node] != nodes[nodes[node]]) {
                        nodes[node] = nodes[nodes[node]];
                    }
                    root = nodes[node];

                    if (root != ind) {
                        pp = pp_base + root * itemsize_p;
                        SET_VALUE_TO(num_type_p, pp, ind);
                        nodes[root] = ind;
                    }
                }
            }
        }
    }

    for (ii = 0; ii < size; ii++) {
        ind = trav[ii];

        SET_VALUE_TO(num_type_t, pt, ind);
        pt += itemsize_t;

        pp = pp_base + ind * itemsize_p;
        GET_VALUE_AS(num_type_p, npy_intp, pp, q) ;

        pp = pp_base + q * itemsize_p;
        GET_VALUE_AS(num_type_p, npy_intp, pp, qq) ;

        pi = pi_base + q * itemsize_i;
        GET_VALUE_AS(num_type_i, double, pi, val1) ;

        pi = pi_base + qq * itemsize_i;
        GET_VALUE_AS(num_type_i, double, pi, val2) ;

        if (val1 == val2) {
            pp = pp_base + ind * itemsize_p;
            SET_VALUE_TO(num_type_p, pp, qq);
        }
    }

    exit:
        free(trav);
        free(offsets_cc);
        free(offsets);
        free(nodes);
        return PyErr_Occurred() ? 0 : 1;
}




