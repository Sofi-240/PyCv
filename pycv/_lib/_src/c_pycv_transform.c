#include "c_pycv_base.h"
#include "c_pycv_transform.h"

// #####################################################################################################################

#define TOLERANCE 1e-15

// #####################################################################################################################

#define PYCV_T_INTERP_NN(_values, _delta, _out)                                                                        \
{                                                                                                                      \
    _out = _values[0];                                                                                                 \
}

#define PYCV_T_INTERP_LINEAR(_values, _delta, _out)                                                                    \
{                                                                                                                      \
    _out = (1 - _delta) * _values[0] + _delta * _values[1];                                                            \
}

#define PYCV_T_INTERP_QUADRATIC(_values, _delta, _out)                                                                 \
{                                                                                                                      \
    _out = _values[1] + 0.5 * _delta * (_values[2] - _values[0]) +                                                     \
            0.5 * _delta * _delta * (_values[2] - 2 * _values[1] + _values[0]);                                        \
}

#define PYCV_T_INTERP_CUBIC(_values, _delta, _out)                                                                     \
{                                                                                                                      \
    _out = _values[1] + 0.5 * _delta * (-_values[0] + _values[2] +                                                     \
           _delta * (2 * _values[0] - 5 * _values[1] + 4 * _values[2] - _values[3] +                                   \
           _delta * (-_values[0] + 3 * _values[1] - 3 * _values[2] + _values[3])));                                    \
}

#define PYCV_T_INTERPOLATION(_order, _values, _delta, _out)                                                            \
{                                                                                                                      \
    switch (_order) {                                                                                                  \
        case 0:                                                                                                        \
            PYCV_T_INTERP_NN(_values, _delta, _out);                                                                   \
            break;                                                                                                     \
        case 1:                                                                                                        \
            PYCV_T_INTERP_LINEAR(_values, _delta, _out);                                                               \
            break;                                                                                                     \
        case 2:                                                                                                        \
            PYCV_T_INTERP_QUADRATIC(_values, _delta, _out);                                                            \
            break;                                                                                                     \
        case 3:                                                                                                        \
            PYCV_T_INTERP_CUBIC(_values, _delta, _out);                                                                \
            break;                                                                                                     \
        default:                                                                                                       \
            _out = 0;                                                                                                  \
    }                                                                                                                  \
}

// #####################################################################################################################

int PYCV_InterpolationAuxObjectInit(npy_intp order, PyArrayObject *array, npy_intp ndim0, InterpolationAuxObject *object)
{
    npy_intp rank, max_dims = 0, max_stride = 0, stride_pos, ii;

    rank = PyArray_NDIM(array) - ndim0;

    for (ii = 0; ii < rank; ii++) {
        object->c_counter[ii] = 0;
        object->c_shifts[ii] = 0;

        object->a_dims[ii] = PyArray_DIM(array, (int)(ii + ndim0));
        object->a_strides[ii] =PyArray_STRIDE(array, (int)(ii + ndim0));

        max_dims = max_dims < object->a_dims[ii] ? object->a_dims[ii] : max_dims;
        stride_pos = object->a_strides[ii] < 0 ? -object->a_strides[ii] : object->a_strides[ii];
        max_stride = max_stride < stride_pos ? stride_pos : max_stride;
    }

    object->flag = max_dims * max_stride + 1;
    object->rank = rank;
    object->order = order;
    object->c_size = order + 1;
    object->c_strides[rank - 1] = 1;

    for (ii = rank - 2; ii >= 0; ii--) {
        object->c_size *= (order + 1);
        object->c_strides[ii] = object->c_strides[ii + 1] * (order + 1);
    }

    object->coefficients = malloc(object->c_size * sizeof(npy_double));
    if (!object->coefficients) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

#define PYCV_T_CASE_INTERP_AUX_BUILD_F(_NTYPE, _dtype, _object, _base_p, _p, _position0, _mode, _c_val)                \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    int _ii, _jj, _outside;                                                                                            \
    npy_intp _cc;                                                                                                      \
    for (_jj = 0; _jj < (_object).c_size; _jj++) {                                                                     \
        _p = _base_p;                                                                                                  \
        _outside = 0;                                                                                                  \
        for (_ii = 0; _ii < (_object).rank; _ii++) {                                                                   \
            if (!_outside) {                                                                                           \
                _cc = _position0[_ii] + (_object).c_shifts[_ii];                                                       \
                _cc = PYCV_FitCoordinate(_cc, (_object).a_dims[_ii], (_object).flag, _mode);                           \
                if (_cc == (_object).flag) {                                                                           \
                    _outside = 1;                                                                                      \
                } else {                                                                                               \
                    _p += (_object).a_strides[_ii] * _cc;                                                              \
                }                                                                                                      \
            }                                                                                                          \
            (_object).c_counter[_ii] += 1;                                                                             \
            if ((_object).c_counter[_ii] == (_object).c_strides[_ii]) {                                                \
                (_object).c_counter[_ii] = 0;                                                                          \
                if ((_object).c_shifts[_ii] == (_object).order) {                                                      \
                    (_object).c_shifts[_ii] = 0;                                                                       \
                } else {                                                                                               \
                    (_object).c_shifts[_ii]  += 1;                                                                     \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if (_outside) {                                                                                                \
            (_object).coefficients[_jj] = (npy_double)_c_val;                                                          \
        } else {                                                                                                       \
            (_object).coefficients[_jj] = (npy_double)(*(_dtype *)(_p));                                               \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_T_INTERP_AUX_BUILD_F(_NTYPE, _object, _base_p, _p, _position0, _mode, _c_val)                             \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(BOOL, npy_bool, _object, _base_p, _p, _position0, _mode, _c_val);               \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(UBYTE, npy_ubyte, _object, _base_p, _p, _position0, _mode, _c_val);             \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(USHORT, npy_ushort, _object, _base_p, _p, _position0, _mode, _c_val);           \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(UINT, npy_uint, _object, _base_p, _p, _position0, _mode, _c_val);               \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(ULONG, npy_ulong, _object, _base_p, _p, _position0, _mode, _c_val);             \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(ULONGLONG, npy_ulonglong, _object, _base_p, _p, _position0, _mode, _c_val);     \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(BYTE, npy_byte, _object, _base_p, _p, _position0, _mode, _c_val);               \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(SHORT, npy_short, _object, _base_p, _p, _position0, _mode, _c_val);             \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(INT, npy_int, _object, _base_p, _p, _position0, _mode, _c_val);                 \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(LONG, npy_long, _object, _base_p, _p, _position0, _mode, _c_val);               \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(LONGLONG, npy_longlong, _object, _base_p, _p, _position0, _mode, _c_val);       \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(FLOAT, npy_float, _object, _base_p, _p, _position0, _mode, _c_val);             \
        PYCV_T_CASE_INTERP_AUX_BUILD_F(DOUBLE, npy_double, _object, _base_p, _p, _position0, _mode, _c_val);           \
    }                                                                                                                  \
}

#define PYCV_T_INTERP_AUX_COMPUTE_FN(_NTYPE, _object, _p, _delta)                                                      \
{                                                                                                                      \
    npy_intp _fc_size, _ii, _jj, _kk;                                                                                  \
    npy_double _out;                                                                                                   \
    _fc_size = (_object).c_size / ((_object).order + 1);                                                               \
    for (_jj = (_object).rank - 1; _jj >= 0; _jj--) {                                                                  \
        _kk = 0;                                                                                                       \
        for (_ii = 0; _ii < _fc_size; _ii++) {                                                                         \
            PYCV_T_INTERPOLATION((_object).order, ((_object).coefficients + _ii * ((_object).order + 1)), _delta[_jj], _out);\
            (_object).coefficients[_kk] = _out;                                                                        \
            _kk++;                                                                                                     \
        }                                                                                                              \
        _fc_size = _kk / ((_object).order + 1);                                                                        \
    }                                                                                                                  \
    _out = (_object).coefficients[0];                                                                                  \
    PYCV_SET_VALUE_F2A(_NTYPE, _p, _out);                                                                              \
}


// #####################################################################################################################

int PYCV_resize(PyArrayObject *input,
                PyArrayObject *output,
                npy_intp order,
                int grid_mode,
                PYCV_ExtendBorder mode,
                npy_double c_val)
{
    PYCV_ArrayIterator iter_o;
    npy_double scale_factor[NPY_MAXDIMS], projection[NPY_MAXDIMS], delta[NPY_MAXDIMS];
    npy_intp out_size, ii, jj, position0[NPY_MAXDIMS];
    int num_type_i, num_type_o;
    char *po = NULL, *pi = NULL, *pi_base = NULL;
    InterpolationAuxObject aux;

    NPY_BEGIN_THREADS_DEF;

    if (!PYCV_InterpolationAuxObjectInit(order, input, 0, &aux)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InterpolationAuxObjectInit \n");
        goto exit;
    }

    out_size = PyArray_SIZE(output);

    for (ii = 0; ii < aux.rank; ii++) {
        if (grid_mode) {
            scale_factor[ii] = (npy_double)(aux.a_dims[ii] - 1) / (npy_double)(PyArray_DIM(output, (int)(ii)) - 1);
        } else {
            scale_factor[ii] = (npy_double)aux.a_dims[ii] / (npy_double)PyArray_DIM(output, (int)(ii));
        }
    }

    PYCV_ArrayIteratorInit(output, &iter_o);

    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);

    NPY_BEGIN_THREADS;

    pi_base = pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    for (ii = 0; ii < out_size; ii++) {
        for (jj = 0; jj < aux.rank; jj++) {
            projection[jj] = (npy_double)iter_o.coordinates[jj] * scale_factor[jj];
            if (order & 1) {
                position0[jj] = (npy_intp)floor(projection[jj]) - order / 2;
                delta[jj] = projection[jj] - (npy_intp)floor(projection[jj]);
            } else {
                position0[jj] = (npy_intp)floor(projection[jj] + 0.5) - order / 2;
                delta[jj] = projection[jj] - (npy_intp)floor(projection[jj] + 0.5);
            }
        }
        PYCV_T_INTERP_AUX_BUILD_F(num_type_i, aux, pi_base, pi, position0, mode, c_val);
        PYCV_T_INTERP_AUX_COMPUTE_FN(num_type_o, aux, po, delta);
        PYCV_ARRAY_ITERATOR_NEXT(iter_o, po);
    }

    NPY_END_THREADS;
    exit:
        free(aux.coefficients);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

static int PYCV_AllocateMatrix(PyArrayObject *matrix, npy_double **h)
{
    PYCV_ArrayIterator iter;
    char *pm = NULL;
    int num_type;
    npy_intp h_size, ii;
    npy_double *hh;

    h_size = PyArray_SIZE(matrix);
    num_type = PyArray_TYPE(matrix);

    *h = malloc(h_size * sizeof(npy_double));
    if (!*h) {
        PyErr_NoMemory();
        return 0;
    }
    hh = *h;

    pm = (void *)PyArray_DATA(matrix);
    PYCV_ArrayIteratorInit(matrix, &iter);

    for (ii = 0; ii < h_size; ii++) {
        PYCV_GET_VALUE(num_type, npy_double, pm, hh[ii]);
        PYCV_ARRAY_ITERATOR_NEXT(iter, pm);
    }
    return 1;
}

#define PYCV_T_G_TRANSFORM_SWAP_XY(_coord)                                                                             \
{                                                                                                                      \
    npy_double _tmp = _coord[0];                                                                                       \
    _coord[0] = _coord[1];                                                                                             \
    _coord[1] = _tmp;                                                                                                  \
}

#define PYCV_T_DOT(_nd, _src, _h, _dst)                                                                                \
{                                                                                                                      \
    npy_intp _ii, _jj;                                                                                                 \
    for (_ii = _nd - 1; _ii >= 0; _ii--) {                                                                             \
        _dst[_ii] = 0;                                                                                                 \
        for (_jj = 0; _jj < _nd; _jj++) {                                                                              \
            _dst[_ii] += _src[_jj] * _h[_ii * _nd + _jj];                                                              \
        }                                                                                                              \
        if (_ii < _nd - 1) {                                                                                           \
            _dst[_ii] /= _dst[_nd - 1];                                                                                \
        } else if (_dst[_ii] == 0) {                                                                                   \
            _dst[_ii] = (npy_double)TOLERANCE;                                                                         \
        }                                                                                                              \
    }                                                                                                                  \
}

int PYCV_geometric_transform(PyArrayObject *matrix,
                             PyArrayObject *input,
                             PyArrayObject *output,
                             PyArrayObject *src,
                             PyArrayObject *dst,
                             npy_intp order,
                             PYCV_ExtendBorder mode,
                             npy_double c_val)
{
    PyArrayObject *p_src, *p_dst;
    PYCV_ArrayIterator iter_s, iter_d;
    npy_double delta[NPY_MAXDIMS], src_buffer[NPY_MAXDIMS], dst_buffer[NPY_MAXDIMS], *h, cc;
    npy_intp ndim, ii, jj, kk, position0[NPY_MAXDIMS], rank_p1, rank, src_size, pre_size, pre_stride = 0;
    char *src_base = NULL, *psrc = NULL, *pdst = NULL;
    int case_input_output, case_src_dst, num_type_s, num_type_d;
    InterpolationAuxObject aux;

    NPY_BEGIN_THREADS_DEF;

    case_input_output = (input != NULL) & (output != NULL) ? 1 : 0;
    case_src_dst = (src != NULL) & (dst != NULL) ? 1 : 0;

    if (!(case_input_output ^ case_src_dst)) {
        PyErr_SetString(PyExc_RuntimeError, "inputs and output mode or scr and dst mode \n");
        goto exit;
    }

    rank_p1 = PyArray_DIM(matrix, 1);
    rank = rank_p1 - 1;

    if (case_input_output) {
        p_src = input;
        p_dst = output;

        ndim = PyArray_NDIM(p_src);

        if (!PYCV_InterpolationAuxObjectInit(order, p_src, ndim - rank, &aux)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InterpolationAuxObjectInit \n");
            goto exit;
        }

        src_size = pre_size = 1;

        for (ii = 0; ii < ndim; ii++) {
            if (ii < ndim - rank) {
                pre_size *= PyArray_DIM(p_dst, (int)ii);
            } else {
                src_size *= PyArray_DIM(p_dst, (int)ii);
            }
        }
        if (ndim - rank) {
            pre_stride = PyArray_STRIDE(p_src, (int)(ndim - rank - 1));
        }
    } else {
        pre_size = 1;
        src_size = PyArray_DIM(src, 0);
        ndim = PyArray_DIM(src, 1);
        p_src = src;
        p_dst = dst;
    }

    num_type_s = PyArray_TYPE(p_src);
    num_type_d = PyArray_TYPE(p_dst);

    if (!PYCV_AllocateMatrix(matrix, &h)) {
        PyErr_NoMemory();
        goto exit;
    }

    PYCV_ArrayIteratorInit(p_src, &iter_s);
    PYCV_ArrayIteratorInit(p_dst, &iter_d);

    NPY_BEGIN_THREADS;

    src_base = psrc = (void *)PyArray_DATA(p_src);
    pdst = (void *)PyArray_DATA(p_dst);

    for (ii = 0; ii < pre_size; ii++) {
        for (jj = 0; jj < src_size; jj++) {
            if (case_input_output) {
                for (kk = 0; kk < rank; kk++) {
                    src_buffer[kk] = (npy_double)iter_d.coordinates[ndim - rank + kk];
                }
                src_buffer[rank] = 1;
                if (rank > 1) {
                    PYCV_T_G_TRANSFORM_SWAP_XY(src_buffer);
                }
                PYCV_T_DOT(rank_p1, src_buffer, h, dst_buffer);
                if (rank > 1) {
                    PYCV_T_G_TRANSFORM_SWAP_XY(dst_buffer);
                }

                for (kk = 0; kk < rank; kk++) {
                    if (order & 1) {
                        position0[kk] = (npy_intp)floor(dst_buffer[kk]) - order / 2;
                        delta[kk] = dst_buffer[kk] - (npy_intp)floor(dst_buffer[kk]);
                    } else {
                        position0[kk] = (npy_intp)floor(dst_buffer[kk] + 0.5) - order / 2;
                        delta[kk] = dst_buffer[kk] - (npy_intp)floor(dst_buffer[kk] + 0.5);
                    }
                }
                PYCV_T_INTERP_AUX_BUILD_F(num_type_s, aux, src_base, psrc, position0, mode, c_val);
                PYCV_T_INTERP_AUX_COMPUTE_FN(num_type_d, aux, pdst, delta);

                PYCV_ARRAY_ITERATOR_NEXT(iter_d, pdst);

            } else {
                for (kk = 0; kk < (ndim - rank_p1); kk++) {
                    PYCV_GET_VALUE(num_type_s, npy_double, psrc, cc);
                    PYCV_SET_VALUE_F2A(num_type_d, pdst, cc);
                    PYCV_ARRAY_ITERATOR_NEXT2(iter_s, psrc, iter_d, pdst);
                }

                for (kk = 0; kk < rank_p1; kk++) {
                    PYCV_GET_VALUE(num_type_s, npy_double, psrc, src_buffer[kk]);
                    PYCV_ARRAY_ITERATOR_NEXT(iter_s, psrc);
                }

                PYCV_T_DOT(rank_p1, src_buffer, h, dst_buffer);

                for (kk = 0; kk < rank_p1; kk++) {
                    PYCV_SET_VALUE_F2A(num_type_d, pdst, dst_buffer[kk]);
                    PYCV_ARRAY_ITERATOR_NEXT(iter_d, pdst);
                }
            }
        }
        if (case_input_output) {
            src_base += pre_stride;
        }
    }
    NPY_END_THREADS;
    exit:
        if (case_input_output) {
            free(aux.coefficients);
        }
        free(h);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define PYCV_T_CASE_HOUGH_ADD_VALUE(_NTYPE, _dtype, _pointer, _val)                                                    \
case NPY_##_NTYPE:                                                                                                     \
    *(_dtype *)_pointer += (_dtype)_val;                                                                               \
    break

#define PYCV_T_HOUGH_ADD_VALUE(_NTYPE, _pointer, _val)                                                                 \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_T_CASE_HOUGH_ADD_VALUE(BOOL, npy_bool, _pointer, _val);                                                   \
        PYCV_T_CASE_HOUGH_ADD_VALUE(UBYTE, npy_ubyte, _pointer, _val);                                                 \
        PYCV_T_CASE_HOUGH_ADD_VALUE(USHORT, npy_ushort, _pointer, _val);                                               \
        PYCV_T_CASE_HOUGH_ADD_VALUE(UINT, npy_uint, _pointer, _val);                                                   \
        PYCV_T_CASE_HOUGH_ADD_VALUE(ULONG, npy_ulong, _pointer, _val);                                                 \
        PYCV_T_CASE_HOUGH_ADD_VALUE(ULONGLONG, npy_ulonglong, _pointer, _val);                                         \
        PYCV_T_CASE_HOUGH_ADD_VALUE(BYTE, npy_byte, _pointer, _val);                                                   \
        PYCV_T_CASE_HOUGH_ADD_VALUE(SHORT, npy_short, _pointer, _val);                                                 \
        PYCV_T_CASE_HOUGH_ADD_VALUE(INT, npy_int, _pointer, _val);                                                     \
        PYCV_T_CASE_HOUGH_ADD_VALUE(LONG, npy_long, _pointer, _val);                                                   \
        PYCV_T_CASE_HOUGH_ADD_VALUE(LONGLONG, npy_longlong, _pointer, _val);                                           \
        PYCV_T_CASE_HOUGH_ADD_VALUE(FLOAT, npy_float, _pointer, _val);                                                 \
        PYCV_T_CASE_HOUGH_ADD_VALUE(DOUBLE, npy_double, _pointer, _val);                                               \
    }                                                                                                                  \
}


#define PYCV_T_HOUGH_GET_CIRCLE_POINTS(_radius, _circle, _size)                                                        \
{                                                                                                                      \
    npy_intp _rr = 0, _sh = 3 - 2 * _radius, _xx = _radius, _yy = 0;                                                   \
    _size = 0;                                                                                                         \
    while (_xx >= _yy) {                                                                                               \
        _circle[_rr++] = _yy;                                                                                          \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = _yy;                                                                                          \
        _circle[_rr++] = _yy;                                                                                          \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = -_yy;                                                                                         \
        _circle[_rr++] = _xx;                                                                                          \
        _circle[_rr++] = -_xx;                                                                                         \
        _circle[_rr++] = _yy;                                                                                          \
        _size += 8;                                                                                                    \
        if (_sh < 0) {                                                                                                 \
            _sh += 4 * _yy + 6;                                                                                        \
        } else {                                                                                                       \
            _sh += 4 * (_yy - _xx) + 10;                                                                               \
            _xx -= 1;                                                                                                  \
        }                                                                                                              \
        _yy += 1;                                                                                                      \
    }                                                                                                                  \
}


PyArrayObject *PYCV_hough_line_transform(PyArrayObject *input,
                                         PyArrayObject *theta,
                                         npy_intp offset)
{
    int num_type_i, num_type_t, num_type_h = NPY_UINT64;
    PYCV_ArrayIterator iter_i, iter_t, iter_h;
    char *pi = NULL, *ph_base = NULL, *ph = NULL, *pt = NULL;
    npy_intp ndim, init_size = 1, stride, input_shape[NPY_MAXDIMS], ndim_init;

    PyArrayObject *h_space;
    npy_intp *h_shape, h_size = 1, n_theta, n_rho;
    npy_double *cosine, *sine, i_val, y, x, proj, angle;
    npy_intp ii, jj, hh;

    NPY_BEGIN_THREADS_DEF;

    ndim = PyArray_NDIM(input);
    ndim_init = ndim - 2;

     h_shape = malloc(ndim * sizeof(npy_intp));

    if (!h_shape) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < ndim; ii++) {
        input_shape[ii] = PyArray_DIM(input, (int)ii);
        if (ii < ndim_init) {
            init_size *= input_shape[ii];
            h_shape[ii] = input_shape[ii];
        } else {
            h_size *= input_shape[ii];
        }
    }

    n_theta = PyArray_SIZE(theta);
    n_rho = 2 * offset + 1;

    h_shape[ndim_init] = n_rho;
    h_shape[ndim_init + 1] = n_theta;

    h_space = (PyArrayObject *)PyArray_ZEROS((int)ndim, h_shape, num_type_h, 0);
    stride = PyArray_STRIDE(h_space, (int)(ndim_init - 1));

    num_type_t = PyArray_TYPE(theta);
    num_type_i = PyArray_TYPE(input);

    PYCV_ArrayIteratorInit(h_space, &iter_h);
    PYCV_ArrayIteratorInit(input, &iter_i);
    PYCV_ArrayIteratorInit(theta, &iter_t);

    cosine = malloc(n_theta * sizeof(npy_double));
    sine = malloc(n_theta * sizeof(npy_double));

    if (!cosine || !sine) {
        PyErr_NoMemory();
        goto exit;
    }

    pt = (void *)PyArray_DATA(theta);
    pi = (void *)PyArray_DATA(input);
    ph = (void *)PyArray_DATA(h_space);

    for (ii = 0; ii < n_theta; ii++) {
        PYCV_GET_VALUE(num_type_t, npy_double, pt, angle);
        cosine[ii] = (npy_double)cos(angle);
        sine[ii] = (npy_double)sin(angle);
        PYCV_ARRAY_ITERATOR_NEXT(iter_t, pt);
    }

    NPY_BEGIN_THREADS;

    while (init_size--) {
        for (ii = 0; ii < h_size; ii++) {
            PYCV_GET_VALUE(num_type_i, npy_double, pi, i_val);
            if (fabs(i_val) > DBL_EPSILON) {
                y = (npy_double)iter_i.coordinates[ndim_init];
                x = (npy_double)iter_i.coordinates[ndim_init + 1];
                for (jj = 0; jj < n_theta; jj++) {
                    proj = cosine[jj] * x + sine[jj] * y;
                    hh = (npy_intp)floor(proj + 0.5) + offset;
                    if (hh >= n_rho) {
                        continue;
                    }
                    hh = hh * iter_h.strides[ndim_init] + jj * iter_h.strides[ndim_init + 1];
                    PYCV_T_HOUGH_ADD_VALUE(num_type_h, (ph + hh), 1);
                }
            }
            PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
        }
        ph += stride;
    }
    NPY_END_THREADS;

    exit:
        free(h_shape);
        free(cosine);
        free(sine);
        return PyErr_Occurred() ? NULL : h_space;
}


PyArrayObject *PYCV_hough_circle_transform(PyArrayObject *input,
                                           PyArrayObject *radius,
                                           int normalize,
                                           int expend)
{
    int num_type_i, num_type_r, num_type_h;
    PYCV_ArrayIterator iter_i, iter_r, iter_h;
    char *pi = NULL, *ph = NULL, *pr = NULL, *pr_base = NULL;
    npy_intp ndim, init_size = 1, input_shape[NPY_MAXDIMS], ndim_init;

    PyArrayObject *h_space;
    npy_intp *h_shape, h_size = 1, stride, n_radius, shift = 0;
    npy_intp **circle_points, circle_size, max_size, max_r = 0, r, y, x, proj_y, proj_x;
    npy_double i_val, incr_val;

    npy_intp ii, jj, hh, hhp, rr, kk;

    NPY_BEGIN_THREADS_DEF;

    ndim = PyArray_NDIM(input);
    ndim_init = ndim - 2;

    h_shape = malloc((ndim + 1) * sizeof(npy_intp));

    if (!h_shape) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < ndim; ii++) {
        input_shape[ii] = PyArray_DIM(input, (int)ii);
        if (ii < ndim_init) {
            init_size *= input_shape[ii];
            h_shape[ii] = input_shape[ii];
        } else {
            h_size *= input_shape[ii];
        }
    }

    n_radius = PyArray_SIZE(radius);

    num_type_h = normalize ? NPY_DOUBLE : NPY_UINT64;
    num_type_i = PyArray_TYPE(input);
    num_type_r = PyArray_TYPE(radius);

    pr_base = pr = (void *)PyArray_DATA(radius);
    PYCV_ArrayIteratorInit(radius, &iter_r);

    for (ii = 0; ii < n_radius; ii++) {
        PYCV_GET_VALUE(num_type_r, npy_intp, pr, r);
        max_r = r > max_r ? r : max_r;
        PYCV_ARRAY_ITERATOR_NEXT(iter_r, pr);
    }
    pr = pr_base;
    PYCV_ARRAY_ITERATOR_RESET(iter_r);

    if (expend) {
        shift = max_r;
    }

    h_shape[ndim_init] = n_radius;
    h_shape[ndim_init + 1] = input_shape[ndim_init] + 2 * shift;
    h_shape[ndim_init + 2] = input_shape[ndim_init + 1] + 2 * shift;

    h_space = (PyArrayObject *)PyArray_ZEROS((int)(ndim + 1), h_shape, num_type_h, 0);
    if (!h_space) {
        PyErr_NoMemory();
        goto exit;
    }

    stride = PyArray_STRIDE(h_space, (int)(ndim_init - 1));

    circle_points = malloc(n_radius * sizeof(npy_intp*));

    if (!circle_points) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = 0; ii < n_radius; ii++) {
        PYCV_GET_VALUE(num_type_r, npy_intp, pr, r);
        max_size = 2 * ((r * 8) + 8);
        circle_points[ii] = malloc((max_size + 1) * sizeof(npy_intp));
        if (!circle_points[ii]) {
            PyErr_NoMemory();
            goto exit;
        }
        PYCV_T_HOUGH_GET_CIRCLE_POINTS(r, circle_points[ii], circle_size);
        circle_points[ii][max_size] = circle_size;
        PYCV_ARRAY_ITERATOR_NEXT(iter_r, pr);
    }
    pr = pr_base;
    PYCV_ARRAY_ITERATOR_RESET(iter_r);

    PYCV_ArrayIteratorInit(h_space, &iter_h);
    PYCV_ArrayIteratorInit(input, &iter_i);

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    ph = (void *)PyArray_DATA(h_space);


    while (init_size--) {
        for (ii = 0; ii < h_size; ii++) {
            y = iter_i.coordinates[ndim_init];
            x = iter_i.coordinates[ndim_init + 1];
            PYCV_GET_VALUE(num_type_i, npy_double, pi, i_val);

            if (fabs(i_val) > DBL_EPSILON) {
                y += shift;
                x += shift;

                for (rr = 0; rr < n_radius; rr++) {
                    PYCV_GET_VALUE(num_type_r, npy_intp, pr, r);
                    max_size = 2 * ((r * 8) + 8);
                    circle_size = circle_points[rr][max_size];

                    incr_val = normalize ? 1 / (npy_double)circle_size : 1;
                    hh = rr * iter_h.strides[ndim_init];
                    kk = 0;
                    for (jj = 0; jj < circle_size; jj++) {
                        proj_y = y + circle_points[rr][kk];
                        proj_x = x + circle_points[rr][kk + 1];
                        kk += 2;

                        if (shift || (proj_y >= 0 && proj_y < h_shape[ndim_init + 1] && proj_x >= 0 && proj_x < h_shape[ndim_init + 2])) {
                            hhp = hh + proj_y * iter_h.strides[ndim_init + 1] + proj_x * iter_h.strides[ndim_init + 2];
                            PYCV_T_HOUGH_ADD_VALUE(num_type_h, (ph + hhp), incr_val);
                        }
                    }
                    PYCV_ARRAY_ITERATOR_NEXT(iter_r, pr);
                }
                pr = pr_base;
                PYCV_ARRAY_ITERATOR_RESET(iter_r);
            }

            PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
        }
        ph += stride;
    }

    NPY_END_THREADS;
    exit:
        free(h_shape);
        for (ii = 0; ii < n_radius; ii++) {
            free(circle_points[ii]);
        }
        free(circle_points);
        return PyErr_Occurred() ? NULL : h_space;
}


// #####################################################################################################################





















