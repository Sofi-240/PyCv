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

// *********************************************************************************************************************


PyArrayObject *PYCV_hough_line_transform(PyArrayObject *input,
                                         PyArrayObject *theta,
                                         npy_intp offset)
{
    int num_type_i, num_type_t, num_type_h;
    PYCV_ArrayIterator iter_i, iter_t, iter_h;
    char *pi = NULL, *ph_base = NULL, *ph = NULL, *pt = NULL;
    npy_intp ndim, init_size = 1, input_shape[NPY_MAXDIMS], ndim_init, transform_size = 1;

    PyArrayObject *h_space;
    npy_intp *h_shape, h_size, n_theta, n_rho;
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
            transform_size *= input_shape[ii];
        }
    }

    n_theta = PyArray_SIZE(theta);
    n_rho = 2 * offset + 1;

    h_size = n_theta * n_rho;

    h_shape[ndim_init] = n_rho;
    h_shape[ndim_init + 1] = n_theta;

    h_space = (PyArrayObject *)PyArray_ZEROS((int)ndim, h_shape, NPY_UINT64, 0);

    num_type_t = PyArray_TYPE(theta);
    num_type_i = PyArray_TYPE(input);
    num_type_h = PyArray_TYPE(h_space);

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
        for (ii = 0; ii < transform_size; ii++) {
            PYCV_GET_VALUE(num_type_i, npy_double, pi, i_val);
            if (fabs(i_val) > DBL_EPSILON) {
                y = (npy_double)iter_i.coordinates[ndim_init];
                x = (npy_double)iter_i.coordinates[ndim_init + 1];
                for (jj = 0; jj < n_theta; jj++) {
                    proj = cosine[jj] * x + sine[jj] * y;
                    hh = (npy_intp)floor(proj + 0.5) + offset;
                    hh = hh * iter_h.strides[ndim_init] + jj * iter_h.strides[ndim_init + 1];
                    PYCV_T_HOUGH_ADD_VALUE(num_type_h, (ph + hh), 1);
                }
            }
            PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
        }
        if (ndim_init) {
            ph += iter_h.strides[ndim_init - 1];
        }
    }
    NPY_END_THREADS;

    exit:
        free(h_shape);
        free(cosine);
        free(sine);
        return PyErr_Occurred() ? NULL : h_space;
}

// *********************************************************************************************************************

#define PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, _yy, _xx, _index)                                                   \
{                                                                                                                      \
    _circle[_index] = _yy;                                                                                             \
    _circle[_index + 1] = _xx;                                                                                         \
    _index += 2;                                                                                                       \
}

#define PYCV_T_HOUGH_CIRCLE_POINTS_ADD_POINTS4(_circle, _yy, _xx, _index)                                              \
{                                                                                                                      \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, -_yy, _xx, _index);                                                     \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, _yy, _xx, _index);                                                      \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, -_yy, -_xx, _index);                                                    \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, _yy, -_xx, _index);                                                     \
}

#define PYCV_T_HOUGH_CIRCLE_POINTS_ADD_POINTS8(_circle, _yy, _xx, _index)                                              \
{                                                                                                                      \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_POINTS4(_circle, _yy, _xx, _index);                                                 \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_POINTS4(_circle, _xx, _yy, _index);                                                 \
}


#define PYCV_T_HOUGH_GET_CIRCLE_POINTS(_radius, _circle, _size)                                                        \
{                                                                                                                      \
    npy_intp _rr = 0, _err = 3 - 2 * _radius, _xx = 0, _yy = _radius;                                                  \
    _size = 0;                                                                                                         \
    while (_yy > _xx) {                                                                                                \
        if (_xx == 0) {                                                                                                \
            PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, _yy, _xx, _rr);                                                 \
            PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, -_yy, _xx, _rr);                                                \
            PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, _xx, _yy, _rr);                                                 \
            PYCV_T_HOUGH_CIRCLE_POINTS_ADD_YX(_circle, _xx, -_yy, _rr);                                                \
            _size += 4;                                                                                                \
        } else {                                                                                                       \
            PYCV_T_HOUGH_CIRCLE_POINTS_ADD_POINTS8(_circle, _yy, _xx, _rr);                                            \
            _size += 8;                                                                                                \
        }                                                                                                              \
        if (_err < 0) {                                                                                                \
            _err += 4 * _xx + 6;                                                                                       \
        } else {                                                                                                       \
            _err += 4 * (_xx - _yy) + 10;                                                                              \
            _yy -= 1;                                                                                                  \
        }                                                                                                              \
        _xx += 1;                                                                                                      \
    }                                                                                                                  \
    PYCV_T_HOUGH_CIRCLE_POINTS_ADD_POINTS4(_circle, _yy, _xx, _rr);                                                    \
    _size += 4;                                                                                                        \
}

PyArrayObject *PYCV_hough_circle_transform(PyArrayObject *input,
                                           PyArrayObject *radius,
                                           int normalize,
                                           int expend)
{
    int num_type_i, num_type_r, num_type_h;
    PYCV_ArrayIterator iter_i, iter_r, iter_h;
    char *pi = NULL, *ph = NULL, *ph_base = NULL, *pr = NULL, *pr_base = NULL, *pi_base = NULL;
    npy_intp ndim, ndim_init, init_size = 1, transform_size = 1, input_shape[NPY_MAXDIMS];

    PyArrayObject *h_space;
    npy_intp h_shape[NPY_MAXDIMS], coordinates[NPY_MAXDIMS], n_radius, shift = 0, max_r = 0;
    npy_intp *circle_points, *circle, circle_size, r, y, x, proj_y, proj_x;
    npy_double i_val, incr_val;

    npy_intp ii, jj, rr, kk;

    NPY_BEGIN_THREADS_DEF;

    ndim = PyArray_NDIM(input);
    ndim_init = ndim - 2;

    for (ii = 0; ii < ndim; ii++) {
        input_shape[ii] = PyArray_DIM(input, (int)ii);
        if (ii < ndim_init) {
            init_size *= input_shape[ii];
            h_shape[ii] = input_shape[ii];
        } else {
            transform_size *= input_shape[ii];
        }
        coordinates[ii] = 0;
    }
    coordinates[ndim] = 0;
    n_radius = PyArray_SIZE(radius);

    num_type_r = PyArray_TYPE(radius);
    pr_base = pr = (void *)PyArray_DATA(radius);
    PYCV_ArrayIteratorInit(radius, &iter_r);

    for (rr = 0; rr < n_radius; rr++) {
        PYCV_GET_VALUE(num_type_r, npy_intp, pr, r);
        max_r = r > max_r ? r : max_r;
        PYCV_ARRAY_ITERATOR_NEXT(iter_r, pr);
    }
    pr = pr_base;
    PYCV_ARRAY_ITERATOR_RESET(iter_r);

    shift = expend ? max_r : 0;
    h_shape[ndim_init] = n_radius;
    h_shape[ndim_init + 1] = input_shape[ndim_init] + 2 * shift;
    h_shape[ndim_init + 2] = input_shape[ndim_init + 1] + 2 * shift;

    h_space = (PyArrayObject *)PyArray_ZEROS((int)(ndim + 1), h_shape, NPY_DOUBLE, 0);
    if (!h_space) {
        PyErr_NoMemory();
        goto exit;
    }

    circle_points = malloc(2 * ((max_r * 8) + 8) * sizeof(npy_intp));
    if (!circle_points) {
        PyErr_NoMemory();
        goto exit;
    }

    num_type_h = PyArray_TYPE(h_space);
    num_type_i = PyArray_TYPE(input);

    PYCV_ArrayIteratorInit(h_space, &iter_h);
    PYCV_ArrayIteratorInit(input, &iter_i);

    NPY_BEGIN_THREADS;

    pi_base = pi = (void *)PyArray_DATA(input);
    ph_base = ph = (void *)PyArray_DATA(h_space);

    for (rr = 0; rr < n_radius; rr++) {
        PYCV_GET_VALUE(num_type_r, npy_intp, pr, r);
        PYCV_T_HOUGH_GET_CIRCLE_POINTS(r, circle_points, circle_size);

        incr_val = normalize ? 1 / (npy_double)circle_size : 1;

        coordinates[ndim_init] = rr;

        for (kk = 0; kk < init_size; kk++) {
            for (ii = 0; ii < ndim_init; ii++) {
                coordinates[ii] = iter_i.coordinates[ii];
            }

            for (ii = 0; ii < transform_size; ii++) {
                PYCV_GET_VALUE(num_type_i, npy_double, pi, i_val);

                if (fabs(i_val) > DBL_EPSILON) {
                    y = iter_i.coordinates[ndim_init] + shift;
                    x = iter_i.coordinates[ndim_init + 1] + shift;

                    circle = circle_points;

                    for (jj = 0; jj < circle_size; jj++) {
                        proj_y = y + circle[0];
                        proj_x = x + circle[1];

                        if (shift || (proj_y >= 0 && proj_y < h_shape[ndim_init + 1] && proj_x >= 0 && proj_x < h_shape[ndim_init + 2])) {
                            coordinates[ndim_init + 1] = proj_y;
                            coordinates[ndim_init + 2] = proj_x;

                            PYCV_ARRAY_ITERATOR_GOTO(iter_h, ph_base, ph, coordinates);
                            PYCV_T_HOUGH_ADD_VALUE(num_type_h, ph, incr_val);
                        }
                        circle += 2;
                    }
                }
                PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
            }
        }
        pi = pi_base;
        PYCV_ARRAY_ITERATOR_RESET(iter_i);
        PYCV_ARRAY_ITERATOR_NEXT(iter_r, pr);
    }

    NPY_END_THREADS;
    exit:
        free(circle_points);
        return PyErr_Occurred() ? NULL : h_space;
}

// *********************************************************************************************************************

#define PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(_NTYPE, _dtype, _itemsize, _ndim, _pointer, _point)                   \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii < _ndim; _ii++) {                                                                                \
        *(_dtype *)_pointer = (_dtype)_point[_ii];                                                                     \
        _pointer += _itemsize;                                                                                         \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_T_PROBABILISTIC_LINE_SET_POINT(_NTYPE, _itemsize, _ndim, _pointer, _point)                                \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(BOOL, npy_bool, _itemsize, _ndim, _pointer, _point);                  \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(UBYTE, npy_ubyte, _itemsize, _ndim, _pointer, _point);                \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(USHORT, npy_ushort, _itemsize, _ndim, _pointer, _point);              \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(UINT, npy_uint, _itemsize, _ndim, _pointer, _point);                  \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(ULONG, npy_ulong, _itemsize, _ndim, _pointer, _point);                \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(ULONGLONG, npy_ulonglong, _itemsize, _ndim, _pointer, _point);        \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(BYTE, npy_byte, _itemsize, _ndim, _pointer, _point);                  \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(SHORT, npy_short, _itemsize, _ndim, _pointer, _point);                \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(INT, npy_int, _itemsize, _ndim, _pointer, _point);                    \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(LONG, npy_long, _itemsize, _ndim, _pointer, _point);                  \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(LONGLONG, npy_longlong, _itemsize, _ndim, _pointer, _point);          \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(FLOAT, npy_float, _itemsize, _ndim, _pointer, _point);                \
        PYCV_T_PROBABILISTIC_LINE_CASE_SET_POINT(DOUBLE, npy_double, _itemsize, _ndim, _pointer, _point);              \
    }                                                                                                                  \
}

PyArrayObject *PYCV_hough_probabilistic_line(PyArrayObject *input,
                                             PyArrayObject *theta,
                                             npy_intp offset,
                                             npy_intp threshold,
                                             npy_intp line_length,
                                             npy_intp line_gap)
{
    int num_type_i, num_type_t, num_type_o;
    PYCV_ArrayIterator iter_i, iter_t;
    char *pi = NULL, *pi_base = NULL, *pt = NULL, *po = NULL;
    npy_intp ndim, init_size = 1, input_shape[NPY_MAXDIMS], coordinate[NPY_MAXDIMS], ndim_init, a_size, transform_size = 1;
    npy_intp *input_mask, *mask;

    npy_intp *h_space, n_theta, n_rho, max_hh, max_theta, y_stride, h_size;
    npy_double *cosine, *sine, i_val, y, x, proj, angle, cosine_abs, sine_abs;
    npy_intp ii, jj, hh, y0, x0, y1, x1, dy, dx, iyx, diy, dix, inter, gap, lx, ly, line[4] = {0, 0, 0, 0};

    PYCV_CoordinatesList line_start, line_end;
    PyArrayObject *output;
    npy_intp output_shape[3] = {2, -1, -1}, itemsize_o;

    NPY_BEGIN_THREADS_DEF;

    ndim = PyArray_NDIM(input);
    a_size = PyArray_SIZE(input);
    ndim_init = ndim - 2;
    output_shape[2] = ndim;

    for (ii = 0; ii < ndim; ii++) {
        input_shape[ii] = PyArray_DIM(input, (int)ii);
        if (ii < ndim_init) {
            init_size *= input_shape[ii];
        } else {
            transform_size *= input_shape[ii];
        }
        coordinate[ii] = 0;
    }

    n_theta = PyArray_SIZE(theta);
    n_rho = 2 * offset + 1;
    h_size = n_theta * n_rho;

    num_type_t = PyArray_TYPE(theta);
    num_type_i = PyArray_TYPE(input);

    PYCV_ArrayIteratorInit(input, &iter_i);
    PYCV_ArrayIteratorInit(theta, &iter_t);

    y_stride = iter_i.strides[ndim_init] / (npy_intp)PyArray_ITEMSIZE(input);

    cosine = malloc(n_theta * sizeof(npy_double));
    sine = malloc(n_theta * sizeof(npy_double));
    h_space = malloc(h_size * sizeof(npy_intp));
    input_mask = malloc(a_size * sizeof(npy_intp));

    if (!cosine || !sine || !h_space || !input_mask) {
        PyErr_NoMemory();
        goto exit;
    }

    if (!PYCV_CoordinatesListInit(ndim, a_size, &line_start)) {
        PyErr_NoMemory();
        goto exit;
    }
    if (!PYCV_CoordinatesListInit(ndim, a_size, &line_end)) {
        PyErr_NoMemory();
        goto exit;
    }

    pt = (void *)PyArray_DATA(theta);
    pi_base = pi = (void *)PyArray_DATA(input);

    for (ii = 0; ii < n_theta; ii++) {
        PYCV_GET_VALUE(num_type_t, npy_double, pt, angle);
        cosine[ii] = (npy_double)cos(angle);
        sine[ii] = (npy_double)sin(angle);
        PYCV_ARRAY_ITERATOR_NEXT(iter_t, pt);
    }
    for (ii = 0; ii < a_size; ii++) {
        PYCV_GET_VALUE(num_type_i, npy_double, pi, i_val);
        if (fabs(i_val) > DBL_EPSILON) {
            input_mask[ii] = 1;
        } else {
            input_mask[ii] = 0;
        }
        PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
    }
    pi = pi_base;
    PYCV_ARRAY_ITERATOR_RESET(iter_i);
    mask = input_mask;

    NPY_BEGIN_THREADS;

    while (init_size--) {
        for (jj = 0; jj < ndim_init; jj++) {
            coordinate[jj] = iter_i.coordinates[jj];
        }
        for (ii = 0; ii < h_size; ii++) {
            h_space[ii] = 0;
        }
        for (ii = 0; ii < transform_size; ii++) {
            PYCV_GET_VALUE(num_type_i, npy_double, pi, i_val);

            if (mask[ii]) {
                y = (npy_double)iter_i.coordinates[ndim_init];
                x = (npy_double)iter_i.coordinates[ndim_init + 1];
                max_hh = -1;
                max_theta = -1;
                for (jj = 0; jj < n_theta; jj++) {
                    proj = cosine[jj] * x + sine[jj] * y;
                    hh = (npy_intp)floor(proj + 0.5) + offset;
                    hh = hh * n_theta + jj;
                    h_space[hh]++;
                    if (max_hh < 0 || h_space[hh] > h_space[max_hh]) {
                        max_hh = hh;
                        max_theta = jj;
                    }
                }

                if (max_hh >= 0 && h_space[max_hh] > threshold) {
                    y0 = iter_i.coordinates[ndim_init];
                    x0 = iter_i.coordinates[ndim_init + 1];

                    cosine_abs = cosine[max_theta] < 0 ? -cosine[max_theta] : cosine[max_theta];
                    sine_abs = sine[max_theta] < 0 ? -sine[max_theta] : sine[max_theta];

                    if (sine_abs > cosine_abs) {
                        dx = -sine[max_theta] > 0 ? 1 : -1;
                        dy = (npy_intp)floor((cosine[max_theta] / sine_abs) + 0.5);
                    } else {
                        dy = cosine[max_theta] > 0 ? 1 : -1;
                        dx = (npy_intp)floor((-sine[max_theta] / cosine_abs) + 0.5);
                    }
                    dy *= -1;
                    dx *= -1;
                    diy = dy * y_stride;
                    dix = dx;
                    for (inter = 0; inter < 2; inter++) {
                        gap = 0;
                        dy *= -1;
                        dx *= -1;
                        y1 = y0;
                        x1 = x0;
                        iyx = ii;
                        while (1) {
                            if (y1 < 0 || y1 > iter_i.dims_m1[ndim_init] || x1 < 0 || x1 > iter_i.dims_m1[ndim_init + 1]) {
                                break;
                            }
                            gap += 1;
                            if (mask[iyx]) {
                                gap = 0;
                                line[2 * inter] = y1;
                                line[2 * inter + 1] = x1;
                            } else if (gap > line_gap) {
                                break;
                            }
                            x1 += dx;
                            y1 += dy;
                            iyx += (diy + dix);
                        }
                    }
                    ly = line[2] > line[0] ? line[2] - line[0] : line[0] - line[2];
                    lx = line[3] > line[1] ? line[3] - line[1] : line[1] - line[3];

                    if (ly >= line_length || lx >= line_length) {

                        for (inter = 0; inter < 2; inter++) {
                            dy *= -1;
                            dx *= -1;
                            y1 = y0;
                            x1 = x0;
                            iyx = ii;
                            while (1) {
                                if (mask[iyx]) {
                                    proj = cosine[max_theta] * (npy_double)x1 + sine[max_theta] * (npy_double)y1;
                                    hh = (npy_intp)floor(proj + 0.5) + offset;
                                    hh = hh * n_theta + max_theta;
                                    h_space[hh] -= 1;
                                    input_mask[iyx] = 0;
                                }
                                if (y1 == line[2 * inter] && x1 == line[2 * inter + 1]) {
                                    break;
                                }
                                x1 += dx;
                                y1 += dy;
                                iyx += (diy + dix);
                            }
                        }
                        coordinate[ndim_init] = line[0];
                        coordinate[ndim_init + 1] = line[1];
                        PYCV_COORDINATES_LIST_APPEND(line_start, coordinate);

                        coordinate[ndim_init] = line[2];
                        coordinate[ndim_init + 1] = line[3];
                        PYCV_COORDINATES_LIST_APPEND(line_end, coordinate);
                    }
                }
            }
            PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
        }
        if (ndim_init) {
            mask += (iter_i.dims_m1[ndim_init] + 1) * (iter_i.dims_m1[ndim_init + 1] + 1);
        }
    }
    NPY_END_THREADS;

    output_shape[1] = line_start.coordinates_size;
    output = (PyArrayObject *)PyArray_EMPTY(3, output_shape, NPY_INT64, 0);

    if (!output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_o = PyArray_ITEMSIZE(output);
    po = (void *)PyArray_DATA(output);
    num_type_o = PyArray_TYPE(output);

    for (ii = 0; ii < line_start.coordinates_size; ii++) {
        PYCV_T_PROBABILISTIC_LINE_SET_POINT(num_type_o, itemsize_o, ndim, po, line_start.coordinates[ii]);
    }
    for (ii = 0; ii < line_end.coordinates_size; ii++) {
        PYCV_T_PROBABILISTIC_LINE_SET_POINT(num_type_o, itemsize_o, ndim, po, line_end.coordinates[ii]);
    }

    exit:
        free(h_space);
        free(input_mask);
        free(cosine);
        free(sine);
        if (line_start.coordinates_size >= 0) {
            PYCV_CoordinatesListFree(&line_start);
        }
        if (line_end.coordinates_size >= 0) {
            PYCV_CoordinatesListFree(&line_end);
        }
        return PyErr_Occurred() ? NULL : output;
}

// #####################################################################################################################

static int integral_image_output_dtype(int inputs_numtype)
{
    int out = -1;
    switch (inputs_numtype) {
        case NPY_BOOL:
        case NPY_UBYTE:
        case NPY_USHORT:
        case NPY_UINT:
        case NPY_ULONG:
        case NPY_ULONGLONG:
            out = (int)NPY_ULONGLONG;
            break;
        case NPY_BYTE:
        case NPY_SHORT:
        case NPY_INT:
        case NPY_LONG:
        case NPY_LONGLONG:
            out = (int)NPY_LONGLONG;
            break;
        case NPY_FLOAT:
        case NPY_DOUBLE:
            out = (int)NPY_DOUBLE;
            break;
    }
    return out;
}

#define CASE_INTEGRAL_IMAGE_COPY_TO(_NTYPE, _dtype, _numtype, _from, _to)                                              \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    switch (_numtype) {                                                                                                \
        case NPY_ULONGLONG:                                                                                            \
            *(npy_ulonglong *)_to = (npy_ulonglong)(*(_dtype *)_from);                                                 \
            break;                                                                                                     \
        case NPY_LONGLONG:                                                                                             \
            *(npy_longlong *)_to = (npy_longlong)(*(_dtype *)_from);                                                   \
            break;                                                                                                     \
        case NPY_DOUBLE:                                                                                               \
            *(npy_double *)_to = (npy_double)(*(_dtype *)_from);                                                       \
            break;                                                                                                     \
    }                                                                                                                  \
}                                                                                                                      \
break;

#define INTEGRAL_IMAGE_COPY_T(_NTYPE, _numtype, _from, _to)                                                            \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        CASE_INTEGRAL_IMAGE_COPY_TO(BOOL, npy_bool, _numtype, _from, _to);                                             \
        CASE_INTEGRAL_IMAGE_COPY_TO(UBYTE, npy_ubyte, _numtype, _from, _to);                                           \
        CASE_INTEGRAL_IMAGE_COPY_TO(USHORT, npy_ushort, _numtype, _from, _to);                                         \
        CASE_INTEGRAL_IMAGE_COPY_TO(UINT, npy_uint, _numtype, _from, _to);                                             \
        CASE_INTEGRAL_IMAGE_COPY_TO(ULONG, npy_ulong, _numtype, _from, _to);                                           \
        CASE_INTEGRAL_IMAGE_COPY_TO(ULONGLONG, npy_ulonglong, _numtype, _from, _to);                                   \
        CASE_INTEGRAL_IMAGE_COPY_TO(BYTE, npy_byte, _numtype, _from, _to);                                             \
        CASE_INTEGRAL_IMAGE_COPY_TO(SHORT, npy_short, _numtype, _from, _to);                                           \
        CASE_INTEGRAL_IMAGE_COPY_TO(INT, npy_int, _numtype, _from, _to);                                               \
        CASE_INTEGRAL_IMAGE_COPY_TO(LONG, npy_long, _numtype, _from, _to);                                             \
        CASE_INTEGRAL_IMAGE_COPY_TO(LONGLONG, npy_longlong, _numtype, _from, _to);                                     \
        CASE_INTEGRAL_IMAGE_COPY_TO(FLOAT, npy_float, _numtype, _from, _to);                                           \
        CASE_INTEGRAL_IMAGE_COPY_TO(DOUBLE, npy_double, _numtype, _from, _to);                                         \
    }                                                                                                                  \
}

int PYCV_integral_image(PyArrayObject *inputs, PyArrayObject **output)
{
    int size, ndim, numtype, *dims_m1 = NULL, *strides = NULL, *strides_back = NULL, *coordinates = NULL;
    char *ptr_i = NULL, *ptr_o = NULL, *ptr = NULL;
    int jj, cc, finish_axis = 0;
    PYCV_ArrayIterator iterator;

    size = (int)PyArray_SIZE(inputs);
    ndim = (int)PyArray_NDIM(inputs);
    numtype = integral_image_output_dtype((int)PyArray_TYPE(inputs));

    if (numtype == -1) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid dtype");
        return 0;
    }

    dims_m1 = malloc(4 * ndim * sizeof(int));
    if (!dims_m1) {
        PyErr_NoMemory();
        return 0;
    }
    strides = dims_m1 + ndim;
    strides_back = strides + ndim;
    coordinates = strides_back + ndim;

    *output = (PyArrayObject *)PyArray_EMPTY(ndim, PyArray_DIMS(inputs), numtype, 0);

    if (!*output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array");
        goto exit;
    }

    for (jj = 0; jj < ndim; jj++) {
        *(dims_m1 + jj) = (int)PyArray_DIM(*output, jj) - 1;
        *(strides + jj) = (int)PyArray_STRIDE(*output, jj);
        *(strides_back + jj) = *(strides + jj) * *(dims_m1 + jj);
        *(coordinates + jj) = 0;
    }

    ptr_i = (void *)PyArray_DATA(inputs);
    ptr_o = ptr = (void *)PyArray_DATA(*output);

    PYCV_ArrayIteratorInit(inputs, &iterator);

    for (jj = 0; jj < size; jj++) {
        INTEGRAL_IMAGE_COPY_T(iterator.numtype, numtype, ptr_i, ptr);
        ptr += *(strides + ndim - 1);
        PYCV_ARRAY_ITERATOR_NEXT(iterator, ptr_i);
    }

    for (jj = 0; jj < ndim; jj++) {
        ptr = ptr_o;
        for (cc = 0; cc < ndim; cc++) {
            *(coordinates + cc) = 0;
        }

        *(coordinates + jj) = 1;
        ptr += *(strides + jj);
        finish_axis = 0;

        while (!finish_axis) {
            switch (numtype) {
                case NPY_ULONGLONG:
                    *(npy_ulonglong *)ptr += *(npy_ulonglong *)(ptr - *(strides + jj));
                    break;
                case NPY_LONGLONG:
                    *(npy_longlong *)ptr += *(npy_longlong *)(ptr - *(strides + jj));
                    break;
                case NPY_DOUBLE:
                    *(npy_double *)ptr += *(npy_double *)(ptr - *(strides + jj));
                    break;
            }

            for (cc = ndim - 1; cc >= 0; cc--) {
                if (*(coordinates + cc) < *(dims_m1 + cc)) {
                    *(coordinates + cc) += 1;
                    ptr += *(strides + cc);
                    break;
                } else {
                    *(coordinates + cc) = 0;
                    ptr -= *(strides_back + cc);
                    if (cc == 0) {
                        finish_axis = 1;
                    } else if (cc == jj) {
                        ptr += *(strides + jj);
                        *(coordinates + cc) = 1;
                    }
                }
            }
        }
    }

    exit:
        free(dims_m1);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################




















