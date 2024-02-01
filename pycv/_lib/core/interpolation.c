#include "ops_base.h"
#include "interpolation.h"

#define TOLERANCE 1e-15

// #####################################################################################################################

#define INTERP_CASE_SET_VALUE(_NUM_TYPE, _dtype, _pointer, _val)                                                       \
case _NUM_TYPE:                                                                                                        \
    *(_dtype *)_pointer = (_dtype)_val;                                                                                \
    break

#define INTERP_CASE_SET_VALUE_UINT(_NUM_TYPE, _dtype, _pointer, _val)                                                  \
case NPY_##_NUM_TYPE:                                                                                                  \
    _val = _val > 0 ? _val + 0.5 : 0;                                                                                  \
    _val = _val > NPY_MAX_##_NUM_TYPE ? NPY_MAX_##_NUM_TYPE : _val;                                                    \
    *(_dtype *)_pointer = (_dtype)_val;                                                                                \
    break

#define INTERP_CASE_SET_VALUE_INT(_NUM_TYPE, _dtype, _pointer, _val)                                                   \
case NPY_##_NUM_TYPE:                                                                                                  \
    _val = _val > 0 ? _val + 0.5 : _val - 0.5;                                                                         \
    _val = _val > NPY_MAX_##_NUM_TYPE ? NPY_MAX_##_NUM_TYPE : _val;                                                    \
    _val = _val < NPY_MIN_##_NUM_TYPE ? NPY_MIN_##_NUM_TYPE : _val;                                                    \
    *(_dtype *)_pointer = (_dtype)_val;                                                                                \
    break

#define INTERP_SET_VALUE_SAFE(_NUM_TYPE, _pointer, _val)                                                               \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        INTERP_CASE_SET_VALUE(NPY_BOOL, npy_bool, _pointer, _val);                                                     \
        INTERP_CASE_SET_VALUE_UINT(UBYTE, npy_ubyte, _pointer, _val);                                                  \
        INTERP_CASE_SET_VALUE_UINT(USHORT, npy_ushort, _pointer, _val);                                                \
        INTERP_CASE_SET_VALUE_UINT(UINT, npy_uint, _pointer, _val);                                                    \
        INTERP_CASE_SET_VALUE_UINT(ULONG, npy_ulong, _pointer, _val);                                                  \
        INTERP_CASE_SET_VALUE_UINT(ULONGLONG, npy_ulonglong, _pointer, _val);                                          \
        INTERP_CASE_SET_VALUE_INT(BYTE, npy_byte, _pointer, _val);                                                     \
        INTERP_CASE_SET_VALUE_INT(SHORT, npy_short, _pointer, _val);                                                   \
        INTERP_CASE_SET_VALUE_INT(INT, npy_int, _pointer, _val);                                                       \
        INTERP_CASE_SET_VALUE_INT(LONG, npy_long, _pointer, _val);                                                     \
        INTERP_CASE_SET_VALUE_INT(LONGLONG, npy_longlong, _pointer, _val);                                             \
        INTERP_CASE_SET_VALUE(NPY_FLOAT, npy_float, _pointer, _val);                                                   \
        INTERP_CASE_SET_VALUE(NPY_DOUBLE, npy_double, _pointer, _val);                                                 \
    }                                                                                                                  \
}

// #####################################################################################################################

#define INTERP_NN(_f, _d, _out)                                                                                        \
{                                                                                                                      \
    _out = _f[0];                                                                                                      \
}

#define INTERP_LINEAR(_f, _d, _out)                                                                                    \
{                                                                                                                      \
    _out = (1 - _d) * _f[0] + _d * _f[1];                                                                              \
}

#define INTERP_QUADRATIC(_f, _d, _out)                                                                                 \
{                                                                                                                      \
    _out = _f[1] + 0.5 * _d * (_f[2] - _f[0]) + 0.5 * _d * _d * (_f[2] - 2 * _f[1] + _f[0]);                           \
}

#define INTERP_CUBIC(_f, _d, _out)                                                                                     \
{                                                                                                                      \
    _out = _f[1] + 0.5 * _d * (-_f[0] + _f[2] + _d * (2 * _f[0] - 5 * _f[1] + 4 * _f[2] - _f[3] + _d * (-_f[0] + 3 * _f[1] - 3 * _f[2] + _f[3]))); \
}

// #####################################################################################################################


int ops_resize(PyArrayObject *input,
               PyArrayObject *output,
               npy_intp order,
               int grid_mode,
               BordersMode mode,
               double constant_value)
{
    npy_intp rank, strides[NPY_MAXDIMS], dims_i[NPY_MAXDIMS], dims_o[NPY_MAXDIMS], flag, array_size;
    npy_intp max_dims = 0, max_stride = 0, stride_pos;
    int num_type_i, num_type_o;
    npy_intp cc, cc_fit, c_size, fc_size, c_strides[NPY_MAXDIMS], c_count[NPY_MAXDIMS], c_shift[NPY_MAXDIMS], start_pos[NPY_MAXDIMS];
    npy_double *cn, fc = 0, d, scale_factor[NPY_MAXDIMS], proj[NPY_MAXDIMS], delta[NPY_MAXDIMS];
    int is_flag;
    char *pi_base = NULL, *pi = NULL, *po = NULL;
    ArrayIter iter_o;
    npy_intp ii, jj, nn, mm, kk;


    NPY_BEGIN_THREADS_DEF;

    rank = PyArray_NDIM(input);
    num_type_i = PyArray_TYPE(input);
    num_type_o = PyArray_TYPE(output);
    array_size = PyArray_SIZE(output);

    for (ii = 0; ii < rank; ii++) {
        dims_i[ii] = PyArray_DIM(input, ii);
        strides[ii] = PyArray_STRIDE(input, ii);

        dims_o[ii] = PyArray_DIM(output, ii);

        if (grid_mode) {
            scale_factor[ii] = (npy_double)(dims_i[ii] - 1) / (npy_double)(dims_o[ii] - 1);
        } else {
            scale_factor[ii] = (npy_double)dims_i[ii] / (npy_double)dims_o[ii];
        }

        max_dims = max_dims < dims_i[ii] ? dims_i[ii] : max_dims;
        stride_pos = strides[ii] < 0 ? -strides[ii] : strides[ii];
        max_stride = max_stride < stride_pos ? stride_pos : max_stride;

        c_count[ii] = 0;
        c_shift[ii] = 0;
    }

    flag = max_dims * max_stride + 1;

    c_size = order + 1;
    c_strides[rank - 1] = 1;

    for (ii = rank - 2; ii >= 0; ii--) {
        c_size *= (order + 1);
        c_strides[ii] = c_strides[ii + 1] * (order + 1);
    }

    cn = malloc(c_size * sizeof(npy_double));
    if (!cn) {
        PyErr_NoMemory();
        goto exit;
    }

    ArrayIterInit(output, &iter_o);

    NPY_BEGIN_THREADS;

    pi_base = pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    for (ii = 0; ii < array_size; ii++) {

        for (jj = 0; jj < rank; jj++) {
            proj[jj] = (npy_double)iter_o.coordinates[jj] * scale_factor[jj];
            if (order & 1) {
                start_pos[jj] = (npy_intp)floor(proj[jj]) - order / 2;
                delta[jj] = proj[jj] - (npy_intp)floor(proj[jj]);
            } else {
                start_pos[jj] = (npy_intp)floor(proj[jj] + 0.5) - order / 2;
                delta[jj] = proj[jj] - (npy_intp)floor(proj[jj] + 0.5);
            }
        }

        for (nn = 0; nn < c_size; nn++) {
            pi = pi_base;
            is_flag = 0;
            for (jj = 0; jj < rank; jj++) {
                cc = start_pos[jj] + c_shift[jj];
                if (!is_flag) {
                    cc_fit = fit_coordinate(cc, dims_i[jj], flag, mode);
                    if (cc_fit == flag) {
                        is_flag = 1;
                    } else {
                        pi += cc_fit * strides[jj];
                    }
                }
                c_count[jj] += 1;
                if (c_count[jj] == c_strides[jj]) {
                    c_count[jj] = 0;
                    if (c_shift[jj] == order) {
                        c_shift[jj] = 0;
                    } else {
                        c_shift[jj] += 1;
                    }
                }
            }

            if (is_flag) {
                cn[nn] = (npy_double)constant_value;
            } else {
                GET_VALUE_AS(num_type_i, npy_double, pi, cn[nn]);
            }

        }

        fc_size = c_size / (order + 1);
        for (jj = rank - 1; jj >= 0; jj--) {
            mm = 0;
            d = delta[jj];
            for (kk = 0; kk < fc_size; kk++) {
                nn = kk * (order + 1);
                switch (order) {
                    case 0:
                        INTERP_NN((cn + nn), d, fc);
                        break;
                    case 1:
                        INTERP_LINEAR((cn + nn), d, fc);
                        break;
                    case 2:
                        INTERP_QUADRATIC((cn + nn), d, fc);
                        break;
                    case 3:
                        INTERP_CUBIC((cn + nn), d, fc);
                        break;
                    default:
                        NPY_END_THREADS;
                        PyErr_SetString(PyExc_RuntimeError, "order is not supported");
                        goto exit;
                }
                cn[mm] = fc;
                mm++;
            }
            fc_size = mm / (order + 1);
        }
        INTERP_SET_VALUE_SAFE(num_type_o, po, cn[0]);
        ARRAY_ITER_NEXT(iter_o, po);
    }

    NPY_END_THREADS;
    exit:
        free(cn);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################

#define INTERP_DOT(_nd, _src, _h, _dst)                                                                                \
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
            _dst[_ii] = TOLERANCE;                                                                                     \
        }                                                                                                              \
    }                                                                                                                  \
}

#define INTERP_GEO_SWAP_YX(_coord)                                                                                     \
{                                                                                                                      \
    npy_double _tmp = _coord[0];                                                                                       \
    _coord[0] = _coord[1];                                                                                             \
    _coord[1] = _tmp;                                                                                                  \
}

int ops_geometric_transform(PyArrayObject *matrix,
                            PyArrayObject *input,
                            PyArrayObject *output,
                            PyArrayObject *src,
                            PyArrayObject *dst,
                            int order,
                            BordersMode mode,
                            double constant_value)
{
    npy_intp nd, rank_p1, rank, strides[NPY_MAXDIMS], dims[NPY_MAXDIMS], flag, src_size, h_size, pre_size;
    PyArrayObject *p_src, *p_dst;
    npy_intp max_dims = 0, max_stride = 0, stride_pos;
    int case_input_output, case_src_dst, num_type_s, num_type_d, num_type_m;
    int is_flag;
    char *src_base = NULL, *psrc = NULL, *pdst = NULL, *pm = NULL, *psrc_t = NULL;
    ArrayIter iter_s, iter_d, iter_m;
    npy_intp ii, jj, nn, mm, kk;

    npy_intp cc, cc_fit, c_size, fc_size, c_strides[NPY_MAXDIMS], c_count[NPY_MAXDIMS], c_shift[NPY_MAXDIMS], start_pos[NPY_MAXDIMS];
    npy_double *cn = NULL, fc = 0, d, delta[NPY_MAXDIMS];

    npy_double *h = NULL, *src_buffer = NULL, *dst_buffer = NULL;

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
        nd = PyArray_NDIM(input);
        p_src = input;
        p_dst = output;

        src_size = 1;
        pre_size = 1;

        for (ii = 0; ii < nd; ii++) {
            dims[ii] = PyArray_DIM(input, ii);
            strides[ii] = PyArray_STRIDE(input, ii);

            if (ii < nd - rank) {
                pre_size *= PyArray_DIM(output, ii);
            } else {
                src_size *= PyArray_DIM(output, ii);
            }
            max_dims = max_dims < dims[ii] ? dims[ii] : max_dims;
            stride_pos = strides[ii] < 0 ? -strides[ii] : strides[ii];
            max_stride = max_stride < stride_pos ? stride_pos : max_stride;

            c_count[ii] = 0;
            c_shift[ii] = 0;
        }

        flag = max_dims * max_stride + 1;

        c_size = order + 1;
        c_strides[rank - 1] = 1;

        for (ii = rank - 2; ii >= 0; ii--) {
            c_size *= (order + 1);
            c_strides[ii] = c_strides[ii + 1] * (order + 1);
        }

        cn = malloc(c_size * sizeof(npy_double));
        if (!cn) {
            PyErr_NoMemory();
            goto exit;
        }
    } else {
        pre_size = 1;
        src_size = PyArray_DIM(src, 0);
        p_src = src;
        p_dst = dst;
    }

    num_type_s = PyArray_TYPE(p_src);
    num_type_d = PyArray_TYPE(p_dst);
    num_type_m = PyArray_TYPE(matrix);

    h_size = PyArray_SIZE(matrix);

    h = malloc(h_size * sizeof(npy_double));
    if (!h) {
        PyErr_NoMemory();
        goto exit;
    }

    pm = (void *)PyArray_DATA(matrix);
    ArrayIterInit(matrix, &iter_m);

    for (ii = 0; ii < h_size; ii++) {
        GET_VALUE_AS(num_type_m, npy_double, pm, h[ii]);
        ARRAY_ITER_NEXT(iter_m, pm);
    }

    src_buffer = malloc((rank + 1) * sizeof(npy_double));
    if (!src_buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    dst_buffer = malloc((rank + 1) * sizeof(npy_double));
    if (!dst_buffer) {
        PyErr_NoMemory();
        goto exit;
    }

    ArrayIterInit(p_src, &iter_s);
    ArrayIterInit(p_dst, &iter_d);

    NPY_BEGIN_THREADS;

    src_base = psrc = (void *)PyArray_DATA(p_src);
    pdst = (void *)PyArray_DATA(p_dst);

    while (pre_size) {
        for (ii = 0; ii < src_size; ii++) {
            if (case_input_output) {

                for (jj = 0; jj < rank; jj++) {
                    src_buffer[jj] = (npy_double)iter_d.coordinates[nd - rank + jj];
                }
                src_buffer[rank] = 1;

                if (rank > 1) {
                    INTERP_GEO_SWAP_YX(src_buffer);
                }

                INTERP_DOT(rank_p1, src_buffer, h, dst_buffer);

                if (rank > 1) {
                    INTERP_GEO_SWAP_YX(dst_buffer);
                }

                for (jj = 0; jj < rank; jj++) {
                    if (order & 1) {
                        start_pos[jj] = (npy_intp)floor(dst_buffer[jj]) - order / 2;
                        delta[jj] = dst_buffer[jj] - (npy_intp)floor(dst_buffer[jj]);
                    } else {
                        start_pos[jj] = (npy_intp)floor(dst_buffer[jj] + 0.5) - order / 2;
                        delta[jj] = dst_buffer[jj] - (npy_intp)floor(dst_buffer[jj] + 0.5);
                    }
                }

                for (nn = 0; nn < c_size; nn++) {
                    psrc_t = psrc;
                    is_flag = 0;
                    for (jj = 0; jj < rank; jj++) {
                        cc = start_pos[jj] + c_shift[jj];
                        if (!is_flag) {
                            cc_fit = fit_coordinate(cc, dims[nd - rank + jj], flag, mode);
                            if (cc_fit == flag) {
                                is_flag = 1;
                            } else {
                                psrc_t += cc_fit * strides[nd - rank + jj];
                            }
                        }
                        c_count[jj] += 1;
                        if (c_count[jj] == c_strides[jj]) {
                            c_count[jj] = 0;
                            if (c_shift[jj] == order) {
                                c_shift[jj] = 0;
                            } else {
                                c_shift[jj] += 1;
                            }
                        }
                    }

                    if (is_flag) {
                        cn[nn] = (npy_double)constant_value;
                    } else {
                        GET_VALUE_AS(num_type_s, npy_double, psrc_t, cn[nn]);
                    }

                }

                fc_size = c_size / (order + 1);
                for (jj = rank - 1; jj >= 0; jj--) {
                    mm = 0;
                    d = delta[jj];
                    for (kk = 0; kk < fc_size; kk++) {
                        nn = kk * (order + 1);
                        switch (order) {
                            case 0:
                                INTERP_NN((cn + nn), d, fc);
                                break;
                            case 1:
                                INTERP_LINEAR((cn + nn), d, fc);
                                break;
                            case 2:
                                INTERP_QUADRATIC((cn + nn), d, fc);
                                break;
                            case 3:
                                INTERP_CUBIC((cn + nn), d, fc);
                                break;
                            default:
                                NPY_END_THREADS;
                                PyErr_SetString(PyExc_RuntimeError, "order is not supported");
                                goto exit;
                        }
                        cn[mm] = fc;
                        mm++;
                    }
                    fc_size = mm / (order + 1);
                }

                INTERP_SET_VALUE_SAFE(num_type_d, pdst, cn[0]);
                ARRAY_ITER_NEXT(iter_d, pdst);

            } else {
                for (jj = 0; jj <= rank; jj++) {
                    GET_VALUE_AS(num_type_s, npy_double, psrc, src_buffer[jj]);
                    ARRAY_ITER_NEXT(iter_s, psrc);
                }
                INTERP_DOT(rank_p1, src_buffer, h, dst_buffer);
                for (jj = 0; jj <= rank; jj++) {
                    SET_VALUE_TO(num_type_d, pdst, dst_buffer[jj]);
                    ARRAY_ITER_NEXT(iter_d, pdst);
                }
            }
        }

        if (case_input_output) {
            ARRAY_ITER_GOTO(iter_s, iter_d.coordinates, src_base, psrc);
        }
        pre_size--;
    }

    NPY_END_THREADS;
    exit:
        if (cn) {
            free(cn);
        }
        if (h) {
            free(h);
        }
        if (src_buffer) {
            free(src_buffer);
        }
        if (dst_buffer) {
            free(dst_buffer);
        }
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################