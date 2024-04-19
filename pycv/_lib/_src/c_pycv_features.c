#include "c_pycv_base.h"
#include "c_pycv_features.h"

// #####################################################################################################################

int PYCV_glcm(PyArrayObject *gray, PyArrayObject *distances, PyArrayObject *angle, int levels, PyArrayObject **glcm)
{
    PYCV_ArrayIterator iter_glcm, iter_ang, iter_dist;
    npy_intp glcm_dims[4] = {0, 0, 0, 0};
    char *p_ang = NULL, *p_dist = NULL, *p_g = NULL, *p_glcm = NULL;
    char *p_gc1_yy = NULL, *p_gc1_xx = NULL, *p_gc2_yy = NULL, *p_gc2_xx = NULL, *p_glcm_ij = NULL;
    int nd, na, ny, nx, sty, stx, dd, aa, yy, xx, shift_y, shift_x, b1y, b1x, b2y, b2x, i, j, numtype;
    double *cosine = NULL, *sine = NULL, ang, dist;

    nd = (int)PyArray_SIZE(distances);
    na = (int)PyArray_SIZE(angle);

    numtype = (int)PyArray_TYPE(gray);

    ny = (int)PyArray_DIM(gray, 0);
    nx = (int)PyArray_DIM(gray, 1);

    sty = (int)PyArray_STRIDE(gray, 0);
    stx = (int)PyArray_STRIDE(gray, 1);

    *glcm_dims = (npy_intp)levels;
    *(glcm_dims + 1) = (npy_intp)levels;
    *(glcm_dims + 2) = (npy_intp)nd;
    *(glcm_dims + 3) = (npy_intp)na;

    *glcm = (PyArrayObject *)PyArray_ZEROS(4, glcm_dims, NPY_INT64, 0);
    if (!*glcm) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return 0;
    }

    PYCV_ArrayIteratorInit(*glcm, &iter_glcm);
    PYCV_ArrayIteratorInit(angle, &iter_ang);
    PYCV_ArrayIteratorInit(distances, &iter_dist);

    p_ang = (void *)PyArray_DATA(angle);
    p_dist = (void *)PyArray_DATA(distances);
    p_g = (void *)PyArray_DATA(gray);
    p_glcm = (void *)PyArray_DATA(*glcm);

    cosine = malloc(na * 2 * sizeof(double));
    if (!cosine) {
        PyErr_NoMemory();
        return 0;
    }
    sine = cosine + na;

    for (aa = 0; aa < na; aa++) {
        PYCV_GET_VALUE(iter_ang.numtype, double, p_ang, ang);
        *(sine + aa) = sin(ang);
        *(cosine + aa) = cos(ang);
        PYCV_ARRAY_ITERATOR_NEXT(iter_ang, p_ang);
    }

    for (dd = 0; dd < nd; dd++) {
        PYCV_GET_VALUE(iter_dist.numtype, double, p_dist, dist);

        for (aa = 0; aa < na; aa++) {
            shift_y = (int)round((*(sine + aa) * dist));
            shift_x = (int)round((*(cosine + aa) * dist));

            b1y = 0 > -shift_y ? 0 : -shift_y;
            b1x = 0 > -shift_x ? 0 : -shift_x;

            b2y = ny < (ny - shift_y) ? ny : (ny - shift_y);
            b2x = nx < (nx - shift_x) ? nx : (nx - shift_x);

            for (yy = b1y; yy < b2y; yy++) {
                p_gc1_yy = p_g + sty * yy;
                p_gc2_yy = p_g + sty * (yy + shift_y);

                for (xx = b1x; xx < b2x; xx++) {
                    p_gc1_xx = p_gc1_yy + stx * xx;
                    p_gc2_xx = p_gc2_yy + stx * (xx + shift_x);

                    PYCV_GET_VALUE(numtype, int, p_gc1_xx, i);
                    PYCV_GET_VALUE(numtype, int, p_gc2_xx, j);

                    if ((i >= 0 && i < levels) && (j >= 0 && j < levels)) {
                        p_glcm_ij = p_glcm + ((int)(*iter_glcm.strides) * i) + ((int)(*(iter_glcm.strides + 1)) * j);
                        *(npy_longlong *)p_glcm_ij += 1;
                    }
                }
            }

            PYCV_ARRAY_ITERATOR_NEXT(iter_glcm, p_glcm);
        }
        PYCV_ARRAY_ITERATOR_NEXT(iter_dist, p_dist);
    }

    free(cosine);
    return 1;
}

// #####################################################################################################################

int PYCV_corner_FAST(PyArrayObject *input, int ncon, double threshold, PyArrayObject **response)
{
//    const int row_offsets[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
//    const int col_offsets[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};

    int offsets[32] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    int bins[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double h[16] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

    int size, ndim, dim0, ii, jj, kind, con, c_p1, c_m1;
    double lo, hi, vi, vo;
    PYCV_ArrayIterator iter_i, iter_r;
    char *ptr_i = NULL, *ptr_r = NULL, *ptr_n = NULL;

    size = (int)PyArray_SIZE(input);
    PYCV_ArrayIteratorInit(input, &iter_i);

    ndim = (int)iter_i.nd_m1 + 1;
    dim0 = ndim - 2;

    *response = (PyArrayObject *)PyArray_ZEROS(ndim, PyArray_DIMS(input), NPY_DOUBLE, 0);
    if (!*response) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return 0;
    }

    if (ndim < 2 || *(iter_i.dims_m1 + ndim - 1) < 6 || *(iter_i.dims_m1 + ndim - 2) < 6) {
        return 1;
    }

    for (jj = 0; jj < 16; jj++) {
        *(offsets + jj) = *(offsets + jj) * ((int)*(iter_i.strides + dim0)) +
                          *(offsets + jj + 16) * ((int)*(iter_i.strides + dim0 + 1));
    }

    PYCV_ArrayIteratorInit(*response, &iter_r);

    ptr_i = (void *)PyArray_DATA(input);
    ptr_r = (void *)PyArray_DATA(*response);

    size = 1;
    for (jj = 0; jj < ndim; jj++) {
        if (jj < dim0) {
            size *= ((int)(*(iter_i.dims_m1 + jj)) + 1);
        } else {
            size *= ((int)(*(iter_i.dims_m1 + jj)) - 5);
        }
    }


    for (jj = ndim - 1; jj >= dim0; jj--) {
        *(iter_i.coordinates + jj) = 3;
        *(iter_r.coordinates + jj) = 3;
        ptr_i += 3 * *(iter_i.strides + jj);
        ptr_r += 3 * *(iter_r.strides + jj);
    }

    for (ii = 0; ii < size; ii++) {
        PYCV_GET_VALUE(iter_i.numtype, double, ptr_i, vi);
        lo = vi - threshold;
        hi = vi + threshold;

        c_p1 = 0;
        c_m1 = 0;
        for (jj = 0; jj < 16; jj++) {
            ptr_n = ptr_i + *(offsets + jj);
            PYCV_GET_VALUE(iter_i.numtype, double, ptr_n, *(h + jj));

            if (*(h + jj) > hi) {
                *(bins + jj) = 1;
                c_p1++;
            } else if (*(h + jj) < lo) {
                *(bins + jj) = -1;
                c_m1++;
            } else {
                *(bins + jj) = 0;
            }
        }

        if (c_p1 >= ncon || c_m1 >= ncon) {
            kind = 1;

            do {
                con = 0;
                for (jj = 0; jj < (ncon + 15); jj++) {
                    if (*(bins + (jj % 16)) == kind) {
                        con++;
                        if (con == ncon) {
                            break;
                        }
                    } else {
                        con = 0;
                    }
                }
                kind = -kind;
                con = con == ncon ? con : 0;
            } while (!con && kind != 1);

            if (con) {
                vo = 0;
                for (jj = 0; jj < 16; jj++) {
                    vo += abs(*(h + jj) - vi);
                }
                PYCV_SET_VALUE(iter_r.numtype, ptr_r, vo);
            }
        }

        for (jj = ndim - 1; jj >= 0; jj--) {
            if (*(iter_i.coordinates + jj) < (*(iter_i.dims_m1 + jj) - (jj < dim0 ? 0 : 3))) {
                *(iter_i.coordinates + jj) += 1;
                *(iter_r.coordinates + jj) += 1;
                ptr_i += *(iter_i.strides + jj);
                ptr_r += *(iter_r.strides + jj);
                break;
            } else {
                if (jj < dim0) {
                    ptr_i -= *(iter_i.strides_back + jj);
                    ptr_r -= *(iter_r.strides_back + jj);
                    *(iter_i.coordinates + jj) = 0;
                    *(iter_r.coordinates + jj) = 0;
                } else {
                    ptr_i += (3 * *(iter_i.strides + jj) - *(iter_i.strides_back + jj));
                    ptr_r += (3 * *(iter_r.strides + jj) - *(iter_r.strides_back + jj));
                    *(iter_i.coordinates + jj) = 3;
                    *(iter_r.coordinates + jj) = 3;
                    ptr_i += 3 * *(iter_i.strides + jj);
                    ptr_r += 3 * *(iter_r.strides + jj);
                }
            }
        }
    }
    return 1;
}