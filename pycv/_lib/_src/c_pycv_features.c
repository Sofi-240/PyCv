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

//#define CASE_CORNER_FAST_BINARIZE(_ntype, _dtype, _x, _th, _h, _b, _offsets)
//case NPY_##_ntype:
//{
//    int _ii;
//    double _l_th, _h_th;
//    _l_th = (double)(*(_dtype *)_x) - _th;
//    _h_th = (double)(*(_dtype *)_x) + _th;
//    for (_ii = 0; _ii < 16; _ii++) {
//        *(_h + _ii) = (double)(*(_dtype *)(_x + *(_offsets + _ii));
//        *(_b + _ii) = 0;
//        if (*(_h + _ii) > _h_th) {
//            *(_b + _ii) = 1;
//        } else if (*(_h + _ii) < _l_th) {
//            *(_b + _ii) = -1;
//        }
//    }
//}
//break
//
//
//int PYCV_corner_FAST(PyArrayObject *input, int ncon, double threshold, PyArrayObject **response)
//{
//    int offsets[16] = {-21, -20, -12, -4, 3, 10, 16, 22, 21, 20, 12, 4, -3, -10, -16};
//
//}