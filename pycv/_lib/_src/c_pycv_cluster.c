#include "c_pycv_base.h"
#include "c_pycv_cluster.h"

// #####################################################################################################################

static double ckm_minkowski_distance_p1(double v, double pnorm, int is_inf)
{
    v = v < 0 ? -v : v;
    return (is_inf || pnorm == 1) ? v : pow(v, pnorm);
}

static double ckm_minkowski_distance_p1p2_double(double *p1, double *p2, int ndim, double pnorm, int is_inf)
{
    double v = 0, vi;
    int ii;
    for (ii = 0; ii < ndim; ii++) {
        vi = ckm_minkowski_distance_p1(*(p2 + ii) - *(p1 + ii), pnorm, is_inf);
        if (is_inf && vi > v) {
            v = vi;
        } else {
            v += vi;
        }
    }
    return v;
}

static double ckm_minkowski_distance_p1p2_char(char *p1, char *p2, int ndim, double pnorm, int is_inf)
{
    char *pp1 = p1, *pp2 = p2;
    double v = 0, vi;

    while (ndim--) {
        vi = ckm_minkowski_distance_p1((double)(*((ckm_dtype *)pp2) - *((ckm_dtype *)pp1)), pnorm, is_inf);
        if (is_inf && vi > v) {
            v = vi;
        } else {
            v += vi;
        }
        pp1 += ckm_dtype_stride;
        pp2 += ckm_dtype_stride;
    }
    return v;
}

static double ckm_minkowski_distance_p1p2_char_double(char *p1, double *p2, int ndim, double pnorm, int is_inf)
{
    char *pp1 = p1;
    double v = 0, vi, *pp2 = p2;

    while (ndim--) {
        vi = ckm_minkowski_distance_p1((double)(*pp2 - *((ckm_dtype *)pp1)), pnorm, is_inf);
        if (is_inf && vi > v) {
            v = vi;
        } else {
            v += vi;
        }
        pp1 += ckm_dtype_stride;
        pp2++;
    }
    return v;
}

// #####################################################################################################################

#define ckm_get_random(_l, _h) (rand() % ((_h + 1) - _l)) + _l

// #####################################################################################################################

static int ckm_input_data(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    if (*output && PyArray_TYPE(*output) != ckm_con_dtype) {
        PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be float64 \n");
       *output = NULL;
    }
    if (*output != NULL) {
        *output = (PyArrayObject *)PyArray_GETCONTIGUOUS(*output);
    }
    return *output != NULL;
}

// #####################################################################################################################

typedef struct {
    int label;
    int size;
    double *center;
    double *prev_center;
} ckmeans_cluster;

static int ckmeans_cluster_init(ckmeans_cluster *self, int ndim, int label)
{
    self->label = label;
    self->size = 0;
    self->center = calloc(ndim * 2, sizeof(double));
    if (!self->center) {
        self->size = -1;
        PyErr_NoMemory();
        return 0;
    }
    self->prev_center = self->center + ndim;
    return 1;
}

static void ckmeans_cluster_free(ckmeans_cluster *self)
{
    if (self->size >= 0) {
        free(self->center);
    }
}

static void ckmeans_cluster_add_point(ckmeans_cluster *self, int n, int ndim, char *dptr)
{
    double *center = self->center, div = (double)n;
    char *ptr = dptr;

    while (ndim--) {
        *center++ += ((double)(*((ckm_dtype *)ptr)) / div);
        ptr += ckm_dtype_stride;
    }
    self->size++;
}

static void ckmeans_cluster_remove_point(ckmeans_cluster *self, int n, int ndim, char *dptr)
{
    double *center = self->center, div = (double)n;
    char *ptr = dptr;

    while (ndim--) {
        *center++ -= ((double)(*((ckm_dtype *)ptr)) / div);
        ptr += ckm_dtype_stride;
    }
    self->size--;
}

static double ckmeans_cluster_set_state(ckmeans_cluster *self, int n, int ndim, double pnorm, int is_inf)
{
    double *center = self->center, *prev_center = self->prev_center, err = 0, e;
    double mul = (double)n, div = (double)(self->size);
    while (ndim--) {
        e = *prev_center;
        *prev_center = ((*center * mul) / div);
        err += ckm_minkowski_distance_p1(e - *prev_center, pnorm, is_inf);
        prev_center++;
        *center++ = 0;
    }
    self->size = 0;
    return err;
}

typedef struct {
    int index;
    double distance;
    ckmeans_cluster *cluster;
} ckmeans_point;

static void ckmeans_point_swap(ckmeans_point *i1, ckmeans_point *i2)
{
    ckmeans_point tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

typedef struct {
    int k;
    int _k_in;
    ckmeans_cluster *clusters;
    ckmeans_point *points;
} ckmeans_clustering;

static void ckmeans_clustering_free(ckmeans_clustering *self)
{
    if (self->k >= 0) {
        free(self->points);
    }
    for (int ii = 0; ii < self->_k_in; ii++) {
        ckmeans_cluster_free(self->clusters + ii);
    }
    if (self->k >= 0) {
        free(self->clusters);
    }
}

static int ckmeans_clustering_init(ckmeans_clustering *self, int k, int n)
{
    self->k = k;
    self->_k_in = 0;
    self->clusters = malloc(k * sizeof(ckmeans_cluster));
    self->points = malloc(n * sizeof(ckmeans_point));
    if (!self->clusters || !self->points) {
        if (self->points) {
            free(self->points);
        } else {
            free(self->clusters);
        }
        self->k = -1;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static int ckmeans_clustering_add_cluster(ckmeans_clustering *self, int ndim)
{
    ckmeans_cluster *c = self->clusters + self->_k_in;
    self->_k_in++;
    if (!ckmeans_cluster_init(c, ndim, self->_k_in - 1)) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

// #####################################################################################################################

static int ckmeans_initialize(CKMeans *self, ckmeans_clustering *clustering, double pnorm, int is_inf)
{
    char *dptr = NULL, *ptr0 = NULL, *ptri = NULL;
    ckmeans_point *p = NULL, *p0 = NULL, *pi = NULL;
    int ii, jc, k = self->k, n = self->n_samples, m = self->ndim, kin = 0;
    int is_plusplus = self->init_method == CKMEANS_PLUSPLUS ? 1 : 0;
    ckmeans_cluster *ci = NULL;
    double vi;
    time_t _seed;

    srand((unsigned)time(&_seed));

    if (!ckmeans_clustering_init(clustering, k, n)) {
        PyErr_NoMemory();
        return 0;
    }

    p0 = p = clustering->points;
    dptr = (void *)PyArray_DATA(self->data);

    while (k--) {
        if (!ckmeans_clustering_add_cluster(clustering, self->ndim)) {
            PyErr_NoMemory();
            return 0;
        }
        ci = clustering->clusters + kin;
        jc = (kin && is_plusplus) ? ci->label : ckm_get_random(ci->label, n - 1);

        if (jc != kin) {
            ckmeans_point_swap(p0, p + jc);
        }

        if (!kin) {
            p0->index = jc;
            p0->distance = 0;
        }

        p0->cluster = clustering->clusters + kin;
        ptr0 = dptr + p0->index * m * ckm_dtype_stride;
        ckmeans_cluster_add_point(ci, n, m, ptr0);
        pi = p0 + 1;

        for (ii = !kin ? 0 : kin + 1; ii < n; ii++) {
            if (!ci->label && ii != jc) {
                pi->index = ii;
            } else if (!ci->label) {
                continue;
            }

            ptri = dptr + pi->index * m * ckm_dtype_stride;
            vi = ckm_minkowski_distance_p1p2_char(ptr0, ptri, m, pnorm, is_inf);

            if (!kin || vi < pi->distance) {
                pi->distance = vi;
                pi->cluster = ci;
            }
            if (kin + 1 == self->k) {
                ckmeans_cluster_add_point(pi->cluster, n, m, ptri);
            }

            if (is_plusplus && pi->distance > (p0 + 1)->distance) {
                ckmeans_point_swap(pi, p0 + 1);
            }
            pi++;
        }
        p0++;
        kin++;
    }
    return 1;
}

static void ckmeans_train(CKMeans *self, ckmeans_clustering *clustering, double pnorm, int is_inf)
{
    char *dptr = NULL, *ptri = NULL;
    ckmeans_point *pi = NULL;
    ckmeans_cluster *ci = NULL;
    int ii, jj, k = self->k, n = self->n_samples, m = self->ndim, iter = self->iterations, go_exit;
    double vi, epsilon, error;

    if (self->tol == 0) {
        epsilon = 0;
    } else if (is_inf) {
        epsilon = self->tol / (1 + self->tol);
    } else {
        epsilon = self->tol / pow((1 + self->tol), pnorm);
    }
    dptr = (void *)PyArray_DATA(self->data);

    do {
        go_exit = 1;
        for (ii = 0; ii < k; ii++) {
            ci = clustering->clusters + ii;
            error = ckmeans_cluster_set_state(ci, self->n_samples, self->ndim, pnorm, is_inf);
            go_exit = go_exit && (error < epsilon) ? 1 : 0;

            pi = clustering->points;
            for (jj = 0; jj < n; jj++) {
                ptri = dptr + pi->index * m * ckm_dtype_stride;
                vi = ckm_minkowski_distance_p1p2_char_double(ptri, ci->prev_center, m, pnorm, is_inf);

                if (!ii || vi < pi->distance) {
                    pi->distance = vi;
                    if (ii) {
                        ckmeans_cluster_remove_point(pi->cluster, n, m, ptri);
                    }
                    pi->cluster = ci;
                    ckmeans_cluster_add_point(pi->cluster, n, m, ptri);
                }
                pi++;
            }
        }
    } while (!go_exit && iter--);
}

static void ckmeans_predict(CKMeans *self, char *points, char *output, int n, double pnorm, int is_inf)
{
    char *ptri = points, *ptro = output, *cptr = NULL, *cptri = NULL;
    int ii, jj, k = self->k, m = self->ndim;
    double vi, v;

    cptr = (void *)PyArray_DATA(self->centers);


    for (jj = 0; jj < n; jj++) {
        cptri = cptr;
        v = ckm_minkowski_distance_p1p2_char(ptri, cptri, m, pnorm, is_inf);
        cptri += m * ckm_dtype_stride;
        *(npy_longlong *)ptro = 0;
        for (ii = 1; ii < k; ii++) {

            vi = ckm_minkowski_distance_p1p2_char(cptri, ptri, m, pnorm, is_inf);
            if (vi < v) {
                *(npy_longlong *)ptro = (npy_longlong)ii;
                v = vi;
            }
            cptri += m * ckm_dtype_stride;
        }
        ptro += ckm_itype_stride;
        ptri += m * ckm_dtype_stride;
    }
}

// #####################################################################################################################

void CKMeansPy_dealloc(CKMeans *self)
{
    Py_XDECREF(self->centers);
    Py_XDECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *CKMeansPy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CKMeans *self;
    self = (CKMeans *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->k = 4;
        self->iterations = 300;
        self->tol = 0.00001;
        self->n_samples = 0;
        self->ndim = 0;
        self->init_method = (int)CKMEANS_PLUSPLUS;

        self->centers = NULL;
        self->data = NULL;
    }

    return (PyObject *)self;
}

int CKMeansPy_init(CKMeans *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"k", "iterations", "tol", "init_method", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "|iidi", kwlist,
            &(self->k), &(self->iterations), &(self->tol), &(self->init_method))
        ) {
        return -1;
    }
    return 0;
}

PyObject *CKMeansPy_fit(CKMeans *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"", "pnorm", NULL};
    double pnorm = 2, *prev_center;
    int is_inf, ii, jj;
    ckmeans_clustering clustering = {0, 0, NULL, NULL};
    ckmeans_point *pi = NULL;
    ckmeans_cluster *ci = NULL;
    npy_intp dims_centers[2] = {0, 0};
    char *cc = NULL;

    Py_XDECREF(self->centers);
    Py_XDECREF(self->data);

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O&|d", kwlist,ckm_input_data, &(self->data), &pnorm)) {
        goto exit;
    }

    if (PyArray_NDIM(self->data) != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Error: data need to be 2D array (N features, features dim)");
        goto exit;
    }

    self->n_samples = (int)PyArray_DIM(self->data, 0);
    self->ndim = (int)PyArray_DIM(self->data, 1);

    if (self->n_samples < self->k) {
        PyErr_SetString(PyExc_RuntimeError, "Error: k is higher then the given n data points");
        goto exit;
    }

    is_inf = (float)pnorm > FLT_MAX ? 1 : 0;

    dims_centers[0] = self->k;
    dims_centers[1] = self->ndim;

    self->centers = (PyArrayObject *)PyArray_EMPTY(2, dims_centers, ckm_con_dtype, 0);

    if (!self->centers) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array");
        goto exit;
    }

    if (!ckmeans_initialize(self, &clustering, pnorm, is_inf)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: ckmeans_initialize");
        goto exit;
    }

    ckmeans_train(self, &clustering, pnorm, is_inf);

    cc = (void *)PyArray_DATA(self->centers);

    ci = clustering.clusters;

    for (ii = 0; ii < self->k; ii++) {
        prev_center = ci->prev_center;
        for (jj = 0; jj < self->ndim; jj++) {
            *(ckm_dtype *)cc = (ckm_dtype)(*prev_center++);
            cc += ckm_dtype_stride;
        }
        ci++;
    }

    exit:
        ckmeans_clustering_free(&clustering);
        return Py_BuildValue("");
}

PyObject *CKMeansPy_predict(CKMeans *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"", "pnorm", NULL};
    int is_inf;
    PyArrayObject *y, *x;
    npy_intp dims_y[1] = {0};
    char *py = NULL, *px = NULL;
    double pnorm = 2;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O&|d", kwlist,ckm_input_data, &x, &pnorm)) {
        goto exit;
    }

    if (PyArray_NDIM(x) != 2 || PyArray_DIM(x, 1) != self->ndim) {
        PyErr_SetString(PyExc_RuntimeError, "Error: x need to be 2D array (N features, features dim)");
        goto exit;
    }

    is_inf = (float)pnorm > FLT_MAX ? 1 : 0;

    dims_y[0] = PyArray_DIM(x, 0);
    y = (PyArrayObject *)PyArray_EMPTY(1, dims_y, ckm_con_itype, 0);

    if (!y) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array");
        goto exit;
    }
    px = (void *)PyArray_DATA(x);
    py = (void *)PyArray_DATA(y);
    ckmeans_predict(self, px, py, (int)PyArray_DIM(x, 0), pnorm, is_inf);

    exit:
        return PyErr_Occurred() ? Py_BuildValue("") : (PyObject *)y;
}

// #####################################################################################################################











