#ifndef C_PYCV_CLUSTER_H
#define C_PYCV_CLUSTER_H

// #####################################################################################################################

typedef enum {
    CKMEANS_RANDOM = 1,
    CKMEANS_PLUSPLUS = 2,
} CKM_Initializer;

#define ckm_dtype npy_double

#define ckm_con_dtype NPY_DOUBLE

#define ckm_dtype_stride (int)NPY_SIZEOF_DOUBLE

#define ckm_itype npy_longlong

#define ckm_con_itype NPY_INT64

#define ckm_itype_stride (int)NPY_SIZEOF_LONGLONG

// #####################################################################################################################

typedef struct {
    PyObject_HEAD
    int k;
    int iterations;
    double tol;
    int n_samples;
    int ndim;
    int init_method;
    PyArrayObject *centers;
    PyArrayObject *data;
} CKMeans;

void CKMeansPy_dealloc(CKMeans *self);

PyObject *CKMeansPy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CKMeansPy_init(CKMeans *self, PyObject *args, PyObject *kw);

PyObject *CKMeansPy_fit(CKMeans *self, PyObject *args, PyObject *kw);

PyObject *CKMeansPy_predict(CKMeans *self, PyObject *args, PyObject *kw);

// #####################################################################################################################


#endif
