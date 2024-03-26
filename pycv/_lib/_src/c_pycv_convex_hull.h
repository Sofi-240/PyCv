#ifndef C_PYCV_CONVEX_HULL_H
#define C_PYCV_CONVEX_HULL_H


// #####################################################################################################################

typedef enum {
    CHULL_GRAHAM_SCAN = 1,
    CHULL_JARVIS_MARCH = 2,
} CHull_Method;

// #####################################################################################################################

#define ch_dtype npy_longlong

#define ch_con_dtype NPY_INT64

#define ch_dtype_stride (int)NPY_SIZEOF_LONGLONG

// #####################################################################################################################

typedef struct {
    PyObject_HEAD
    int ndim;
    int n_vertices;
    PyArrayObject *points;
    PyArrayObject *vertices;
} CConvexHull;

void CConvexHullPy_dealloc(CConvexHull *self);

PyObject *CConvexHullPy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CConvexHullPy_init(CConvexHull *self, PyObject *args, PyObject *kw);

PyObject *CConvexHullPy_convex_to_image(CConvexHull *self, PyObject *args);

PyObject *CConvexHullPy_query_point(CConvexHull *self, PyObject *args);

// #####################################################################################################################

#endif
