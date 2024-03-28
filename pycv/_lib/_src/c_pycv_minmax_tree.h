#ifndef C_PYCV_MINMAX_TREE_H
#define C_PYCV_MINMAX_TREE_H

// #####################################################################################################################

typedef struct {
    PyObject_HEAD
    int connectivity;
    int ndim;
    int size;
    PyObject *dims;
    PyArrayObject *data;
    PyArrayObject *traverser;
    PyArrayObject *nodes;
    int _is_max;
} CMinMaxTree;

void CMinMaxTreePy_dealloc(CMinMaxTree *self);

PyObject *CMinMaxTreePy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CMinMaxTreePy_init(CMinMaxTree *self, PyObject *args, PyObject *kw);

PyObject *CMinMaxTreePy_compute_area(CMinMaxTree *self);

PyObject *CMinMaxTreePy_tree_filter(CMinMaxTree *self, PyObject *args, PyObject *kw);

PyObject *CMinMaxTreePy_label_image(CMinMaxTree *self);

// #####################################################################################################################


#endif