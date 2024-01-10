#ifndef OPS_H
#define OPS_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _pycv_ARRAY_API
#include <numpy/arrayobject.h>

PyObject* convolve(PyObject* self, PyObject* args);
PyObject* binary_erosion(PyObject* self, PyObject* args);
PyObject* rank_filter(PyObject* self, PyObject* args);

#endif