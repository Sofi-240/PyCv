#ifndef OPS_H
#define OPS_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _pycv_ARRAY_API
#include <numpy/arrayobject.h>

PyObject* convolve(PyObject* self, PyObject* args);

PyObject* rank_filter(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* binary_erosion(PyObject* self, PyObject* args);

PyObject* erosion(PyObject* self, PyObject* args);

PyObject* dilation(PyObject* self, PyObject* args);

PyObject* binary_region_fill(PyObject* self, PyObject* args);

PyObject* labeling(PyObject* self, PyObject* args);

PyObject* skeletonize(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* canny_nonmaximum_suppression(PyObject* self, PyObject* args);

PyObject* build_max_tree(PyObject* self, PyObject* args);

PyObject* area_threshold(PyObject* self, PyObject* args);

PyObject* draw(PyObject* self, PyObject* args, PyObject* keywords);

PyObject* hough_transform(PyObject* self, PyObject* args, PyObject* keywords);

// #####################################################################################################################

PyObject* resize(PyObject* self, PyObject* args);

PyObject* geometric_transform(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* convex_hull(PyObject* self, PyObject* args);

// #####################################################################################################################

#endif