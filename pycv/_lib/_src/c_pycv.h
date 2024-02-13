#ifndef C_PYCV_H
#define C_PYCV_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _c_pycv_ARRAY_API
#include <numpy/arrayobject.h>

// #####################################################################################################################

PyObject* convolve(PyObject* self, PyObject* args);

PyObject* rank_filter(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* binary_erosion(PyObject* self, PyObject* args);

PyObject* gray_erosion_dilation(PyObject* self, PyObject* args);

PyObject* binary_region_fill(PyObject* self, PyObject* args);

PyObject* labeling(PyObject* self, PyObject* args);

PyObject* skeletonize(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* resize(PyObject* self, PyObject* args);

PyObject* geometric_transform(PyObject* self, PyObject* args);

PyObject* hough_transform(PyObject* self, PyObject* args, PyObject* keywords);

// #####################################################################################################################

PyObject* canny_nonmaximum_suppression(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* build_max_tree(PyObject* self, PyObject* args);

PyObject* max_tree_compute_area(PyObject* self, PyObject* args);

PyObject* max_tree_filter(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* draw(PyObject* self, PyObject* args, PyObject* keywords);

// #####################################################################################################################

PyObject* convex_hull(PyObject* self, PyObject* args);

// #####################################################################################################################


#endif