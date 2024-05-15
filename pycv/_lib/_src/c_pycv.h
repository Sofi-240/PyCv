#ifndef C_PYCV_H
#define C_PYCV_H

#include <Python.h>
#include "structmember.h"
#include <stddef.h>

#define PY_ARRAY_UNIQUE_SYMBOL _c_pycv_ARRAY_API
#include <numpy/arrayobject.h>

// #####################################################################################################################

PyObject* convolve(PyObject* self, PyObject* args);

PyObject* rank_filter(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* binary_erosion(PyObject* self, PyObject* args);

PyObject* gray_erosion(PyObject* self, PyObject* args);

PyObject* binary_region_fill(PyObject* self, PyObject* args);

PyObject* labeling(PyObject* self, PyObject* args);

PyObject* skeletonize(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* resize(PyObject* self, PyObject* args);

PyObject* geometric_transform(PyObject* self, PyObject* args);

PyObject* hough_transform(PyObject* self, PyObject* args, PyObject* keywords);

PyObject* linear_interp1D(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* canny_nonmaximum_suppression(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* draw(PyObject* self, PyObject* args, PyObject* keywords);

// #####################################################################################################################

PyObject* integral_image(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* peak_nonmaximum_suppression(PyObject* self, PyObject* args);

// #####################################################################################################################

PyObject* gray_co_occurrence_matrix(PyObject* self, PyObject* args);

PyObject* corner_FAST(PyObject* self, PyObject* args);

// #####################################################################################################################

PyTypeObject CKDnode_Type;

PyTypeObject CKDtree_Type;

PyTypeObject CConvexHull_Type;

PyTypeObject CKMeans_Type;

PyTypeObject CMinMaxTree_Type;

PyTypeObject CHaarFeatures_Type;

PyTypeObject CLayer_Type;

// #####################################################################################################################


#endif