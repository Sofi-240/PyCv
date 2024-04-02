#ifndef C_PYCV_HAAR_LIKE_H
#define C_PYCV_HAAR_LIKE_H

// #####################################################################################################################

typedef enum {
    HAAR_EDGE = 1,
    HAAR_LINE = 2,
    HAAR_DIAG = 3,
} HaarType;

// #####################################################################################################################

typedef struct {
    PyObject_HEAD
    PyObject *htype;
    int ndim;
    int n_rect;
    int n_features;
    PyObject *axis;
    PyObject *feature_dims;
    int _htype;
} CHaarFeatures;

void CHaarFeaturesPy_dealloc(CHaarFeatures *self);

PyObject *CHaarFeaturesPy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CHaarFeaturesPy_init(CHaarFeatures *self, PyObject *args);

PyObject *CHaarFeaturesPy_coordinates(CHaarFeatures *self, PyObject *args, PyObject *kw);

PyObject *CHaarFeaturesPy_like_features(CHaarFeatures *self, PyObject *args, PyObject *kw);

// #####################################################################################################################


#endif

