#ifndef C_PYCV_PYRAMIDS_H
#define C_PYCV_PYRAMIDS_H

#include "c_pycv_pyr_utils.h"
#include "c_pycv_interpolation.h"

// ################################################ IO #################################################################

#define IO_object_op(_type, _op) _type##_op

#define IO_object_size(_type, _obj) (int)IO_object_op(Py##_type, _Size)(_obj)

#define IO_object_get_item(_type, _obj, _pos) IO_object_op(Py##_type, _GetItem)(_obj, (Py_ssize_t)_pos)

#define IO_object_parse(_obj, _as, _out) PyArg_Parse(_obj, _as, &_out)

#define IO_object_new(_type, _len) IO_object_op(Py##_type, _New)((Py_ssize_t)_len)

#define IO_object_set_item(_type, _obj, _pos, _item) IO_object_op(Py##_type, _SetItem)(_obj, (Py_ssize_t)_pos, _item)

#define IO_object_SET_ITEM(_type, _obj, _pos, _item) IO_object_op(Py##_type, _SET_ITEM)(_obj, (Py_ssize_t)_pos, _item)

#define IO_object_build(_as, _item) Py_BuildValue(_as, _item)

#define IO_object_check(_type, _obj) IO_object_op(Py##_type, _Check)(_obj)

// ###################################### CPyramid iterator ############################################################

typedef struct {
    PyObject_HEAD
    Layer *layer;
} CLayer;

// ********************************************** build ****************************************************************

void CLayerPy_dealloc(CLayer *self);

PyObject *CLayerPy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CLayerPy_init(CLayer *self, PyObject *args, PyObject *kw);

// ********************************************* setter ****************************************************************

int CLayer_set_input_dtype(CLayer *self, PyObject *descr);

int CLayer_set_padding_mode(CLayer *self, PyObject *extend_mode);

int CLayer_set_order(CLayer *self, PyObject *order);

int CLayer_set_scalespace(CLayer *self, PyObject *scalespace);

int CLayer_set_factors(CLayer *self, PyObject *factors);

int CLayer_set_input_dims(CLayer *self, PyObject *dims);

int CLayer_set_cval(CLayer *self, PyObject *cval);

// ********************************************* getter ****************************************************************

PyObject *CLayer_get_ndim(CLayer *self);

PyObject *CLayer_get_input_dtype(CLayer *self);

PyObject *CLayer_get_nscales(CLayer *self);

PyObject *CLayer_get_padding_mode(CLayer *self);

PyObject *CLayer_get_order(CLayer *self);

PyObject *CLayer_get_scalespace(CLayer *self);

PyObject *CLayer_get_factors(CLayer *self);

PyObject *CLayer_get_anti_alias_scales(CLayer *self);

PyObject *CLayer_get_input_dims(CLayer *self);

PyObject *CLayer_get_output_dims(CLayer *self);

PyObject *CLayer_get_cval(CLayer *self);

// ********************************************* methods ***************************************************************

PyObject *CLayer_scale(CLayer *self, PyObject *args);

PyObject *CLayer_rescale(CLayer *self, PyObject *args);

PyObject *CLayer_reduce(CLayer *self);

// #####################################################################################################################


#endif