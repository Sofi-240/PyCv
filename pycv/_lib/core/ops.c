#include "ops.h"
#include "ops_base.h"

#include "filters.h"
#include "morphology.h"

// #####################################################################################################################

static int Input_To_Array(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

static int InputOptional_To_Array(PyObject *object, PyArrayObject **output)
{
    if (object == Py_None) {
        *output = NULL;
        return 1;
    }
    return Input_To_Array(object, output);
}

static int Output_To_Array(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_BEHAVED_NS | NPY_ARRAY_WRITEBACKIFCOPY;

    if (PyArray_Check(object) && !PyArray_ISWRITEABLE((PyArrayObject *)object)) {
        PyErr_SetString(PyExc_ValueError, "Output array is read-only.");
        return 0;
    }
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

// #####################################################################################################################

PyObject* convolve(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *kernel = NULL, *output = NULL;
    PyArray_Dims origins = {NULL, 0};
    int mode;
    double constant_value;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&id",
            Input_To_Array, &input,
            Input_To_Array, &kernel,
            Output_To_Array, &output,
            PyArray_IntpConverter, &origins,
            &mode, &constant_value)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(kernel))) {
        PyErr_SetString(PyExc_RuntimeError, "kernel dtype not supported");
        goto exit;
    }

    ops_convolve(input, kernel, output, origins.ptr, (BordersMode)mode, constant_value);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(kernel);
        Py_XDECREF(output);
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* rank_filter(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *footprint = NULL, *output = NULL;
    PyArray_Dims origins = {NULL, 0};
    int rank;
    int mode;
    double constant_value;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&iO&id",
            Input_To_Array, &input,
            Input_To_Array, &footprint,
            Output_To_Array, &output,
            &rank,
            PyArray_IntpConverter, &origins,
            &mode, &constant_value)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(footprint))) {
        PyErr_SetString(PyExc_RuntimeError, "footprint dtype not supported");
        goto exit;
    }

    ops_rank_filter(input, footprint, output, rank, origins.ptr, (BordersMode)mode, constant_value);
    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(footprint);
        Py_XDECREF(output);
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* binary_erosion(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *strel = NULL, *output = NULL, *mask = NULL;
    PyArray_Dims origins = {NULL, 0};
    int iterations, invert;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&iO&i",
            Input_To_Array, &input,
            Input_To_Array, &strel,
            Output_To_Array, &output,
            PyArray_IntpConverter, &origins,
            &iterations,
            InputOptional_To_Array, &mask,
            &invert)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (mask && !valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(strel))) {
        PyErr_SetString(PyExc_RuntimeError, "strel dtype not supported");
        goto exit;
    }

    ops_binary_erosion(input, strel, output, origins.ptr, iterations, mask, invert);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(strel);
        Py_XDECREF(output);
        if (mask) {
            Py_XDECREF(mask);
        }
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* erosion(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *flat_strel = NULL, *non_flat_strel = NULL, *output = NULL, *mask = NULL;
    PyArray_Dims origins = {NULL, 0};
    double cast_value;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&O&O&d",
            Input_To_Array, &input,
            Input_To_Array, &flat_strel,
            InputOptional_To_Array, &non_flat_strel,
            Output_To_Array, &output,
            PyArray_IntpConverter, &origins,
            InputOptional_To_Array, &mask,
            &cast_value)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (mask && !valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "flat strel dtype not supported");
        goto exit;
    }
    if (non_flat_strel && !valid_dtype(PyArray_TYPE(non_flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "non flat strel dtype not supported");
        goto exit;
    }

    ops_gray_ero_or_dil(input, flat_strel, non_flat_strel, output, origins.ptr, mask, cast_value, ERO);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(flat_strel);
        if (non_flat_strel) {
            Py_XDECREF(non_flat_strel);
        }
        if (mask) {
            Py_XDECREF(mask);
        }
        Py_XDECREF(output);
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* dilation(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *flat_strel = NULL, *non_flat_strel = NULL, *output = NULL, *mask = NULL;
    PyArray_Dims origins = {NULL, 0};
    double cast_value;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&O&O&d",
            Input_To_Array, &input,
            Input_To_Array, &flat_strel,
            InputOptional_To_Array, &non_flat_strel,
            Output_To_Array, &output,
            PyArray_IntpConverter, &origins,
            InputOptional_To_Array, &mask,
            &cast_value)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (mask && !valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "flat strel dtype not supported");
        goto exit;
    }
    if (non_flat_strel && !valid_dtype(PyArray_TYPE(non_flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "non flat strel dtype not supported");
        goto exit;
    }

    ops_gray_ero_or_dil(input, flat_strel, non_flat_strel, output, origins.ptr, mask, cast_value, DIL);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(flat_strel);
        if (non_flat_strel) {
            Py_XDECREF(non_flat_strel);
        }
        if (mask) {
            Py_XDECREF(mask);
        }
        Py_XDECREF(output);
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* binary_region_fill(PyObject* self, PyObject* args)
{
    PyArrayObject *strel = NULL, *output = NULL;
    PyArray_Dims seed_point = {NULL, 0};
    PyArray_Dims origins = {NULL, 0};

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&",
            Output_To_Array, &output,
            Input_To_Array, &strel,
            PyArray_IntpConverter, &seed_point,
            PyArray_IntpConverter, &origins)) {
        goto exit;
    }


    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(strel))) {
        PyErr_SetString(PyExc_RuntimeError, "strel dtype not supported");
        goto exit;
    }

    ops_binary_region_fill(output, strel, seed_point.ptr, origins.ptr);
    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(strel);
        Py_XDECREF(output);
        PyDimMem_FREE(seed_point.ptr);
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* labeling(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *output = NULL, *values_map = NULL;
    int connectivity;

    if (!PyArg_ParseTuple(
            args,
            "O&iO&O&",
            Input_To_Array, &input,
            &connectivity,
            InputOptional_To_Array, &values_map,
            Output_To_Array, &output)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (values_map && !valid_dtype(PyArray_TYPE(values_map))) {
        PyErr_SetString(PyExc_RuntimeError, "values map dtype not supported");
        goto exit;
    }
    if (values_map && PyArray_NDIM(values_map) > 1) {
        PyErr_SetString(PyExc_RuntimeError, "values map need to be 1D array");
        goto exit;
    }


    ops_labeling(input, connectivity, values_map, output);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(output);
        if (values_map) {
            Py_XDECREF(values_map);
        }
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* skeletonize(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *output = NULL;

    if (!PyArg_ParseTuple(
            args,
            "O&O&",
            Input_To_Array, &input,
            Output_To_Array, &output)) {
        goto exit;
    }

    if (!valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    ops_skeletonize(input, output);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(output);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

// #####################################################################################################################

static PyMethodDef methods[] = {
    {
        "convolve",
        (PyCFunction)convolve,
        METH_VARARGS,
        NULL
    },
    {
        "rank_filter",
        (PyCFunction)rank_filter,
        METH_VARARGS,
        NULL
    },
    {
        "binary_erosion",
        (PyCFunction)binary_erosion,
        METH_VARARGS,
        NULL
    },
    {
        "erosion",
        (PyCFunction)erosion,
        METH_VARARGS,
        NULL
    },
    {
        "dilation",
        (PyCFunction)dilation,
        METH_VARARGS,
        NULL
    },
    {
        "binary_region_fill",
        (PyCFunction)binary_region_fill,
        METH_VARARGS,
        NULL
    },
    {
        "labeling",
        (PyCFunction)labeling,
        METH_VARARGS,
        NULL
    },
    {
        "skeletonize",
        (PyCFunction)skeletonize,
        METH_VARARGS,
        NULL
    },
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "ops",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_ops(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&module);
};

// #####################################################################################################################