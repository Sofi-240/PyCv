#include "ops.h"
#include "ops_support.h"

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

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&",
            Input_To_Array, &input,
            Input_To_Array, &kernel,
            Output_To_Array, &output,
            PyArray_IntpConverter, &origins)) {
        goto exit;
    }

    ops_convolve(input, kernel, output, origins.ptr);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(kernel);
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

    ops_binary_erosion(input, strel, output, origins.ptr, iterations, mask, invert);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(strel);
        Py_XDECREF(output);
        PyDimMem_FREE(origins.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* rank_filter(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *footprint = NULL, *output = NULL;
    PyArray_Dims origins = {NULL, 0};
    int rank;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&iO&",
            Input_To_Array, &input,
            Input_To_Array, &footprint,
            Output_To_Array, &output,
            &rank,
            PyArray_IntpConverter, &origins)) {
        goto exit;
    }

    ops_rank_filter(input, footprint, output, rank, origins.ptr);
    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(footprint);
        Py_XDECREF(output);
        PyDimMem_FREE(origins.ptr);
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
        "binary_erosion",
        (PyCFunction)binary_erosion,
        METH_VARARGS,
        NULL
    },
    {
        "rank_filter",
        (PyCFunction)rank_filter,
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