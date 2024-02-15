#include "c_pycv.h"
#include "c_pycv_base.h"

#include "c_pycv_filters.h"
#include "c_pycv_morphology.h"
#include "c_pycv_transform.h"
#include "c_pycv_canny.h"
#include "c_pycv_maxtree.h"
#include "c_pycv_draw.h"
#include "c_pycv_convexhull.h"
#include "c_pycv_features.h"

// #####################################################################################################################

static int InputToArray(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

static int InputOptionalToArray(PyObject *object, PyArrayObject **output)
{
    if (object == Py_None) {
        *output = NULL;
        return 1;
    }
    return InputToArray(object, output);
}

static int OutputToArray(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_BEHAVED_NS | NPY_ARRAY_WRITEBACKIFCOPY;

    if (PyArray_Check(object) && !PyArray_ISWRITEABLE((PyArrayObject *)object)) {
        PyErr_SetString(PyExc_ValueError, "Output array is read-only.");
        return 0;
    }
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

static int OutputOptionalToArray(PyObject *object, PyArrayObject **output)
{
    if (object == Py_None) {
        *output = NULL;
        return 1;
    }
    return OutputToArray(object, output);
}

// #####################################################################################################################

PyObject* convolve(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *kernel = NULL, *output = NULL;
    PyArray_Dims center = {NULL, 0};
    int mode;
    double c_val;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&id",
            InputToArray, &input,
            InputToArray, &kernel,
            OutputToArray, &output,
            PyArray_IntpConverter, &center,
            &mode, &c_val)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(kernel))) {
        PyErr_SetString(PyExc_RuntimeError, "kernel dtype not supported");
        goto exit;
    }

    PYCV_convolve(input, kernel, output, center.ptr, (PYCV_ExtendBorder)mode, (npy_double)c_val);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(kernel);
        Py_XDECREF(output);
        PyDimMem_FREE(center.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* rank_filter(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *footprint = NULL, *output = NULL;
    PyArray_Dims center = {NULL, 0};
    int mode, rank;
    double c_val;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&IO&id",
            InputToArray, &input,
            InputToArray, &footprint,
            OutputToArray, &output,
            &rank,
            PyArray_IntpConverter, &center,
            &mode, &c_val)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(footprint))) {
        PyErr_SetString(PyExc_RuntimeError, "footprint dtype not supported");
        goto exit;
    }

    PYCV_rank_filter(input, footprint, output, (npy_intp)rank, center.ptr, (PYCV_ExtendBorder)mode, (npy_double)c_val);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(footprint);
        Py_XDECREF(output);
        PyDimMem_FREE(center.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

// #####################################################################################################################

PyObject* binary_erosion(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *strel = NULL, *output = NULL, *mask = NULL;
    PyArray_Dims center = {NULL, 0};
    int iterations, op, c_val;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&iO&ii",
            InputToArray, &input,
            InputToArray, &strel,
            OutputToArray, &output,
            PyArray_IntpConverter, &center,
            &iterations,
            InputOptionalToArray, &mask,
            &op, &c_val)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(strel))) {
        PyErr_SetString(PyExc_RuntimeError, "strel dtype not supported");
        goto exit;
    }

    if (mask && !PYCV_valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }


    PYCV_binary_erosion(input, strel, output, center.ptr, iterations, mask, (PYCV_MorphOP)op, c_val);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(strel);
        Py_XDECREF(output);
        PyDimMem_FREE(center.ptr);
        if (mask) {
            Py_XDECREF(mask);
        }
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* gray_erosion_dilation(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *flat_strel = NULL, *non_flat_strel = NULL, *output = NULL, *mask = NULL;
    PyArray_Dims center = {NULL, 0};
    int op, c_val;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&O&O&ii",
            InputToArray, &input,
            InputToArray, &flat_strel,
            InputOptionalToArray, &non_flat_strel,
            OutputToArray, &output,
            PyArray_IntpConverter, &center,
            InputOptionalToArray, &mask,
            &op, &c_val)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "flat_strel dtype not supported");
        goto exit;
    }

    if (non_flat_strel && !PYCV_valid_dtype(PyArray_TYPE(non_flat_strel))) {
        PyErr_SetString(PyExc_RuntimeError, "non_flat_strel dtype not supported");
        goto exit;
    }

    if (mask && !PYCV_valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }


    PYCV_gray_erosion_dilation(input, flat_strel, non_flat_strel, output, center.ptr, mask, (PYCV_MorphOP)op, c_val);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(flat_strel);
        if (non_flat_strel) {
            Py_XDECREF(non_flat_strel);
        }
        Py_XDECREF(output);
        PyDimMem_FREE(center.ptr);
        if (mask) {
            Py_XDECREF(mask);
        }
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* binary_region_fill(PyObject* self, PyObject* args)
{
    PyArrayObject *output = NULL, *strel = NULL;
    PyArray_Dims center = {NULL, 0};
    PyArray_Dims seed_point = {NULL, 0};

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&",
            OutputToArray, &output,
            PyArray_IntpConverter, &seed_point,
            InputToArray, &strel,
            PyArray_IntpConverter, &center)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(strel))) {
        PyErr_SetString(PyExc_RuntimeError, "strel dtype not supported");
        goto exit;
    }

    if (!seed_point.ptr) {
        PyErr_SetString(PyExc_RuntimeError, "seed_point is NULL");
        goto exit;
    }

    PYCV_binary_region_fill(output, seed_point.ptr, strel, center.ptr);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(strel);
        Py_XDECREF(output);
        PyDimMem_FREE(center.ptr);
        PyDimMem_FREE(seed_point.ptr);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* labeling(PyObject* self, PyObject* args)
{
    PyArrayObject *output = NULL, *input = NULL;
    int connectivity;

    if (!PyArg_ParseTuple(
            args,
            "O&iO&",
            InputToArray, &input,
            &connectivity,
            OutputToArray, &output)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    PYCV_labeling(input, connectivity, output);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(output);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* skeletonize(PyObject* self, PyObject* args)
{
    PyArrayObject *output = NULL, *input = NULL;

    if (!PyArg_ParseTuple(
            args,
            "O&",
            InputToArray, &input)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }

    output = PYCV_skeletonize(input);

    exit:
        Py_XDECREF(input);
        return output ? (PyObject *)output : NULL;
}

// #####################################################################################################################


PyObject* resize(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *output = NULL;
    int mode, order, grid_mode;
    double constant_value;

    if (!PyArg_ParseTuple(
            args,
            "O&O&iiid",
            InputToArray, &input,
            OutputToArray, &output,
            &order, &grid_mode, &mode, &constant_value)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    PYCV_resize(input, output, (npy_intp)order, (npy_intp)grid_mode, (PYCV_ExtendBorder)mode, constant_value);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(output);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* geometric_transform(PyObject* self, PyObject* args)
{
    PyArrayObject *matrix = NULL, *input = NULL, *output = NULL, *src = NULL, *dst = NULL;
    int mode, order;
    double constant_value;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&O&iid",
            InputToArray, &matrix,
            InputOptionalToArray, &input,
            OutputOptionalToArray, &output,
            InputOptionalToArray, &src,
            OutputOptionalToArray, &dst,
            &order, &mode, &constant_value)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(matrix))) {
        PyErr_SetString(PyExc_RuntimeError, "matrix dtype not supported");
        goto exit;
    }

    if (input && !PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (output && !PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    if (src && !PYCV_valid_dtype(PyArray_TYPE(src))) {
        PyErr_SetString(PyExc_RuntimeError, "src dtype not supported");
        goto exit;
    }
    if (dst && !PYCV_valid_dtype(PyArray_TYPE(dst))) {
        PyErr_SetString(PyExc_RuntimeError, "dst dtype not supported");
        goto exit;
    }

    PYCV_geometric_transform(matrix, input, output, src, dst, (npy_intp)order, (PYCV_ExtendBorder)mode, (npy_double)constant_value);

    if (output) {
        PyArray_ResolveWritebackIfCopy(output);
    }
    if (dst) {
        PyArray_ResolveWritebackIfCopy(dst);
    }


    exit:
        Py_XDECREF(matrix);
        if (input) {
            Py_XDECREF(input);
        }
        if (output) {
            Py_XDECREF(output);
        }
        if (src) {
            Py_XDECREF(src);
        }
        if (dst) {
            Py_XDECREF(dst);
        }
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* hough_transform(PyObject* self, PyObject* args, PyObject* keywords)
{
    int hough_mode;
    static char* kwlist[] = {"", "", "", "offset", "normalize", "expend", NULL};
    PyArrayObject *input = NULL, *param = NULL, *h_space = NULL;
    int offset, normalize, expend;

    if (!PyArg_ParseTupleAndKeywords(
            args, keywords,
            "iO&O&|iii", kwlist,
            &hough_mode,
            InputToArray, &input,
            InputToArray, &param,
            &offset, &normalize, &expend)) {
        PyErr_SetString(PyExc_RuntimeError, "invalid args or keywords");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(param))) {
        PyErr_SetString(PyExc_RuntimeError, "param dtype not supported");
        goto exit;
    }

    switch ((PYCV_HoughMode)hough_mode) {
        case PYCV_HOUGH_LINE:
            h_space = PYCV_hough_line_transform(input, param, (npy_intp)offset);
            break;
        case PYCV_HOUGH_CIRCLE:
            h_space = PYCV_hough_circle_transform(input, param, normalize, expend);
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "hough mode not supported");
            goto exit;
    }

    exit:
        Py_XDECREF(input);
        Py_XDECREF(param);
        return PyErr_Occurred() ? NULL : (PyObject *)h_space;
}

// #####################################################################################################################

PyObject* canny_nonmaximum_suppression(PyObject* self, PyObject* args)
{
    PyArrayObject *magnitude = NULL, *grad_y = NULL, *grad_x = NULL, *mask = NULL, *output = NULL;
    double threshold;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&dO&O&",
            InputToArray, &magnitude,
            InputToArray, &grad_y,
            InputToArray, &grad_x,
            &threshold,
            InputOptionalToArray, &mask,
            OutputToArray, &output)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(magnitude))) {
        PyErr_SetString(PyExc_RuntimeError, "magnitude dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(grad_y))) {
        PyErr_SetString(PyExc_RuntimeError, "grad_y dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(grad_x))) {
        PyErr_SetString(PyExc_RuntimeError, "grad_x dtype not supported");
        goto exit;
    }
    if (mask && !PYCV_valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    PYCV_canny_nonmaximum_suppression(magnitude, grad_y, grad_x, (npy_double)threshold, mask, output);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(magnitude);
        Py_XDECREF(grad_y);
        Py_XDECREF(grad_x);
        if (mask) {
            Py_XDECREF(mask);
        }
        Py_XDECREF(output);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

// #####################################################################################################################

PyObject* build_max_tree(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *traverser = NULL, *parent = NULL;
    int connectivity;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&i",
            InputToArray, &input,
            OutputToArray, &traverser,
            OutputToArray, &parent,
            &connectivity)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(traverser))) {
        PyErr_SetString(PyExc_RuntimeError, "traverser dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(parent))) {
        PyErr_SetString(PyExc_RuntimeError, "parent dtype not supported");
        goto exit;
    }

    PYCV_build_max_tree(input, traverser, parent, (npy_intp)connectivity);

    PyArray_ResolveWritebackIfCopy(traverser);
    PyArray_ResolveWritebackIfCopy(parent);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(traverser);
        Py_XDECREF(parent);
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* max_tree_compute_area(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *traverser = NULL, *parent = NULL, *output = NULL;
    int connectivity;

    if (!PyArg_ParseTuple(
            args,
            "O&O&iO&O&",
            InputOptionalToArray, &input,
            OutputToArray, &output,
            &connectivity,
            InputOptionalToArray, &traverser,
            InputOptionalToArray, &parent)) {
        goto exit;
    }

    if (input && !PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (traverser && !PYCV_valid_dtype(PyArray_TYPE(traverser))) {
        PyErr_SetString(PyExc_RuntimeError, "traverser dtype not supported");
        goto exit;
    }
    if (parent && !PYCV_valid_dtype(PyArray_TYPE(parent))) {
        PyErr_SetString(PyExc_RuntimeError, "parent dtype not supported");
        goto exit;
    }

    PYCV_max_tree_compute_area(input, output, connectivity, traverser, parent);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        if (input) {
            Py_XDECREF(input);
        }
        Py_XDECREF(output);
        if (traverser) {
            Py_XDECREF(traverser);
        }
        if (parent) {
            Py_XDECREF(parent);
        }
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject* max_tree_filter(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *traverser = NULL, *parent = NULL, *output = NULL, *values_map = NULL;
    int connectivity;
    double threshold;

    if (!PyArg_ParseTuple(
            args,
            "O&dO&O&iO&O&",
            InputToArray, &input,
            &threshold,
            InputToArray, &values_map,
            OutputToArray, &output,
            &connectivity,
            InputOptionalToArray, &traverser,
            InputOptionalToArray, &parent)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(values_map))) {
        PyErr_SetString(PyExc_RuntimeError, "values_map dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }
    if (traverser && !PYCV_valid_dtype(PyArray_TYPE(traverser))) {
        PyErr_SetString(PyExc_RuntimeError, "traverser dtype not supported");
        goto exit;
    }
    if (parent && !PYCV_valid_dtype(PyArray_TYPE(parent))) {
        PyErr_SetString(PyExc_RuntimeError, "parent dtype not supported");
        goto exit;
    }

    PYCV_max_tree_filter(input, (npy_double)threshold, values_map, output, connectivity, traverser, parent);

    PyArray_ResolveWritebackIfCopy(output);

    exit:
        Py_XDECREF(input);
        Py_XDECREF(values_map);
        Py_XDECREF(output);
        if (traverser) {
            Py_XDECREF(traverser);
        }
        if (parent) {
            Py_XDECREF(parent);
        }
        return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

// #####################################################################################################################

PyObject* draw(PyObject* self, PyObject* args, PyObject* keywords)
{
    int draw_mode;
    static char* kwlist[] = {"", "point1", "point2", "center_point", "radius", "a", "b", NULL};
    PyArrayObject *output = NULL;
    PyArray_Dims point1 = {NULL, 0};
    PyArray_Dims point2 = {NULL, 0};
    PyArray_Dims center_point = {NULL, 0};
    int radius = 0, a = 0, b = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, keywords,
            "i|O&O&O&iii", kwlist,
            &draw_mode,
            PyArray_IntpConverter, &point1,
            PyArray_IntpConverter, &point2,
            PyArray_IntpConverter, &center_point,
            &radius, &a, &b)) {
        PyErr_SetString(PyExc_RuntimeError, "invalid args or keywords");
        goto exit;
    }

    switch ((PYCV_DrawMode)draw_mode) {
        case PYCV_DRAW_LINE:
            if (!point1.ptr || !point2.ptr) {
                PyErr_SetString(PyExc_RuntimeError, "missing point1 or point2 arguments");
                goto exit;
            }
            output = PYCV_draw_line(point1.ptr, point2.ptr);
            break;
        case PYCV_DRAW_CIRCLE:
            if (!center_point.ptr || !radius) {
                PyErr_SetString(PyExc_RuntimeError, "missing center_point or radius arguments");
                goto exit;
            }
            output = PYCV_draw_circle(center_point.ptr, radius);
            break;
        case PYCV_DRAW_ELLIPSE:
            if (!center_point.ptr || !a || !b) {
                PyErr_SetString(PyExc_RuntimeError, "missing center_point or a or b arguments");
                goto exit;
            }
            if (a == b) {
                output = PYCV_draw_circle(center_point.ptr, a);
            } else {
                output = PYCV_draw_ellipse(center_point.ptr, a, b);
            }
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "draw mode not supported");
            goto exit;
    }

    exit:
        PyDimMem_FREE(point1.ptr);
        PyDimMem_FREE(point2.ptr);
        PyDimMem_FREE(center_point.ptr);
        return PyErr_Occurred() ? NULL : (PyObject *)output;
}

// #####################################################################################################################

PyObject* convex_hull(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *mask = NULL, *convex_hull = NULL, *output = NULL, *points_array = NULL;
    int hull_mode;

    if (!PyArg_ParseTuple(
            args,
            "iO&O&O&O&",
            &hull_mode,
            InputOptionalToArray, &input,
            InputOptionalToArray, &mask,
            InputOptionalToArray, &points_array,
            OutputOptionalToArray, &output)) {
        goto exit;
    }

    if (input && !PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }
    if (mask && !PYCV_valid_dtype(PyArray_TYPE(mask))) {
        PyErr_SetString(PyExc_RuntimeError, "mask dtype not supported");
        goto exit;
    }
    if (points_array && !PYCV_valid_dtype(PyArray_TYPE(points_array))) {
        PyErr_SetString(PyExc_RuntimeError, "points array dtype not supported");
        goto exit;
    }
    if (output && !PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output array dtype not supported");
        goto exit;
    }

    switch ((PYCV_HullMode)hull_mode) {
        case PYCV_HULL2D_GRAM_SCAN:
            convex_hull = PYCV_graham_scan_convex_hull(input, mask, points_array, output);
            break;
        case PYCV_HULL2D_GIFT_WRAPPING:
            convex_hull = PYCV_jarvis_march_convex_hull(input, mask, points_array, output);
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "hull mode not supported");
            goto exit;
    }

    if (output) {
        PyArray_ResolveWritebackIfCopy(output);
    }

    exit:
        Py_XDECREF(input);
        if (mask) {
            Py_XDECREF(mask);
        }
        if (output) {
            Py_XDECREF(output);
        }
        if (points_array) {
            Py_XDECREF(points_array);
        }
        return convex_hull ? (PyObject *)convex_hull : NULL;
}

// #####################################################################################################################


PyObject* integral_image(PyObject* self, PyObject* args)
{
    PyArrayObject *output = NULL;

    if (!PyArg_ParseTuple(
            args,
            "O&",
            OutputToArray, &output)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(output))) {
        PyErr_SetString(PyExc_RuntimeError, "output dtype not supported");
        goto exit;
    }

    PYCV_integral_image(output);
    PyArray_ResolveWritebackIfCopy(output);

    exit:
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
        "gray_erosion_dilation",
        (PyCFunction)gray_erosion_dilation,
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
    {
        "resize",
        (PyCFunction)resize,
        METH_VARARGS,
        NULL
    },
    {
        "geometric_transform",
        (PyCFunction)geometric_transform,
        METH_VARARGS,
        NULL
    },
    {
        "hough_transform",
        (PyCFunction)hough_transform,
        METH_VARARGS|METH_KEYWORDS,
        NULL
    },
    {
        "canny_nonmaximum_suppression",
        (PyCFunction)canny_nonmaximum_suppression,
        METH_VARARGS,
        NULL
    },
    {
        "build_max_tree",
        (PyCFunction)build_max_tree,
        METH_VARARGS,
        NULL
    },
    {
        "max_tree_compute_area",
        (PyCFunction)max_tree_compute_area,
        METH_VARARGS,
        NULL
    },
    {
        "max_tree_filter",
        (PyCFunction)max_tree_filter,
        METH_VARARGS,
        NULL
    },
    {
        "draw",
        (PyCFunction)draw,
        METH_VARARGS|METH_KEYWORDS,
        NULL
    },
    {
        "convex_hull",
        (PyCFunction)convex_hull,
        METH_VARARGS,
        NULL
    },
    {
        "integral_image",
        (PyCFunction)integral_image,
        METH_VARARGS,
        NULL
    },
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "c_pycv",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_c_pycv(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&module);
};














