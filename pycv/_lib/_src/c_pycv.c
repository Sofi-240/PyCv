#include "c_pycv.h"
#include "c_pycv_base.h"

#include "c_pycv_filters.h"
#include "c_pycv_morphology.h"
#include "c_pycv_transform.h"
#include "c_pycv_canny.h"
#include "c_pycv_draw.h"
#include "c_pycv_kd_tree.h"
#include "c_pycv_convex_hull.h"
#include "c_pycv_cluster.h"
#include "c_pycv_haar_like.h"
#include "c_pycv_minmax_tree.h"
#include "c_pycv_features.h"
#include "c_pycv_peaks.h"

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
    int invert, c_val, iterations;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&iO&ii",
            InputToArray, &input,
            InputToArray, &strel,
            OutputToArray, &output,
            PyArray_IntpConverter, &center,
            &iterations,
            InputOptionalToArray, &mask,
            &invert, &c_val)) {
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


    PYCV_binary_erosion(input, strel, output, center.ptr, iterations, mask, invert, c_val);

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

PyObject* gray_erosion(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *strel = NULL, *output = NULL, *mask = NULL;
    PyArray_Dims center = {NULL, 0};
    int invert;
    double c_val;

    if (!PyArg_ParseTuple(
            args,
            "O&O&O&O&O&id",
            InputToArray, &input,
            InputToArray, &strel,
            OutputToArray, &output,
            PyArray_IntpConverter, &center,
            InputOptionalToArray, &mask,
            &invert, &c_val)) {
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


    PYCV_gray_erosion(input, strel, output, center.ptr, mask, invert, (npy_double)c_val);

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
    static char* kwlist[] = {"", "", "", "offset", "normalize", "expend", "threshold", "line_length", "line_gap", NULL};
    PyArrayObject *input = NULL, *param = NULL, *output = NULL;
    int offset, normalize, expend, threshold, line_length, line_gap;

    if (!PyArg_ParseTupleAndKeywords(
            args, keywords,
            "iO&O&|iiiiii", kwlist,
            &hough_mode,
            InputToArray, &input,
            InputToArray, &param,
            &offset, &normalize, &expend,
            &threshold, &line_length, &line_gap)) {
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
            output = PYCV_hough_line_transform(input, param, (npy_intp)offset);
            break;
        case PYCV_HOUGH_CIRCLE:
            output = PYCV_hough_circle_transform(input, param, normalize, expend);
            break;
        case PYCV_HOUGH_LINE_PROBABILISTIC:
            output = PYCV_hough_probabilistic_line(input, param, (npy_intp)offset,
                                                   (npy_intp)threshold, (npy_intp)line_length, (npy_intp)line_gap);
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "hough mode not supported");
            goto exit;
    }

    exit:
        Py_XDECREF(input);
        Py_XDECREF(param);
        return PyErr_Occurred() ? NULL : (PyObject *)output;
}

PyObject* linear_interp1D(PyObject* self, PyObject* args)
{
    PyArrayObject *xn = NULL, *xp = NULL, *fp = NULL, *fn;
    double l, h;

    if (!PyArg_ParseTuple(args, "O&O&O&dd", InputToArray, &xn, InputToArray, &xp, InputToArray, &fp, &l, &h)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(xn))) {
        PyErr_SetString(PyExc_RuntimeError, "xn dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(xp))) {
        PyErr_SetString(PyExc_RuntimeError, "xp dtype not supported");
        goto exit;
    }
    if (!PYCV_valid_dtype(PyArray_TYPE(fp))) {
        PyErr_SetString(PyExc_RuntimeError, "fp dtype not supported");
        goto exit;
    }

    int valid = PYCV_linear_interp1D(xn, xp, fp, l, h, &fn);

    exit:
        Py_XDECREF(xn);
        Py_XDECREF(xp);
        Py_XDECREF(fp);
        return valid ? (PyObject *)fn : Py_BuildValue("");
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
            output = PYCV_draw_circle(center_point.ptr, (npy_intp)radius);
            break;
        case PYCV_DRAW_ELLIPSE:
            if (!center_point.ptr || !a || !b) {
                PyErr_SetString(PyExc_RuntimeError, "missing center_point or a or b arguments");
                goto exit;
            }
            if (a == b) {
                output = PYCV_draw_circle(center_point.ptr, (npy_intp)a);
            } else {
                output = PYCV_draw_ellipse(center_point.ptr, (npy_intp)a, (npy_intp)b);
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

PyObject* integral_image(PyObject* self, PyObject* args)
{
    PyArrayObject *output = NULL, *inputs = NULL;

    if (!PyArg_ParseTuple(args, "O&", InputToArray, &inputs)) {
        goto exit;
    }

    PYCV_integral_image(inputs, &output);

    exit:
        Py_XDECREF(inputs);
        return PyErr_Occurred() ? Py_BuildValue("") : (PyObject *)output;
}

// #####################################################################################################################

PyObject* peak_nonmaximum_suppression(PyObject* self, PyObject* args)
{
    PyArrayObject *input = NULL, *output;
    PyArray_Dims min_distance = {NULL, 0};
    int mode;
    double threshold, c_val;

    if (!PyArg_ParseTuple(
            args, "O&O&did", InputToArray, &input, PyArray_IntpConverter, &min_distance, &threshold, &mode, &c_val)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(input))) {
        PyErr_SetString(PyExc_RuntimeError, "input dtype not supported");
        goto exit;
    }


    int valid = PYCV_peaks_nonmaximum_suppression(input, min_distance.ptr, threshold, mode, c_val, &output);


    exit:
        Py_XDECREF(input);
        PyDimMem_FREE(min_distance.ptr);
        return valid ? (PyObject *)output : Py_BuildValue("");
}

// #####################################################################################################################

PyObject* gray_co_occurrence_matrix(PyObject* self, PyObject* args)
{
    PyArrayObject *gray = NULL, *distances = NULL, *angle = NULL, *glcm;
    int levels;

    if (!PyArg_ParseTuple(args, "O&O&O&i",
                          InputToArray, &gray, InputToArray, &distances, InputToArray, &angle, &levels)) {
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(gray))) {
        PyErr_SetString(PyExc_RuntimeError, "gray dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(distances))) {
        PyErr_SetString(PyExc_RuntimeError, "distances dtype not supported");
        goto exit;
    }

    if (!PYCV_valid_dtype(PyArray_TYPE(angle))) {
        PyErr_SetString(PyExc_RuntimeError, "angle dtype not supported");
        goto exit;
    }

    int valid = PYCV_glcm(gray, distances, angle, levels, &glcm);

    exit:
        Py_XDECREF(gray);
        Py_XDECREF(distances);
        Py_XDECREF(angle);
        return valid ? (PyObject *)glcm : Py_BuildValue("");
}

// #####################################################################################################################

static PyMemberDef CKDnode_members[] = {
    {"start_index", T_INT, offsetof(CKDnode, start_index), 0, NULL},
    {"end_index", T_INT, offsetof(CKDnode, end_index), 0, NULL},
    {"children", T_INT, offsetof(CKDnode, children), 0, NULL},
    {"split_dim", T_INT, offsetof(CKDnode, split_dim), 0, NULL},
    {"split_val", T_DOUBLE, offsetof(CKDnode, split_val), 0, NULL},
    {"lesser_index", T_INT, offsetof(CKDnode, lesser_index), 0, NULL},
    {"higher_index", T_INT, offsetof(CKDnode, higher_index), 0, NULL},
    {"level", T_INT, offsetof(CKDnode, level), 0, NULL},
    {"lesser", T_OBJECT, offsetof(CKDnode, lesser), 0, NULL},
    {"higher", T_OBJECT, offsetof(CKDnode, higher), 0, NULL},
    {NULL}  /* Sentinel */
};

PyTypeObject CKDnode_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_pycv.CKDnode",
    .tp_doc = PyDoc_STR("CKDnode objects"),
    .tp_basicsize = sizeof(CKDnode),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = CKDnodePy_new,
    .tp_init = (initproc)CKDnodePy_init,
    .tp_dealloc = (destructor)CKDnodePy_dealloc,
    .tp_members = CKDnode_members,
};

// *********************************************************************************************************************

static PyMemberDef CKDtree_members[] = {
    {"m", T_INT, offsetof(CKDtree, m), 0, NULL},
    {"n", T_INT, offsetof(CKDtree, n), 0, NULL},
    {"leafsize", T_INT, offsetof(CKDtree, leafsize), 0, NULL},
    {"data", T_OBJECT, offsetof(CKDtree, data), 0, NULL},
    {"dims_min", T_OBJECT, offsetof(CKDtree, dims_min), 0, NULL},
    {"dims_max", T_OBJECT, offsetof(CKDtree, dims_max), 0, NULL},
    {"indices", T_OBJECT, offsetof(CKDtree, indices), 0, NULL},
    {"tree_list", T_OBJECT, offsetof(CKDtree, tree_list), 0, NULL},
    {"tree", T_OBJECT, offsetof(CKDtree, tree), 0, NULL},
    {"size", T_INT, offsetof(CKDtree, size), 0, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef CKDtree_methods[] = {
    {"knn_query", (PyCFunction)CKDtree_knn_query, METH_VARARGS, NULL},
    {"ball_point_query", (PyCFunction)CKDtree_ball_point_query, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyTypeObject CKDtree_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_pycv.CKDtree",
    .tp_doc = PyDoc_STR("CKDtree objects"),
    .tp_basicsize = sizeof(CKDtree),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = CKDtreePy_new,
    .tp_init = (initproc)CKDtreePy_init,
    .tp_dealloc = (destructor)CKDtreePy_dealloc,
    .tp_members = CKDtree_members,
    .tp_methods = CKDtree_methods,
};

// *********************************************************************************************************************

static PyMemberDef CConvexHull_members[] = {
    {"ndim", T_INT, offsetof(CConvexHull, ndim), 0, NULL},
    {"n_vertices", T_INT, offsetof(CConvexHull, n_vertices), 0, NULL},
    {"points", T_OBJECT, offsetof(CConvexHull, points), 0, NULL},
    {"vertices", T_OBJECT, offsetof(CConvexHull, vertices), 0, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef CConvexHull_methods[] = {
    {"to_image", (PyCFunction)CConvexHullPy_convex_to_image, METH_VARARGS, NULL},
    {"query_point", (PyCFunction)CConvexHullPy_query_point, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyTypeObject CConvexHull_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_pycv.CConvexHull",
    .tp_doc = PyDoc_STR("CConvexHull objects"),
    .tp_basicsize = sizeof(CConvexHull),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = CConvexHullPy_new,
    .tp_init = (initproc)CConvexHullPy_init,
    .tp_dealloc = (destructor)CConvexHullPy_dealloc,
    .tp_members = CConvexHull_members,
    .tp_methods = CConvexHull_methods,
};

// *********************************************************************************************************************

static PyMemberDef CKMeans_members[] = {
    {"k", T_INT, offsetof(CKMeans, k), 0, NULL},
    {"iterations", T_INT, offsetof(CKMeans, iterations), 0, NULL},
    {"tol", T_DOUBLE, offsetof(CKMeans, tol), 0, NULL},
    {"n_samples", T_INT, offsetof(CKMeans, n_samples), 0, NULL},
    {"ndim", T_INT, offsetof(CKMeans, ndim), 0, NULL},
    {"init_method", T_INT, offsetof(CKMeans, init_method), 0, NULL},
    {"centers", T_OBJECT, offsetof(CKMeans, centers), 0, NULL},
    {"data", T_OBJECT, offsetof(CKMeans, data), 0, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef CKMeans_methods[] = {
    {"fit", (PyCFunction)CKMeansPy_fit, METH_VARARGS|METH_KEYWORDS, NULL},
    {"predict", (PyCFunction)CKMeansPy_predict, METH_VARARGS|METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyTypeObject CKMeans_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_pycv.CKMeans",
    .tp_doc = PyDoc_STR("CKMeans objects"),
    .tp_basicsize = sizeof(CKMeans),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = CKMeansPy_new,
    .tp_init = (initproc)CKMeansPy_init,
    .tp_dealloc = (destructor)CKMeansPy_dealloc,
    .tp_members = CKMeans_members,
    .tp_methods = CKMeans_methods,
};

// *********************************************************************************************************************

static PyMemberDef CMinMaxTree_members[] = {
    {"connectivity", T_INT, offsetof(CMinMaxTree, connectivity), 0, NULL},
    {"ndim", T_INT, offsetof(CMinMaxTree, ndim), 0, NULL},
    {"size", T_INT, offsetof(CMinMaxTree, size), 0, NULL},
    {"_is_max", T_INT, offsetof(CMinMaxTree, _is_max), 0, NULL},
    {"dims", T_OBJECT, offsetof(CMinMaxTree, dims), 0, NULL},
    {"data", T_OBJECT, offsetof(CMinMaxTree, data), 0, NULL},
    {"traverser", T_OBJECT, offsetof(CMinMaxTree, traverser), 0, NULL},
    {"nodes", T_OBJECT, offsetof(CMinMaxTree, nodes), 0, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef CMinMaxTree_methods[] = {
    {"compute_area", (PyCFunction)CMinMaxTreePy_compute_area, METH_NOARGS, NULL},
    {"tree_filter", (PyCFunction)CMinMaxTreePy_tree_filter, METH_VARARGS|METH_KEYWORDS, NULL},
    {"label_image", (PyCFunction)CMinMaxTreePy_label_image, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyTypeObject CMinMaxTree_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_pycv.CMinMaxTree",
    .tp_doc = PyDoc_STR("CMinMaxTree objects"),
    .tp_basicsize = sizeof(CMinMaxTree),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = CMinMaxTreePy_new,
    .tp_init = (initproc)CMinMaxTreePy_init,
    .tp_dealloc = (destructor)CMinMaxTreePy_dealloc,
    .tp_members = CMinMaxTree_members,
    .tp_methods = CMinMaxTree_methods,
};

// *********************************************************************************************************************

static PyMemberDef CHaarFeatures_members[] = {
    {"htype", T_OBJECT, offsetof(CHaarFeatures, htype), 0, NULL},
    {"ndim", T_INT, offsetof(CHaarFeatures, ndim), 0, NULL},
    {"n_features", T_INT, offsetof(CHaarFeatures, n_features), 0, NULL},
    {"n_rect", T_INT, offsetof(CHaarFeatures, n_rect), 0, NULL},
    {"axis", T_OBJECT, offsetof(CHaarFeatures, axis), 0, NULL},
    {"feature_dims", T_OBJECT, offsetof(CHaarFeatures, feature_dims), 0, NULL},
    {"_htype", T_INT, offsetof(CHaarFeatures, _htype), 0, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef CHaarFeatures_methods[] = {
    {"coordinates", (PyCFunction)CHaarFeaturesPy_coordinates, METH_VARARGS|METH_KEYWORDS, NULL},
    {"haar_like_feature", (PyCFunction)CHaarFeaturesPy_like_features, METH_VARARGS|METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyTypeObject CHaarFeatures_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_pycv.CHaarFeatures",
    .tp_doc = PyDoc_STR("CHaarFeatures objects"),
    .tp_basicsize = sizeof(CHaarFeatures),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = CHaarFeaturesPy_new,
    .tp_init = (initproc)CHaarFeaturesPy_init,
    .tp_dealloc = (destructor)CHaarFeaturesPy_dealloc,
    .tp_members = CHaarFeatures_members,
    .tp_methods = CHaarFeatures_methods,
};


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
        "gray_erosion",
        (PyCFunction)gray_erosion,
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
        "linear_interp1D",
        (PyCFunction)linear_interp1D,
        METH_VARARGS,
        NULL
    },
    {
        "canny_nonmaximum_suppression",
        (PyCFunction)canny_nonmaximum_suppression,
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
        "integral_image",
        (PyCFunction)integral_image,
        METH_VARARGS,
        NULL
    },
    {
        "peak_nonmaximum_suppression",
        (PyCFunction)peak_nonmaximum_suppression,
        METH_VARARGS,
        NULL
    },
    {
        "gray_co_occurrence_matrix",
        (PyCFunction)gray_co_occurrence_matrix,
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
    PyObject *m;
    if ((PyType_Ready(&CKDnode_Type) < 0) ||
        (PyType_Ready(&CKDtree_Type) < 0) ||
        (PyType_Ready(&CConvexHull_Type) < 0) ||
        (PyType_Ready(&CKMeans_Type) < 0) ||
        (PyType_Ready(&CMinMaxTree_Type) < 0) ||
        (PyType_Ready(&CHaarFeatures_Type) < 0)) {
        return NULL;
    }

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;

    if ((PyModule_AddObjectRef(m, "CKDnode", (PyObject *) &CKDnode_Type) < 0) ||
        (PyModule_AddObjectRef(m, "CKDtree", (PyObject *) &CKDtree_Type) < 0) ||
        (PyModule_AddObjectRef(m, "CConvexHull", (PyObject *) &CConvexHull_Type) < 0) ||
        (PyModule_AddObjectRef(m, "CKMeans", (PyObject *) &CKMeans_Type) < 0) ||
        (PyModule_AddObjectRef(m, "CMinMaxTree", (PyObject *) &CMinMaxTree_Type) < 0) ||
        (PyModule_AddObjectRef(m, "CHaarFeatures", (PyObject *) &CHaarFeatures_Type) < 0)) {
        Py_DECREF(m);
        return NULL;
    }
    import_array(); // Initialize NumPy API
    return m;
};

// #####################################################################################################################













