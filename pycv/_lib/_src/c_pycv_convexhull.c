#include "c_pycv_base.h"
#include "c_pycv_convexhull.h"

// #####################################################################################################################

int PYCV_HullPointsInit(npy_intp ndim, npy_intp max_size, PYCV_HullPoints *object)
{
    npy_intp ii;
    object->ndim = ndim;
    object->max_size = max_size;
    object->points = malloc(max_size * sizeof(npy_intp*));
    if (!object->points) {
        object->points_size = -1;
        return 0;
    }
    for (ii = 0; ii < max_size; ii++) {
        object->points[ii] = malloc(ndim * sizeof(npy_intp));
        if (!object->points[ii]) {
            object->points_size = -1;
            return 0;
        }
    }
    object->points_size = 0;
    return 1;
}

int PYCV_HullPointsFree(PYCV_HullPoints *object)
{
    npy_intp ii;
    for (ii = 0; ii < object->max_size; ii++) {
        free(object->points[ii]);
    }
    free(object->points);
    object->points_size = 0;
    return 1;
}

PYCV_HullPoints PYCV_HullPoints_AllocateFromArray(PyArrayObject *input, PyArrayObject *mask)
{
    npy_intp ndim, array_size, ii;
    PYCV_ArrayIterator iter_i, iter_ma;
    char *pi = NULL, *pm = NULL;
    int num_type_i, num_type_m;
    npy_double inp_val;
    npy_bool ma_vla = 1;
    PYCV_HullPoints object;

    num_type_i = PyArray_TYPE(input);
    ndim = PyArray_NDIM(input);
    array_size = PyArray_SIZE(input);

    if (!PYCV_HullPointsInit(ndim, array_size, &object)) {
        return object;
    }

    pi = (void *)PyArray_DATA(input);
    PYCV_ArrayIteratorInit(input, &iter_i);

    if (mask) {
        num_type_m = mask ? PyArray_TYPE(mask) : -1;
        pm = (void *)PyArray_DATA(mask);
        PYCV_ArrayIteratorInit(mask, &iter_ma);
    }

    for (ii = 0; ii < array_size; ii++) {
        if (mask) {
            PYCV_GET_VALUE(num_type_m, npy_bool, pm, ma_vla);
            PYCV_ARRAY_ITERATOR_NEXT(iter_ma, pm);
        }
        PYCV_GET_VALUE(num_type_i, npy_double, pi, inp_val);
        if (ma_vla && fabs(inp_val) > DBL_EPSILON) {
            PYCV_HULL_POINTS_APPEND(object, iter_i.coordinates);
        }
        PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
    }
    return object;
}

PYCV_HullPoints PYCV_HullPoints_AllocateFromPointsArray(PyArrayObject *points_array)
{
    npy_intp ndim, array_size, ii, jj, coordinates[NPY_MAXDIMS];
    PYCV_ArrayIterator iter_i;
    char *pi = NULL;
    int num_type_i;
    PYCV_HullPoints object;

    num_type_i = PyArray_TYPE(points_array);
    ndim = PyArray_DIM(points_array, 1);
    array_size = PyArray_DIM(points_array, 0);

    if (!PYCV_HullPointsInit(ndim, array_size, &object)) {
        return object;
    }

    pi = (void *)PyArray_DATA(points_array);
    PYCV_ArrayIteratorInit(points_array, &iter_i);

    for (ii = 0; ii < array_size; ii++) {
        for (jj = 0; jj < ndim; jj++) {
            PYCV_GET_VALUE(num_type_i, npy_intp, pi, coordinates[jj]);
            PYCV_ARRAY_ITERATOR_NEXT(iter_i, pi);
        }
        PYCV_HULL_POINTS_APPEND(object, coordinates);
    }
    return object;
}

// #####################################################################################################################

#define PYCV_HULL_CASE_SET_POINT(_NTYPE, _dtype, _itemsize, _ndim, _pointer, _point)                                   \
case NPY_##_NTYPE:                                                                                                     \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii < _ndim; _ii++) {                                                                                \
        *(_dtype *)_pointer = (_dtype)_point[_ii];                                                                     \
        _pointer += _itemsize;                                                                                         \
    }                                                                                                                  \
}                                                                                                                      \
break

#define PYCV_HULL_SET_POINT(_NTYPE, _itemsize, _ndim, _pointer, _point)                                                \
{                                                                                                                      \
    switch (_NTYPE) {                                                                                                  \
        PYCV_HULL_CASE_SET_POINT(BOOL, npy_bool, _itemsize, _ndim, _pointer, _point);                                  \
        PYCV_HULL_CASE_SET_POINT(UBYTE, npy_ubyte, _itemsize, _ndim, _pointer, _point);                                \
        PYCV_HULL_CASE_SET_POINT(USHORT, npy_ushort, _itemsize, _ndim, _pointer, _point);                              \
        PYCV_HULL_CASE_SET_POINT(UINT, npy_uint, _itemsize, _ndim, _pointer, _point);                                  \
        PYCV_HULL_CASE_SET_POINT(ULONG, npy_ulong, _itemsize, _ndim, _pointer, _point);                                \
        PYCV_HULL_CASE_SET_POINT(ULONGLONG, npy_ulonglong, _itemsize, _ndim, _pointer, _point);                        \
        PYCV_HULL_CASE_SET_POINT(BYTE, npy_byte, _itemsize, _ndim, _pointer, _point);                                  \
        PYCV_HULL_CASE_SET_POINT(SHORT, npy_short, _itemsize, _ndim, _pointer, _point);                                \
        PYCV_HULL_CASE_SET_POINT(INT, npy_int, _itemsize, _ndim, _pointer, _point);                                    \
        PYCV_HULL_CASE_SET_POINT(LONG, npy_long, _itemsize, _ndim, _pointer, _point);                                  \
        PYCV_HULL_CASE_SET_POINT(LONGLONG, npy_longlong, _itemsize, _ndim, _pointer, _point);                          \
        PYCV_HULL_CASE_SET_POINT(FLOAT, npy_float, _itemsize, _ndim, _pointer, _point);                                \
        PYCV_HULL_CASE_SET_POINT(DOUBLE, npy_double, _itemsize, _ndim, _pointer, _point);                              \
    }                                                                                                                  \
}

// #####################################################################################################################

#define PYCV_HULL_CROSS_PRODUCT(_p0, _p1, _p2, _out)                                                                   \
{                                                                                                                      \
    _out = (_p2[0] - _p0[0]) * (_p1[1] - _p0[1]) - (_p1[0] - _p0[0]) * (_p2[1] - _p0[1]);                              \
}

#define PYCV_HULL_DISTANCE(_p1, _p2, _out)                                                                             \
{                                                                                                                      \
    npy_intp _d1, _d2;                                                                                                 \
    _d1 = _p1[0] > _p2[0] ? _p1[0] - _p2[0] : _p2[0] - _p1[0];                                                         \
    _d2 = _p1[1] > _p2[1] ? _p1[1] - _p2[1] : _p2[1] - _p1[1];                                                         \
    _out = _d1 + _d2;                                                                                                  \
}

#define PYCV_HULL_DIRECTION(_p0, _p1, _p2, _out)                                                                       \
{                                                                                                                      \
    npy_intp _ori;                                                                                                     \
    PYCV_HULL_CROSS_PRODUCT(_p0, _p1, _p2, _ori);                                                                      \
    if (_ori == 0) {                                                                                                   \
        _out = 0;                                                                                                      \
    } else if (_ori < 0) {                                                                                             \
        _out = -1;                                                                                                     \
    } else {                                                                                                           \
        _out = 1;                                                                                                      \
    }                                                                                                                  \
}

#define PYCV_HULL_COMPARE(_p0, _p1, _p2, _out)                                                                         \
{                                                                                                                      \
    npy_intp _ori;                                                                                                     \
    PYCV_HULL_CROSS_PRODUCT(_p0, _p1, _p2, _ori);                                                                      \
    if (_ori < 0) {                                                                                                    \
        _out = -1;                                                                                                     \
    } else if (_ori > 0) {                                                                                             \
        _out = 1;                                                                                                      \
    } else {                                                                                                           \
        npy_intp _dist1, _dist2;                                                                                       \
        PYCV_HULL_DISTANCE(_p0, _p1, _dist1);                                                                          \
        PYCV_HULL_DISTANCE(_p0, _p2, _dist2);                                                                          \
        _out = _dist1 < _dist2 ? -1 : 1;                                                                               \
    }                                                                                                                  \
}

#define PYCV_HULL_FIND_LEFTMOST(_points, _p_size, _ndim, _leftmost)                                                    \
{                                                                                                                      \
    npy_intp _ii, _jj, _ch = 0;                                                                                        \
    _leftmost = 0;                                                                                                     \
    for (_ii = 0; _ii < _p_size; _ii++) {                                                                              \
        if (_points[_ii][0] < _points[_leftmost][0]) {                                                                 \
            _leftmost = _ii;                                                                                           \
        } else if (_points[_ii][0] == _points[_leftmost][0]) {                                                         \
            _ch = 1;                                                                                                   \
            for (_jj = 1; _jj < _ndim; _jj++) {                                                                        \
                if (_points[_ii][_jj] > _points[_leftmost][_jj]) {                                                     \
                    _ch = 0;                                                                                           \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
            _leftmost = _ch ? _ii : _leftmost;                                                                         \
        }                                                                                                              \
    }                                                                                                                  \
}

// #####################################################################################################################

static void PYCV_hull_heapsort_swap(npy_intp **i1, npy_intp **i2)
{
    npy_intp *tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static void PYCV_hull_heapsort_heapify(npy_intp **points, npy_intp n, npy_intp i, npy_intp *p0)
{
    npy_intp cmp, largest = i, l = 2 * i + 1, r = 2 * i + 2;

    if (l < n) {
        PYCV_HULL_COMPARE(p0, points[largest], points[l], cmp);
        largest = cmp < 0 ? l : largest;
    }

    if (r < n) {
        PYCV_HULL_COMPARE(p0, points[largest], points[r], cmp);
        largest = cmp < 0 ? r : largest;
    }

    if (largest != i) {
        PYCV_hull_heapsort_swap(&points[i], &points[largest]);
        PYCV_hull_heapsort_heapify(points, n, largest, p0);
    }
}

static void PYCV_hull_heapsort(npy_intp **points, npy_intp n, npy_intp *p0)
{
    npy_intp i;

    for (i = n / 2 - 1; i >= 0; i--) {
        PYCV_hull_heapsort_heapify(points, n, i, p0);
    }

    for (i = n - 1; i >= 0; i--) {
        PYCV_hull_heapsort_swap(&points[0], &points[i]);
        PYCV_hull_heapsort_heapify(points, i, 0, p0);
    }
}

// #####################################################################################################################

static npy_intp PYCV_hull_is_point_inside(npy_intp **convex_hull, npy_intp convex_size, npy_intp *point)
{
    npy_intp a, b, c, d, sign = 0, sign_d = 0, ii, jj;

    for (ii = 0; ii < convex_size; ii++) {
        jj = (ii + 1) % convex_size;

        a = -convex_hull[jj][0] + convex_hull[ii][0];
        b = convex_hull[jj][1] - convex_hull[ii][1];
        c = -a * convex_hull[ii][1] - b * convex_hull[ii][0];

        d = a * point[1] + b * point[0] + c;

        if (d) {
            sign_d = d < 0 ? -1 : 1;
        } else {
            sign_d = sign;
        }

        if (!sign && sign_d) {
            sign = sign_d;
        } else if (sign_d != sign) {
            return 0;
        }
    }
    return 1;
}

static int PYCV_hull_to_output_array(npy_intp **convex_hull, npy_intp convex_size, npy_intp ndim, PyArrayObject *output)
{
    npy_intp array_size, ii;
    PYCV_ArrayIterator iter_o;
    char *po = NULL;
    int num_type_o;
    npy_intp inside;

    num_type_o = PyArray_TYPE(output);
    array_size = PyArray_SIZE(output);

    po = (void *)PyArray_DATA(output);
    PYCV_ArrayIteratorInit(output, &iter_o);

    if (iter_o.nd_m1 != ndim - 1) {
        PyErr_SetString(PyExc_RuntimeError, "output ndims is not consist with convex hull ndims \n");
        return 0;
    }

    for (ii = 0; ii < array_size; ii++) {
        inside = PYCV_hull_is_point_inside(convex_hull, convex_size, iter_o.coordinates);
        PYCV_SET_VALUE(num_type_o, po, inside);
        PYCV_ARRAY_ITERATOR_NEXT(iter_o, po);
    }
    return 1;
}

// #####################################################################################################################

PyArrayObject *PYCV_graham_scan_convex_hull(PyArrayObject *input,
                                            PyArrayObject *mask,
                                            PyArrayObject *points_array,
                                            PyArrayObject *output)
{
    PYCV_HullPoints points, stack;
    char *pc = NULL;
    npy_intp itemsize_c;
    npy_intp leftmost_index;
    npy_intp convex_size = 0, candidate_size = 0, der, ii;
    npy_intp convex_dims[2] = {-1, 2};
    PyArrayObject *convex_hull;

    if (input) {
        points = PYCV_HullPoints_AllocateFromArray(input, mask);
    } else if (points_array) {
        points = PYCV_HullPoints_AllocateFromPointsArray(points_array);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "No points are given \n");
        return NULL;
    }

    if (points.points_size < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    if (points.ndim != 2) {
        PyErr_SetString(PyExc_RuntimeError, "ND need to be 2 \n");
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    if (points.points_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    if (points_array) {
        PYCV_HULL_FIND_LEFTMOST(points.points, points.points_size, points.ndim, leftmost_index);
        if (leftmost_index != 0) {
            PYCV_hull_heapsort_swap(&points.points[0], &points.points[leftmost_index]);
        }
    }

    PYCV_hull_heapsort((points.points + 1), points.points_size - 1, points.points[0]);
    candidate_size = 1;

    for (ii = 1; ii < points.points_size - 1; ii++) {
        PYCV_HULL_DIRECTION(points.points[0], points.points[ii], points.points[ii + 1], der);
        if (der) {
            PYCV_HULL_POINTS_SET(points, points.points[ii], candidate_size);
            candidate_size++;
        }
    }

    PYCV_HULL_POINTS_SET(points, points.points[points.points_size - 1], candidate_size);
    candidate_size++;

    if (candidate_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    if (!PYCV_HullPointsInit(2, candidate_size, &stack)) {
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    for (ii = 0; ii < 3; ii++) {
        PYCV_HULL_POINTS_APPEND(stack, points.points[ii]);
        convex_size++;
    }

    for (ii = 3; ii < candidate_size; ii++) {
        while (convex_size > 2) {
            PYCV_HULL_DIRECTION(stack.points[convex_size - 2], stack.points[convex_size - 1], points.points[ii], der);
            if (der < 0) {
                break;
            } else {
                convex_size--;
            }
        }
        if (convex_size == stack.points_size) {
            PYCV_HULL_POINTS_APPEND(stack, points.points[ii]);
        } else {
            PYCV_HULL_POINTS_SET(stack, points.points[ii], convex_size);
        }
        convex_size++;
    }

    convex_dims[0] = convex_size;
    convex_hull = (PyArrayObject *)PyArray_EMPTY(2, convex_dims, NPY_INT64, 0);

    if (!convex_hull) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_c = PyArray_ITEMSIZE(convex_hull);
    pc = (void *)PyArray_DATA(convex_hull);

    for (ii = 0; ii < convex_size; ii++) {
        PYCV_HULL_SET_POINT(NPY_INT64, itemsize_c, 2, pc, stack.points[ii]);
    }

    if (output) {
        PYCV_hull_to_output_array(stack.points, convex_size, 2, output);
    }

    exit:
        PYCV_HullPointsFree(&points);
        PYCV_HullPointsFree(&stack);
        return PyErr_Occurred() ? NULL : convex_hull;
}

PyArrayObject *PYCV_jarvis_march_convex_hull(PyArrayObject *input,
                                             PyArrayObject *mask,
                                             PyArrayObject *points_array,
                                             PyArrayObject *output)
{
    PYCV_HullPoints points, stack;
    char *pc = NULL;
    int itemsize_c;
    npy_intp candidate = 0, der, ii, left, index = 0, cmp, leftmost_index;
    npy_intp convex_dims[2] = {-1, 2};
    PyArrayObject *convex_hull;

    if (input) {
        points = PYCV_HullPoints_AllocateFromArray(input, mask);
    } else if (points_array) {
        points = PYCV_HullPoints_AllocateFromPointsArray(points_array);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "No points are given \n");
        return NULL;
    }

    if (points.points_size < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    if (points.ndim != 2) {
        PyErr_SetString(PyExc_RuntimeError, "ND need to be 2 \n");
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    if (points.points_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    if (points_array) {
        PYCV_HULL_FIND_LEFTMOST(points.points, points.points_size, points.ndim, leftmost_index);
        if (leftmost_index != 0) {
            PYCV_hull_heapsort_swap(&points.points[0], &points.points[leftmost_index]);
        }
    }

    if (!PYCV_HullPointsInit(2, points.points_size, &stack)) {
        PyErr_NoMemory();
        PYCV_HullPointsFree(&points);
        return NULL;
    }

    PYCV_HULL_POINTS_APPEND(stack, points.points[0]);
    left = index;
    while (1) {
        candidate = (left + 1) % (npy_intp)points.points_size;
        for (ii = 0; ii < points.points_size; ii++) {
            if (ii == left) {
                continue;
            }
            PYCV_HULL_COMPARE(points.points[left], points.points[ii], points.points[candidate], cmp);
            if (cmp > 0) {
                candidate = ii;
            }
        }
        left = candidate;
        if (left == index) {
            break;
        }
        PYCV_HULL_POINTS_APPEND(stack, points.points[candidate]);
    }

    convex_dims[0] = stack.points_size;
    convex_hull = (PyArrayObject *)PyArray_EMPTY(2, convex_dims, NPY_INT64, 0);

    if (!convex_hull) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_c = PyArray_ITEMSIZE(convex_hull);
    pc = (void *)PyArray_DATA(convex_hull);

    for (ii = 0; ii < stack.points_size; ii++) {
        PYCV_HULL_SET_POINT(NPY_INT64, itemsize_c, 2, pc, stack.points[ii]);
    }

    if (output) {
        PYCV_hull_to_output_array(stack.points, stack.points_size, 2, output);
    }

    exit:
        PYCV_HullPointsFree(&points);
        PYCV_HullPointsFree(&stack);
        return PyErr_Occurred() ? NULL : convex_hull;
}

// #####################################################################################################################






















