#include "ops_base.h"
#include "convexhull.h"

// #####################################################################################################################

int HullPointsInit(int nd, int max_size, HullPoints *object)
{
    npy_intp ii;
    object->nd = nd;
    object->max_size = max_size;
    object->points = malloc(max_size * sizeof(npy_intp*));
    if (!object->points) {
        return 0;
    }
    for (ii = 0; ii < max_size; ii++) {
        object->points[ii] = malloc(nd * sizeof(npy_intp));
        if (!object->points[ii]) {
            return 0;
        }
    }
    object->points_size = 0;
    return 1;
}

int HullPointsFree(HullPoints *object)
{
    npy_intp ii;
    for (ii = 0; ii < object->max_size; ii++) {
        free(object->points[ii]);
    }
    free(object->points);
    object->points_size = 0;
    return 1;
}

// #####################################################################################################################

#define TYPE_CASE_HULL_SET_POINT(_NUM_TYPE, _dtype, _itemsize, _nd, _pointer, _point)                                  \
case _NUM_TYPE:                                                                                                        \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    for (_ii = 0; _ii < _nd; _ii++) {                                                                                  \
        *(_dtype *)_pointer = (_dtype)_point[_ii];                                                                     \
        _pointer += _itemsize;                                                                                         \
    }                                                                                                                  \
}                                                                                                                      \
break

#define HULL_SET_POINT(_NUM_TYPE, _itemsize, _nd, _pointer, _point)                                                    \
{                                                                                                                      \
    switch (_NUM_TYPE) {                                                                                               \
        TYPE_CASE_HULL_SET_POINT(NPY_BOOL, npy_bool, _itemsize, _nd, _pointer, _point);                                \
        TYPE_CASE_HULL_SET_POINT(NPY_UBYTE, npy_ubyte, _itemsize, _nd, _pointer, _point);                              \
        TYPE_CASE_HULL_SET_POINT(NPY_USHORT, npy_ushort, _itemsize, _nd, _pointer, _point);                            \
        TYPE_CASE_HULL_SET_POINT(NPY_UINT, npy_uint, _itemsize, _nd, _pointer, _point);                                \
        TYPE_CASE_HULL_SET_POINT(NPY_ULONG, npy_ulong, _itemsize, _nd, _pointer, _point);                              \
        TYPE_CASE_HULL_SET_POINT(NPY_ULONGLONG, npy_ulonglong, _itemsize, _nd, _pointer, _point);                      \
        TYPE_CASE_HULL_SET_POINT(NPY_BYTE, npy_byte, _itemsize, _nd, _pointer, _point);                                \
        TYPE_CASE_HULL_SET_POINT(NPY_SHORT, npy_short, _itemsize, _nd, _pointer, _point);                              \
        TYPE_CASE_HULL_SET_POINT(NPY_INT, npy_int, _itemsize, _nd, _pointer, _point);                                  \
        TYPE_CASE_HULL_SET_POINT(NPY_LONG, npy_long, _itemsize, _nd, _pointer, _point);                                \
        TYPE_CASE_HULL_SET_POINT(NPY_LONGLONG, npy_longlong, _itemsize, _nd, _pointer, _point);                        \
        TYPE_CASE_HULL_SET_POINT(NPY_FLOAT, npy_float, _itemsize, _nd, _pointer, _point);                              \
        TYPE_CASE_HULL_SET_POINT(NPY_DOUBLE, npy_double, _itemsize, _nd, _pointer, _point);                            \
    }                                                                                                                  \
}

// #####################################################################################################################

#define HULL_CROSS_PRODUCT(_p0, _p1, _p2, _out)                                                                        \
{                                                                                                                      \
    _out = (_p2[0] - _p0[0]) * (_p1[1] - _p0[1]) - (_p1[0] - _p0[0]) * (_p2[1] - _p0[1]);                              \
}

#define HULL_DISTANCE(_p1, _p2, _out)                                                                                  \
{                                                                                                                      \
    npy_intp _d1, _d2;                                                                                                 \
    _d1 = _p1[0] > _p2[0] ? _p1[0] - _p2[0] : _p2[0] - _p1[0];                                                         \
    _d2 = _p1[1] > _p2[1] ? _p1[1] - _p2[1] : _p2[1] - _p1[1];                                                         \
    _out = _d1 + _d2;                                                                                                  \
}

#define HULL_DIRECTION(_p0, _p1, _p2, _out)                                                                            \
{                                                                                                                      \
    npy_intp _ori;                                                                                                     \
    HULL_CROSS_PRODUCT(_p0, _p1, _p2, _ori);                                                                           \
    if (_ori == 0) {                                                                                                   \
        _out = 0;                                                                                                      \
    } else if (_ori < 0) {                                                                                             \
        _out = -1;                                                                                                     \
    } else {                                                                                                           \
        _out = 1;                                                                                                      \
    }                                                                                                                  \
}

#define HULL_COMPARE(_p0, _p1, _p2, _out)                                                                              \
{                                                                                                                      \
    npy_intp _ori;                                                                                                     \
    HULL_CROSS_PRODUCT(_p0, _p1, _p2, _ori);                                                                           \
    if (_ori < 0) {                                                                                                    \
        _out = -1;                                                                                                     \
    } else if (_ori > 0) {                                                                                             \
        _out = 1;                                                                                                      \
    } else {                                                                                                           \
        npy_intp _dist1, _dist2;                                                                                       \
        HULL_DISTANCE(_p0, _p1, _dist1);                                                                               \
        HULL_DISTANCE(_p0, _p2, _dist2);                                                                               \
        _out = _dist1 < _dist2 ? -1 : 1;                                                                               \
    }                                                                                                                  \
}

// #####################################################################################################################

static void hull_heapsort_swap(npy_intp **i1, npy_intp **i2)
{
    npy_intp *tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static void hull_heapsort_heapify(npy_intp **points, npy_intp n, npy_intp i, npy_intp *p0)
{
    npy_intp cmp, largest = i, l = 2 * i + 1, r = 2 * i + 2;

    if (l < n) {
        HULL_COMPARE(p0, points[largest], points[l], cmp);
        largest = cmp < 0 ? l : largest;
    }

    if (r < n) {
        HULL_COMPARE(p0, points[largest], points[r], cmp);
        largest = cmp < 0 ? r : largest;
    }

    if (largest != i) {
        hull_heapsort_swap(&points[i], &points[largest]);
        hull_heapsort_heapify(points, n, largest, p0);
    }
}

static void hull_heapsort(npy_intp **points, npy_intp n, npy_intp *p0)
{
    npy_intp i;

    for (i = n / 2 - 1; i >= 0; i--) {
        hull_heapsort_heapify(points, n, i, p0);
    }

    for (i = n - 1; i >= 0; i--) {
        hull_heapsort_swap(&points[0], &points[i]);
        hull_heapsort_heapify(points, i, 0, p0);
    }
}

// #####################################################################################################################

PyArrayObject *ops_graham_scan_convex_hull(PyArrayObject *input, PyArrayObject *mask)
{
    npy_intp nd = 2;
    HullPoints points, stack;
    ArrayIter iter_i, iter_ma;
    char *pi = NULL, *pc = NULL, *pm = NULL;
    int num_type_i, num_type_m, num_type_c, itemsize_c;

    npy_intp array_size, convex_size = 0, candidate_size = 0, der, ii;
    npy_double inp_val;
    npy_bool ma_vla = 1;

    npy_intp convex_dims[2] = {-1, 2};
    PyArrayObject *convex_hull;

    num_type_i = PyArray_TYPE(input);
    array_size = PyArray_SIZE(input);

    if (!HullPointsInit(nd, array_size, &points)) {
        goto exit;
    }

    pi = (void *)PyArray_DATA(input);
    ArrayIterInit(input, &iter_i);

    if (mask) {
        num_type_m = mask ? PyArray_TYPE(mask) : -1;
        pm = (void *)PyArray_DATA(mask);
        ArrayIterInit(mask, &iter_ma);
    }

    for (ii = 0; ii < array_size; ii++) {
        if (mask) {
            GET_VALUE_AS(num_type_m, npy_bool, pm, ma_vla);
            ARRAY_ITER_NEXT(iter_ma, pm);
        }
        GET_VALUE_AS(num_type_i, npy_double, pi, inp_val);
        if (ma_vla && fabs(inp_val) > DBL_EPSILON) {
            HULL_POINTS_APPEND(points, iter_i.coordinates);
        }
        ARRAY_ITER_NEXT(iter_i, pi);
    }

    if (points.points_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        goto exit;
    }

    hull_heapsort((points.points + 1), points.points_size - 1, points.points[0]);
    candidate_size = 1;

    for (ii = 1; ii < points.points_size - 1; ii++) {
        HULL_DIRECTION(points.points[0], points.points[ii], points.points[ii + 1], der);
        if (der) {
            HULL_POINTS_SET(points, points.points[ii], candidate_size);
            candidate_size++;
        }
    }

    HULL_POINTS_SET(points, points.points[points.points_size - 1], candidate_size);
    candidate_size++;

    if (candidate_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        goto exit;
    }

    if (!HullPointsInit(nd, candidate_size, &stack)) {
        goto exit;
    }

    for (ii = 0; ii < 3; ii++) {
        HULL_POINTS_APPEND(stack, points.points[ii]);
        convex_size++;
    }

    for (ii = 3; ii < candidate_size; ii++) {
        while (convex_size > 2) {
            HULL_DIRECTION(stack.points[convex_size - 2], stack.points[convex_size - 1], points.points[ii], der);
            if (der < 0) {
                break;
            } else {
                convex_size--;
            }
        }
        if (convex_size == stack.points_size) {
            HULL_POINTS_APPEND(stack, points.points[ii]);
        } else {
            HULL_POINTS_SET(stack, points.points[ii], convex_size);
        }
        convex_size++;
    }

    convex_dims[0] = convex_size;
    convex_hull = (PyArrayObject *)PyArray_EMPTY(nd, convex_dims, NPY_INT64, 0);

    if (!convex_hull) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_c = PyArray_ITEMSIZE(convex_hull);
    pc = (void *)PyArray_DATA(convex_hull);

    for (ii = 0; ii < convex_size; ii++) {
        HULL_SET_POINT(NPY_INT64, itemsize_c, nd, pc, stack.points[ii]);
    }

    exit:
        HullPointsFree(&points);
        HullPointsFree(&stack);
        return PyErr_Occurred() ? NULL : convex_hull;
}

PyArrayObject *ops_jarvis_march_convex_hull(PyArrayObject *input, PyArrayObject *mask)
{
    npy_intp nd = 2;
    HullPoints points, stack;
    ArrayIter iter_i, iter_ma;
    char *pi = NULL, *pc = NULL, *pm = NULL;
    int num_type_i, num_type_m, num_type_c, itemsize_c;

    npy_intp array_size, candidate = 0, der, ii, left, index = 0, cmp;
    npy_double inp_val;
    npy_bool ma_vla = 1;

    npy_intp convex_dims[2] = {-1, 2};
    PyArrayObject *convex_hull;

    num_type_i = PyArray_TYPE(input);
    array_size = PyArray_SIZE(input);

    if (!HullPointsInit(nd, array_size, &points)) {
        goto exit;
    }

    pi = (void *)PyArray_DATA(input);
    ArrayIterInit(input, &iter_i);

    if (mask) {
        num_type_m = mask ? PyArray_TYPE(mask) : -1;
        pm = (void *)PyArray_DATA(mask);
        ArrayIterInit(mask, &iter_ma);
    }

    for (ii = 0; ii < array_size; ii++) {
        if (mask) {
            GET_VALUE_AS(num_type_m, npy_bool, pm, ma_vla);
            ARRAY_ITER_NEXT(iter_ma, pm);
        }
        GET_VALUE_AS(num_type_i, npy_double, pi, inp_val);
        if (ma_vla && fabs(inp_val) > DBL_EPSILON) {
            HULL_POINTS_APPEND(points, iter_i.coordinates);
        }
        ARRAY_ITER_NEXT(iter_i, pi);
    }

    if (points.points_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        goto exit;
    }

    if (!HullPointsInit(nd, points.points_size, &stack)) {
        goto exit;
    }

    HULL_POINTS_APPEND(stack, points.points[0]);

    left = index;

    while (1) {
        candidate = (left + 1) % (npy_intp)points.points_size;
        for (ii = 0; ii < points.points_size; ii++) {
            if (ii == left) {
                continue;
            }
            HULL_COMPARE(points.points[ii], points.points[candidate], points.points[left], cmp);
            if (cmp > 0) {
                candidate = ii;
            }
        }
        left = candidate;
        if (left == index) {
            break;
        }
        HULL_POINTS_APPEND(stack, points.points[candidate]);
    }

    convex_dims[0] = stack.points_size;
    convex_hull = (PyArrayObject *)PyArray_EMPTY(nd, convex_dims, NPY_INT64, 0);

    if (!convex_hull) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_c = PyArray_ITEMSIZE(convex_hull);
    pc = (void *)PyArray_DATA(convex_hull);

    for (ii = 0; ii < stack.points_size; ii++) {
        HULL_SET_POINT(NPY_INT64, itemsize_c, nd, pc, stack.points[ii]);
    }

    exit:
        HullPointsFree(&points);
        HullPointsFree(&stack);
        return PyErr_Occurred() ? NULL : convex_hull;
}