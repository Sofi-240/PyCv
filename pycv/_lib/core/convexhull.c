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
        object->points_size = -1;
        return 0;
    }
    for (ii = 0; ii < max_size; ii++) {
        object->points[ii] = malloc(nd * sizeof(npy_intp));
        if (!object->points[ii]) {
            object->points_size = -1;
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

HullPoints HullPoints_AllocateFromArray(PyArrayObject *input, PyArrayObject *mask)
{
    npy_intp nd, array_size, ii;
    ArrayIter iter_i, iter_ma;
    char *pi = NULL, *pm = NULL;
    int num_type_i, num_type_m;
    npy_double inp_val;
    npy_bool ma_vla = 1;
    HullPoints object;

    num_type_i = PyArray_TYPE(input);
    nd = PyArray_NDIM(input);
    array_size = PyArray_SIZE(input);

    if (!HullPointsInit(nd, array_size, &object)) {
        return object;
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
            HULL_POINTS_APPEND(object, iter_i.coordinates);
        }
        ARRAY_ITER_NEXT(iter_i, pi);
    }
    return object;
}

HullPoints HullPoints_AllocateFromPointsArray(PyArrayObject *points_array)
{
    npy_intp nd, array_size, ii, jj, coordinates[NPY_MAXDIMS];
    ArrayIter iter_i;
    char *pi = NULL;
    int num_type_i;
    HullPoints object;

    num_type_i = PyArray_TYPE(points_array);
    nd = PyArray_DIM(points_array, 1);
    array_size = PyArray_DIM(points_array, 0);

    if (!HullPointsInit(nd, array_size, &object)) {
        return object;
    }

    pi = (void *)PyArray_DATA(points_array);
    ArrayIterInit(points_array, &iter_i);

    for (ii = 0; ii < array_size; ii++) {
        for (jj = 0; jj < nd; jj++) {
            GET_VALUE_AS(num_type_i, npy_intp, pi, coordinates[jj]);
            ARRAY_ITER_NEXT(iter_i, pi);
        }
        HULL_POINTS_APPEND(object, coordinates);
    }
    return object;
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

static npy_intp hull_find_leftmost(npy_intp **points, npy_intp points_size, npy_intp nd)
{
    npy_intp ll = 0, ii, jj, change = 0;
    if (!points_size) {
        return -1;
    }
    for (ii = 0; ii < points_size; ii++) {
        if (points[ii][0] < points[ll][0]) {
            ll = ii;
        } else if (points[ii][0] == points[ll][0]) {
            change = 1;
            for (jj = 1; jj < nd; jj++) {
                if (points[ii][jj] > points[ll][jj]) {
                    change = 0;
                    break;
                }
            }
            if (change) {
                ll = ii;
            }
        }
    }
    return ll;
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

static npy_intp hull_is_point_inside(npy_intp **convex_hull, npy_intp convex_size, npy_intp nd, npy_intp *point)
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

static int hull_to_output_array(npy_intp **convex_hull, npy_intp convex_size, npy_intp nd, PyArrayObject *output)
{
    npy_intp array_size, ii;
    ArrayIter iter_o;
    char *po = NULL;
    int num_type_o;
    npy_intp inside;

    num_type_o = PyArray_TYPE(output);
    array_size = PyArray_SIZE(output);

    po = (void *)PyArray_DATA(output);
    ArrayIterInit(output, &iter_o);

    if (iter_o.nd_m1 != nd - 1) {
        PyErr_SetString(PyExc_RuntimeError, "output ndims is not consist with convex hull ndims \n");
        return 0;
    }

    for (ii = 0; ii < array_size; ii++) {
        inside = hull_is_point_inside(convex_hull, convex_size, nd, iter_o.coordinates);
        SET_VALUE_TO(num_type_o, po, inside);
        ARRAY_ITER_NEXT(iter_o, po);
    }
    return 1;
}

// #####################################################################################################################

PyArrayObject *ops_graham_scan_convex_hull(PyArrayObject *input, PyArrayObject *mask, PyArrayObject *points_array, PyArrayObject *output)
{
    HullPoints points, stack;
    char *pc = NULL;
    int itemsize_c;
    npy_intp leftmost_index;

    npy_intp convex_size = 0, candidate_size = 0, der, ii;

    npy_intp convex_dims[2] = {-1, 2};
    PyArrayObject *convex_hull;

    if (input) {
        points = HullPoints_AllocateFromArray(input, mask);
    } else if (points_array) {
        points = HullPoints_AllocateFromPointsArray(points_array);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "No points are given \n");
        return NULL;
    }

    if (points.points_size < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    if (points.nd != 2) {
        PyErr_SetString(PyExc_RuntimeError, "ND need to be 2 \n");
        HullPointsFree(&points);
        return NULL;
    }

    if (points.points_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        HullPointsFree(&points);
        return NULL;
    }

    if (points_array) {
        leftmost_index = hull_find_leftmost(points.points, points.points_size, 2);
        if (leftmost_index != 0) {
            hull_heapsort_swap(&points.points[0], &points.points[leftmost_index]);
        }
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
        HullPointsFree(&points);
        return NULL;
    }

    if (!HullPointsInit(2, candidate_size, &stack)) {
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
    convex_hull = (PyArrayObject *)PyArray_EMPTY(2, convex_dims, NPY_INT64, 0);

    if (!convex_hull) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_c = PyArray_ITEMSIZE(convex_hull);
    pc = (void *)PyArray_DATA(convex_hull);

    for (ii = 0; ii < convex_size; ii++) {
        HULL_SET_POINT(NPY_INT64, itemsize_c, 2, pc, stack.points[ii]);
    }

    if (output) {
        hull_to_output_array(stack.points, convex_size, 2, output);
    }

    exit:
        HullPointsFree(&points);
        HullPointsFree(&stack);
        return PyErr_Occurred() ? NULL : convex_hull;
}

PyArrayObject *ops_jarvis_march_convex_hull(PyArrayObject *input, PyArrayObject *mask, PyArrayObject *points_array, PyArrayObject *output)
{
    HullPoints points, stack;
    char *pc = NULL;
    int itemsize_c;

    npy_intp candidate = 0, der, ii, left, index = 0, cmp, leftmost_index;

    npy_intp convex_dims[2] = {-1, 2};
    PyArrayObject *convex_hull;

    if (input) {
        points = HullPoints_AllocateFromArray(input, mask);
    } else if (points_array) {
        points = HullPoints_AllocateFromPointsArray(points_array);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "No points are given \n");
        return NULL;
    }

    if (points.points_size < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    if (points.nd != 2) {
        PyErr_SetString(PyExc_RuntimeError, "ND need to be 2 \n");
        HullPointsFree(&points);
        return NULL;
    }

    if (points.points_size < 3) {
        PyErr_SetString(PyExc_RuntimeError, "Convex hull is empty \n");
        HullPointsFree(&points);
        return NULL;
    }

    if (points_array) {
        leftmost_index = hull_find_leftmost(points.points, points.points_size, 2);
        if (leftmost_index != 0) {
            hull_heapsort_swap(&points.points[0], &points.points[leftmost_index]);
        }
    }

    if (!HullPointsInit(2, points.points_size, &stack)) {
        PyErr_NoMemory();
        HullPointsFree(&points);
        return NULL;
    }

    HULL_POINTS_APPEND(stack, points.points[0]);
    left = index;
    while (1) {
        candidate = (left + 1) % (npy_intp)points.points_size;
        for (ii = 0; ii < points.points_size; ii++) {
            if (ii == left) {
                continue;
            }
            HULL_COMPARE(points.points[left], points.points[ii], points.points[candidate], cmp);
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
    convex_hull = (PyArrayObject *)PyArray_EMPTY(2, convex_dims, NPY_INT64, 0);

    if (!convex_hull) {
        PyErr_SetString(PyExc_RuntimeError, "Error: creating array \n");
        goto exit;
    }

    itemsize_c = PyArray_ITEMSIZE(convex_hull);
    pc = (void *)PyArray_DATA(convex_hull);

    for (ii = 0; ii < stack.points_size; ii++) {
        HULL_SET_POINT(NPY_INT64, itemsize_c, 2, pc, stack.points[ii]);
    }

    if (output) {
        hull_to_output_array(stack.points, stack.points_size, 2, output);
    }

    exit:
        HullPointsFree(&points);
        HullPointsFree(&stack);
        return PyErr_Occurred() ? NULL : convex_hull;
}

// #####################################################################################################################

