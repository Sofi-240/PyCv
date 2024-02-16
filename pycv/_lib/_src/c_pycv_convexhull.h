#ifndef C_PYCV_CONVEXHULL_H
#define C_PYCV_CONVEXHULL_H

// #####################################################################################################################

typedef struct {
    npy_intp ndim;
    npy_intp max_size;
    npy_intp points_size;
    npy_intp **points;
} PYCV_HullPoints;

int PYCV_HullPointsInit(npy_intp ndim, npy_intp max_size, PYCV_HullPoints *object);

#define PYCV_HULL_POINTS_APPEND(_object, _point)                                                                       \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    if ((_object).max_size > (_object).points_size) {                                                                  \
        for (_ii = 0; _ii < (_object).ndim; _ii++) {                                                                   \
            (_object).points[(_object).points_size][_ii] = _point[_ii];                                                \
        }                                                                                                              \
    }                                                                                                                  \
    (_object).points_size++;                                                                                           \
}

#define PYCV_HULL_POINTS_SET(_object, _point, _index)                                                                  \
{                                                                                                                      \
    npy_intp _ii;                                                                                                      \
    if ((_object).points_size > _index) {                                                                              \
        for (_ii = 0; _ii < (_object).ndim; _ii++) {                                                                   \
            (_object).points[_index][_ii] = _point[_ii];                                                               \
        }                                                                                                              \
    }                                                                                                                  \
}

int PYCV_HullPointsFree(PYCV_HullPoints *object);

PYCV_HullPoints PYCV_HullPoints_AllocateFromArray(PyArrayObject *input, PyArrayObject *mask);

PYCV_HullPoints PYCV_HullPoints_AllocateFromPointsArray(PyArrayObject *points_array);

// #####################################################################################################################

typedef enum {
    PYCV_HULL2D_GRAM_SCAN = 1,
    PYCV_HULL2D_GIFT_WRAPPING = 2,
} PYCV_HullMode;

// #####################################################################################################################

PyArrayObject *PYCV_graham_scan_convex_hull(PyArrayObject *input,
                                            PyArrayObject *mask,
                                            PyArrayObject *points_array,
                                            PyArrayObject *output);

PyArrayObject *PYCV_jarvis_march_convex_hull(PyArrayObject *input,
                                             PyArrayObject *mask,
                                             PyArrayObject *points_array,
                                             PyArrayObject *output);


// #####################################################################################################################

int PYCV_convex_hull_image(PyArrayObject *output, PyArrayObject *convex_hull);

// #####################################################################################################################


#endif