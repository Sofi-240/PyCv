#ifndef CONVEXHULL_H
#define CONVEXHULL_H

// #####################################################################################################################

typedef struct {
    int nd;
    int max_size;
    int points_size;
    npy_intp **points;
} HullPoints;

int HullPointsInit(int nd, int max_size, HullPoints *object);

#define HULL_POINTS_APPEND(_object, _point)                                                                           \
{                                                                                                                      \
    int _ii;                                                                                                           \
    if ((_object).max_size > (_object).points_size) {                                                                  \
        for (_ii = 0; _ii < (_object).nd; _ii++) {                                                                     \
            (_object).points[(_object).points_size][_ii] = _point[_ii];                                                \
        }                                                                                                              \
    }                                                                                                                  \
    (_object).points_size++;                                                                                           \
}

#define HULL_POINTS_SET(_object, _point, _index)                                                                       \
{                                                                                                                      \
    int _ii;                                                                                                           \
    if ((_object).points_size > _index) {                                                                              \
        for (_ii = 0; _ii < (_object).nd; _ii++) {                                                                     \
            (_object).points[_index][_ii] = _point[_ii];                                                               \
        }                                                                                                              \
    }                                                                                                                  \
}

#define HULL_POINTS_PRINT(_object)                                                                                     \
{                                                                                                                      \
    npy_intp _ii, _jj;                                                                                                 \
    for (_ii = 0; _ii < (_object).points_size; _ii++) {                                                                \
        printf("[");                                                                                                   \
        for (_jj = 0; _jj < (_object).nd; _jj++) {                                                                     \
            printf("%" NPY_INTP_FMT ", ", (_object).points[_ii][_jj]);                                                 \
        }                                                                                                              \
        printf("]\n");                                                                                                 \
    }                                                                                                                  \
    printf("\n");                                                                                                      \
}

int HullPointsFree(HullPoints *object);

HullPoints HullPoints_AllocateFromArray(PyArrayObject *input, PyArrayObject *mask);

HullPoints HullPoints_AllocateFromPointsArray(PyArrayObject *points_array);

// #####################################################################################################################

typedef enum {
    HULL_GRAM_SCAN = 1,
    HULL_GIFT_WRAPPING = 2,
} HullMode;

PyArrayObject *ops_graham_scan_convex_hull(PyArrayObject *input, PyArrayObject *mask, PyArrayObject *points_array, PyArrayObject *output);

PyArrayObject *ops_jarvis_march_convex_hull(PyArrayObject *input, PyArrayObject *mask, PyArrayObject *points_array, PyArrayObject *output);

// #####################################################################################################################


#endif