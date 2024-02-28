#ifndef C_PYCV_KD_TREE_H
#define C_PYCV_KD_TREE_H

// #####################################################################################################################

#define kdtree_intp npy_intp
#define kdtree_double npy_double

// #####################################################################################################################

typedef struct {
    char *ptr;
    char *base;
    int numtype;
    kdtree_intp itemsize;
    PyArrayObject *object;
} KDarray;

void PYCV_KDarray_init(KDarray *self, PyArrayObject *array);

// #####################################################################################################################

typedef struct KDnode {
    kdtree_intp start_index;
    kdtree_intp end_index;
    kdtree_intp children;
    kdtree_intp split_dim;
    kdtree_double split_val;
    kdtree_intp lesser_index;
    kdtree_intp higher_index;
    struct KDnode *lesser;
    struct KDnode *higher;
    kdtree_intp level;
} KDnode;

typedef struct {
    kdtree_intp m;
    kdtree_intp n;
    kdtree_intp leafsize;
    KDarray data;
    KDarray dims_min;
    KDarray dims_max;
    KDarray indices;
    KDnode *tree_list;
    KDnode *tree;
    kdtree_intp size;
} KDtree;


union KDheap_contents {
    kdtree_intp index;
    void *data_ptr;
};

typedef struct {
    kdtree_double priority;
    union KDheap_contents contents;
} KDheap_item;

typedef struct {
    KDheap_item *heap;
    kdtree_intp n;
    kdtree_intp _has_n;
} KDheap;

typedef struct {
    kdtree_double min_distance;
    kdtree_double *split_distance;
    KDnode *node;
} KDquery_item;

typedef struct {
    KDquery_item *buffer;
    kdtree_intp n;
    kdtree_intp _has_n;
} KDquery_buffer;

// #####################################################################################################################

int PYCV_KDtree_build(KDtree *self,
                      PyArrayObject *data,
                      PyArrayObject *dims_min,
                      PyArrayObject *dims_max,
                      PyArrayObject *indices,
                      kdtree_intp leafsize);

void PYCV_KDtree_free(KDtree *self);

// #####################################################################################################################

int PYCV_KDtree_query_knn(KDtree tree,
                          PyArrayObject *points,
                          KDarray k,
                          kdtree_intp p,
                          int is_inf,
                          kdtree_double distance_max,
                          kdtree_double epsilon,
                          KDarray dist,
                          KDarray indices,
                          KDarray slice);

// #####################################################################################################################

typedef struct {
    kdtree_intp *list;
    kdtree_intp n;
    kdtree_intp _loc_n;
} KDball_results;

typedef struct {
    kdtree_intp split_dim;
    kdtree_double min_distance;
    kdtree_double max_distance;
    kdtree_double split_min_distance;
    kdtree_double split_max_distance;
} KDtracking_item;

typedef struct {
    kdtree_intp m;
    kdtree_intp p;
    int is_inf;
    kdtree_double *bound_min;
    kdtree_double *bound_max;
    kdtree_double min_distance;
    kdtree_double max_distance;
    KDtracking_item *stack;
    kdtree_intp n;
    kdtree_intp _loc_n;
} KDball_tracking;

// #####################################################################################################################

int PYCV_query_ball_points(KDtree tree,
                           PyArrayObject *points,
                           PyArrayObject *radius,
                           kdtree_intp p,
                           int is_inf,
                           kdtree_double epsilon,
                           KDball_results *indices,
                           KDarray slice);

// #####################################################################################################################


#endif
