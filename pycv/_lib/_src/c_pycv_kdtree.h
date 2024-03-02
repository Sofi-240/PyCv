#ifndef C_PYCV_KD_TREE_H
#define C_PYCV_KD_TREE_H

// #####################################################################################################################

#define kd_intp npy_intp
#define kd_double npy_double

// #####################################################################################################################

typedef struct {
    char *base;
    char *ptr;
    int numtype;
    kd_intp itemsize;
    PyArrayObject *object;
} KDarray;

typedef struct KDnode {
    kd_intp start_index;
    kd_intp end_index;
    kd_intp children;
    kd_intp split_dim;
    kd_double split_val;
    kd_intp lesser_index;
    kd_intp higher_index;
    struct KDnode *lesser;
    struct KDnode *higher;
    kd_intp level;
} KDnode;

typedef struct {
    kd_intp m;
    kd_intp n;
    kd_intp leafsize;
    KDarray data;
    KDarray dims_min;
    KDarray dims_max;
    KDarray indices;
    KDnode *tree_list;
    KDnode *tree;
    kd_intp size;
} KDtree;

// ######################################### Helpers ###################################################################

// ----------------------------------  KNN query helpers  --------------------------------------------------------------
// *************************************  Heap  ************************************************************************

union KDheap_contents {
    kd_intp index;
    void *data_ptr;
};

typedef struct {
    kd_double priority;
    union KDheap_contents contents;
} KDheap_item;

typedef struct {
    KDheap_item *heap;
    kd_intp n;
    kd_intp _loc_n;
} KDheap;

#define KDHEAP_INIT {NULL, 0, 0}

// **********************************  Knn query stack  ****************************************************************

typedef struct {
    kd_double min_distance;
    kd_double *split_distance;
    KDnode *node;
} KDknn_query;


typedef struct {
    KDknn_query *stack;
    kd_intp n;
    kd_intp _loc_n;
} KDknn_stack;

#define KDKNN_STACK_INIT {NULL, 0, 0}

// ----------------------------------  Ball query helpers  -------------------------------------------------------------
// **********************************  Ball query stack  ***************************************************************

typedef struct {
    kd_intp split_dim;
    kd_double min_distance;
    kd_double max_distance;
    kd_double split_min_distance;
    kd_double split_max_distance;
} KDball_query;

typedef struct {
    kd_double *bound_min;
    kd_double *bound_max;
    kd_double min_distance;
    kd_double max_distance;
    KDball_query *stack;
    kd_intp n;
    kd_intp _loc_n;
} KDball_stack;

#define KDBALL_STACK_INIT {NULL, NULL, 0, 0, NULL, 0, 0}

// ***************************************  query results  *************************************************************

typedef struct {
    kd_intp *indices;
    kd_double *distance;
    kd_intp n;
    kd_intp _tree_size;
} KDresults;

#define KDRESULTS_INIT {NULL, NULL, 0, 0}

// #####################################################################################################################
// **********************************  communication  ******************************************************************

int PYCV_input_to_KDtree(PyObject *pytree, KDtree *tree);

void PYCV_KDtree_free(KDtree *self);

// #####################################################################################################################
// **********************************  python calls  *******************************************************************

int PYCV_KDtree_build(KDtree *self, PyArrayObject *data, PyArrayObject *dims_min, PyArrayObject *dims_max,
                      PyArrayObject *indices, kd_intp leafsize, PyObject **output_list);

int PYCV_KDtree_knn_query(KDtree *self, PyArrayObject *points, PyArrayObject *k,
                          kd_intp pnorm, int is_inf, kd_double epsilon, kd_double distance_max,
                          PyObject **output);

int PYCV_query_ball_points(KDtree *self, PyArrayObject *points, PyArrayObject *radius, kd_intp pnorm,
                           int is_inf, kd_double epsilon, PyObject **output);

// #####################################################################################################################


#endif














