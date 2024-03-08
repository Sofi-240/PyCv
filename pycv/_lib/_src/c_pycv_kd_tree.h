#ifndef C_PYCV_KD_TREE_H
#define C_PYCV_KD_TREE_H

// #####################################################################################################################

typedef struct CKDnode {
    PyObject_HEAD
    int start_index;
    int end_index;
    int children;
    int split_dim;
    double split_val;
    int lesser_index;
    int higher_index;
    struct CKDnode *lesser;
    struct CKDnode *higher;
    int level;
} CKDnode;

typedef struct {
    PyObject_HEAD
    int m;
    int n;
    int leafsize;
    PyArrayObject *data;
    PyArrayObject *dims_min;
    PyArrayObject *dims_max;
    PyArrayObject *indices;
    PyObject *tree_list;
    CKDnode *tree;
    int size;
} CKDtree;


// ######################################### Helpers ###################################################################
// ----------------------------------  Hyperparameter  -----------------------------------------------------------------

typedef struct {
    double pnorm;
    int is_inf;
    double eps;
    double eps_frac;
    double bound;
} CKDHyp;

// ----------------------------------  KNN query helpers  --------------------------------------------------------------
// *************************************  Heap  ************************************************************************

union CKDheap_contents {
    int index;
    void *data_ptr;
};

typedef struct {
    double priority;
    union CKDheap_contents contents;
} CKDheap_item;

typedef struct {
    CKDheap_item *heap;
    int n;
    int _loc_n;
} CKDheap;

#define CKDHEAP_INIT {NULL, 0, 0}

// **********************************  Knn query stack  ****************************************************************

typedef struct {
    double min_distance;
    double *split_distance;
    CKDnode *node;
} CKDknn_query;


typedef struct {
    CKDknn_query *stack;
    int n;
    int _loc_n;
} CKDknn_stack;

#define CKDKNN_STACK_INIT {NULL, 0, 0}

// ----------------------------------  Ball query helpers  -------------------------------------------------------------
// **********************************  Ball query stack  ***************************************************************

typedef struct {
    int split_dim;
    double min_distance;
    double max_distance;
    double split_min_distance;
    double split_max_distance;
} CKDball_query;

typedef struct {
    double *bound_min;
    double *bound_max;
    double min_distance;
    double max_distance;
    CKDball_query *stack;
    int n;
    int _loc_n;
} CKDball_stack;

#define CKDBALL_STACK_INIT {NULL, NULL, 0, 0, NULL, 0, 0}

// -------------------------------  query results  ---------------------------------------------------------------------

typedef struct {
    PyObject *indices;
    PyObject *distance;
    PyObject *_ind;
    PyObject *_dist;
} CKDresults;

#define CKDRESULTS_INIT {NULL, NULL, NULL, NULL}

// #####################################################################################################################
// ----------------------------------  communication  ------------------------------------------------------------------

void CKDnodePy_dealloc(CKDnode *self);

PyObject *CKDnodePy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CKDnodePy_init(CKDnode *self, PyObject *args, PyObject *kw);

void CKDtreePy_dealloc(CKDtree *self);

PyObject *CKDtreePy_new(PyTypeObject *type, PyObject *args, PyObject *kw);

int CKDtreePy_init(CKDtree *self, PyObject *args, PyObject *kw);

// #####################################################################################################################

PyObject *CKDtree_knn_query(CKDtree *self, PyObject *args);

PyObject *CKDtree_ball_point_query(CKDtree *self, PyObject *args);

// #####################################################################################################################


#endif
