#include "c_pycv_base.h"
#include "c_pycv_kd_tree.h"

// #####################################################################################################################

#define CKD_SET_DOUBLE(_ptr, _val) (*(npy_double *)_ptr = (npy_double)_val)

#define CKD_SET_INT(_ptr, _val) (*(npy_longlong *)_ptr = (npy_longlong)_val)

#define CKD_SET_DOUBLE_PTR(_dst, _src) (*(npy_double *)_dst = *((npy_double *)_src))

#define CKD_SET_INT_PTR(_dst, _src) (*(npy_longlong *)_dst = *((npy_longlong *)_src))

#define CKD_GET_DOUBLE(_ptr) (double)(*((npy_double *)_ptr))

#define CKD_GET_INT(_ptr) (int)(*((npy_longlong *)_ptr))

#define CKD_GET_DOUBLE_BY_INDEX(_data, _ind, _m) (double)(*((npy_double *)(_data + (8 * _m * (*((npy_longlong *)_ind))))))

#define CKD_PTR_GOTO(_ptr, _ind) _ptr + (_ind * 8)

#define CKD_PTR_SET_GOTO(_ptr, _ind) (_ptr += _ind * 8)

#define CKD_PTR_NEXT(_ptr) _ptr + 8

#define CKD_PTR_SET_NEXT(_ptr) (_ptr += 8)

#define CKD_PTR_PREV(_ptr) _ptr - 8

#define CKD_PTR_SET_PREV(_ptr) (_ptr -= 8)

#define CKD_PTR_GOTO_BY_INDEX(_data, _ind, _m) _data + (8 * _m * (*((npy_longlong *)_ind)))

// #####################################################################################################################

#define CKD_CMP_DOUBLE_LT_PTR(_p1, _p2) *(npy_double *)_p1 < *((npy_double *)_p2) ? 1 : 0

#define CKD_CMP_DOUBLE_LE_PTR(_p1, _p2) *(npy_double *)_p1 <= *((npy_double *)_p2) ? 1 : 0

#define CKD_CMP_DOUBLE_GT_PTR(_p1, _p2) *(npy_double *)_p1 > *((npy_double *)_p2) ? 1 : 0

#define CKD_CMP_DOUBLE_GE_PTR(_p1, _p2) *(npy_double *)_p1 >= *((npy_double *)_p2) ? 1 : 0

#define CKD_CMP_DOUBLE_EQ_PTR(_p1, _p2) *(npy_double *)_p1 == *((npy_double *)_p2) ? 1 : 0

#define CKD_CMP_DOUBLE_NE_PTR(_p1, _p2) *(npy_double *)_p1 == *((npy_double *)_p2) ? 0 : 1

// #####################################################################################################################

static void ckd_copy_data_point(char *data, int m, double *point)
{
    int ii;
    char *dp = data;
    for (ii = 0; ii < m; ii++) {
        *(point + ii) = CKD_GET_DOUBLE(dp);
        CKD_PTR_SET_NEXT(dp);
    }
}

// #####################################################################################################################
// ----------------------------------  Build tree helpers  -------------------------------------------------------------

static void ckd_swap(char *i1, char *i2)
{
    npy_longlong tmp = *(npy_longlong *)i1;
    *(npy_longlong *)i1 = *(npy_longlong *)i2;
    *(npy_longlong *)i2 = tmp;
}

static void ckd_nth_element(char *indices, char *data, int l, int h, int nth, int m)
{
    int ii, jj;
    char *ptr_jj = NULL, *ptr_ii = NULL, *ptr_v = NULL;
    double v, v_ii, v_jj;

    while (l <= h) {
        ii = l;
        ptr_ii = CKD_PTR_GOTO(indices, ii);

        jj = h - 1;
        ptr_jj = CKD_PTR_GOTO(indices, jj);

        ptr_v = CKD_PTR_GOTO(indices, h);
        v = CKD_GET_DOUBLE_BY_INDEX(data, ptr_v, m);

        while (ii <= jj) {
            v_ii = CKD_GET_DOUBLE_BY_INDEX(data, ptr_ii, m);
            v_jj = CKD_GET_DOUBLE_BY_INDEX(data, ptr_jj, m);

            if (v_ii > v && v_jj < v) {
                ckd_swap(ptr_ii, ptr_jj);
                double tmp = v_ii;
                v_ii = v_jj;
                v_jj = tmp;
            }
            if (v_ii <= v) {
                ii++;
                CKD_PTR_SET_NEXT(ptr_ii);
            }
            if (v_jj >= v) {
                jj--;
                CKD_PTR_SET_PREV(ptr_jj);
            }
        }

        ptr_jj = CKD_PTR_GOTO(indices, h);
        ckd_swap(ptr_ii, ptr_jj);
        if (ii == nth - 1) {
            break;
        } else if (nth - 1 < ii) {
            h = ii - 1;
        } else {
            l = ii + 1;
        }
    }
}

static int ckd_partition(char *indices, char *data, int l, int h, int m, double v)
{
    char *ptr_pp, *ptr_ii;
    ptr_pp = ptr_ii = CKD_PTR_GOTO(indices, l);
    int pp = l, ii;
    double vp;

    for (ii = l; ii < h; ii++) {
        vp = CKD_GET_DOUBLE_BY_INDEX(data, ptr_ii, m);
        if (vp < v) {
            ckd_swap(ptr_pp, ptr_ii);
            pp++;
            CKD_PTR_SET_NEXT(ptr_pp);
        }
        CKD_PTR_SET_NEXT(ptr_ii);
    }
    ckd_swap(ptr_pp, ptr_ii);
    return pp;
}

static int ckd_min_element(char *indices, char *data, int l, int h, int m)
{
    int min_ = l, ii;
    char *ptr_min, *ptr_ii, *ptr_di;

    ptr_ii = CKD_PTR_GOTO(indices, l);
    ptr_min = CKD_PTR_GOTO_BY_INDEX(data, ptr_ii, m);
    CKD_PTR_SET_NEXT(ptr_ii);

    for (ii = l + 1; ii < h; ii++) {
        ptr_di = CKD_PTR_GOTO_BY_INDEX(data, ptr_ii, m);
        if (CKD_CMP_DOUBLE_LT_PTR(ptr_di, ptr_min)) {
            min_ = ii;
            ptr_min = ptr_di;
        }
        CKD_PTR_SET_NEXT(ptr_ii);
    }
    return min_;
}

static int ckd_max_element(char *indices, char *data, int l, int h, int m)
{
    int max_ = l, ii;
    char *ptr_max, *ptr_ii, *ptr_di;

    ptr_ii = CKD_PTR_GOTO(indices, l);
    ptr_max = CKD_PTR_GOTO_BY_INDEX(data, ptr_ii, m);
    CKD_PTR_SET_NEXT(ptr_ii);

    for (ii = l + 1; ii < h; ii++) {
        ptr_di = CKD_PTR_GOTO_BY_INDEX(data, ptr_ii, m);
        if (CKD_CMP_DOUBLE_GT_PTR(ptr_di, ptr_max)) {
            max_ = ii;
            ptr_max = ptr_di;
        }
        CKD_PTR_SET_NEXT(ptr_ii);
    }
    return max_;
}

// ----------------------------------  Hyperparameter  -----------------------------------------------------------------

static void ckd_hyperparameter_init(CKDHyp *self, int pnorm, int is_inf, double dist_bound, double epsilon)
{
    self->pnorm = (double)pnorm;
    self->is_inf = is_inf;
    self->bound = dist_bound;
    if (!is_inf && pnorm > 1) {
        self->bound = pow(self->bound, self->pnorm);
    }
    self->eps = epsilon;
    if (epsilon == 0) {
        self->eps_frac = 1;
    } else if (is_inf) {
        self->eps_frac = 1 / (1 + epsilon);
    } else {
        self->eps_frac = 1 / pow((1 + epsilon), self->pnorm);
    }
}

// ----------------------------------  Distance  -----------------------------------------------------------------------

static double ckd_minkowski_distance_p1(CKDHyp *self, double p1)
{
    p1 = p1 < 0 ? -p1 : p1;
    if (self->pnorm == 1 || self->is_inf) {
        return p1;
    }
    return pow(p1, self->pnorm);
}

static double ckd_minkowski_distance_p1p2(CKDHyp *self, double *p1, double *p2, int m)
{
    double out = 0, v;
    int ii;
    for (ii = 0; ii < m; ii++) {
        v = ckd_minkowski_distance_p1(self, *(p2 + ii) - *(p1 + ii));
        if (self->is_inf && v > out) {
            out = v;
        } else {
            out += v;
        }
    }
    return out;
}

static double ckd_minkowski_distance_min_max_p1(CKDHyp *self, double p1, double l, double h)
{
    if (p1 > h) {
        return ckd_minkowski_distance_p1(self, p1 - h);
    } else if (p1 < l) {
        return ckd_minkowski_distance_p1(self, l - p1);
    }
    return 0;
}

static void ckd_interval_distance_p1(double p1, double l, double h, double *min_, double *max_)
{
    *min_ = p1 - h > l - p1 ? p1 - h : l - p1;
    if (*min_ < 0) {
        *min_ = 0;
    }
    *max_ = h - p1 > p1 - l ? h - p1 : p1 - l;
}

static void ckd_minkowski_interval_distance_p1(CKDHyp *self, double *p1, double *l, double *h, int m, double *l_dist, double *h_dist)
{
    int ii;
    double min_, max_;
    *l_dist = 0;
    *h_dist = 0;
    for (ii = 0; ii < m; ii++) {
        ckd_interval_distance_p1(*(p1 + ii), *(l + ii), *(h + ii), &min_, &max_);
        if (self->is_inf) {
            *l_dist = *l_dist > min_ ? *l_dist : min_;
            *h_dist = *h_dist > max_ ? *h_dist : max_;
        } else {
            *l_dist += ckd_minkowski_distance_p1(self, min_);
            *h_dist += ckd_minkowski_distance_p1(self, max_);
        }
    }
}

// ----------------------------------  KNN query helpers  --------------------------------------------------------------
// *************************************  Heap  ************************************************************************

static int ckdheap_init(CKDheap *self, int init_size)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->heap = malloc(init_size * sizeof(CKDheap_item));
    if (!self->heap) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void ckdheap_free(CKDheap *self)
{
    if (self->_loc_n) {
        free(self->heap);
    }
}

static void ckdheap_push_down(CKDheap_item *heap, int pos, int l)
{
    CKDheap_item new_item = *(heap + pos);
    int parent_pos;
    while (pos > l) {
        parent_pos = (pos - 1) >> 1;
        if (new_item.priority < (heap + parent_pos)->priority) {
            *(heap + pos) = *(heap + parent_pos);
            pos = parent_pos;
        } else {
            break;
        }
    }
    *(heap + pos) = new_item;
}

static void ckdheap_push_up(CKDheap_item *heap, int pos, int h)
{
    CKDheap_item new_item = *(heap + pos);
    int l = pos, child_pos = 2 * pos + 1;
    while (child_pos < h) {
        if (child_pos + 1 < h && !((heap + child_pos)->priority < (heap + child_pos + 1)->priority)) {
            child_pos += 1;
        }
        *(heap + pos) = *(heap + child_pos);
        pos = child_pos;
        child_pos = 2 * pos + 1;
    }
    *(heap + pos) = new_item;
    ckdheap_push_down(heap, pos, l);
}

static int ckdheap_push(CKDheap *self, CKDheap_item new_item)
{
    int nn = self->n;
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n = 2 * self->_loc_n + 1;
        self->heap = realloc(self->heap, self->_loc_n * sizeof(CKDheap_item));
        if (!self->heap) {
            PyErr_NoMemory();
            return 0;
        }
    }
    *(self->heap + nn) = new_item;
    ckdheap_push_down(self->heap, nn, 0);
    return 1;
}

static CKDheap_item ckdheap_pop(CKDheap *self)
{
    CKDheap_item last = *(self->heap + self->n - 1), out;
    self->n--;
    if (self->n > 0) {
        out = *(self->heap);
        *(self->heap) = last;
        ckdheap_push_up(self->heap, 0, self->n);
        return out;
    }
    return last;
}

// **********************************  Knn query stack  ****************************************************************

static int ckdknn_query_init(CKDknn_query *self, int m)
{
    self->min_distance = 0;
    self->split_distance = malloc(m * sizeof(double));
    if (!self->split_distance) {
        PyErr_NoMemory();
        return 0;
    }
    self->node = NULL;
    return 1;
}

static int ckdknn_query_init_from(CKDknn_query *self, CKDknn_query *from, int m)
{
    self->min_distance = from->min_distance;
    memcpy(self->split_distance, from->split_distance, m * sizeof(double));
    if (!self->split_distance) {
        PyErr_NoMemory();
        return 0;
    }
    self->node = NULL;
    return 1;
}

static void ckdknn_query_free(CKDknn_query *self)
{
    free(self->split_distance);
    self->node = NULL;
}

static int ckdknn_stack_init(CKDknn_stack *self, int init_size)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->stack = malloc(init_size * sizeof(CKDknn_query));
    if (!self->stack) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void ckdknn_stack_free(CKDknn_stack *self)
{
    int ii;
    if (self->_loc_n) {
        for (ii = 0; ii < self->n; ii++) {
            ckdknn_query_free(self->stack + ii);
        }
        free(self->stack);
    }
}

static int ckdknn_stack_push(CKDknn_stack *self, int m)
{
    int nn = self->n;
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n++;
        self->stack = realloc(self->stack, self->_loc_n * sizeof(CKDknn_query));
        if (!self->stack) {
            PyErr_NoMemory();
            return 0;
        }
    }
    if (!ckdknn_query_init(self->stack + nn, m)) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

// ----------------------------------  Ball query helpers  -------------------------------------------------------------
// **********************************  Ball query stack  ***************************************************************

static int ckdball_stack_init(CKDball_stack *self, int init_size, int m)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->stack = malloc(init_size * sizeof(CKDball_query));
    self->bound_min = malloc(m * 2 * sizeof(double));
    self->bound_max = self->bound_min + m;
    if (!self->stack || !self->bound_min) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void ckdball_stack_adapt_tree(CKDball_stack *self, CKDtree *tree, CKDHyp *params, double *point)
{
    char *dims_min = NULL, *dims_max = NULL;

    dims_min = (void *)PyArray_DATA(tree->dims_min);
    dims_max = (void *)PyArray_DATA(tree->dims_max);

    ckd_copy_data_point(dims_min, tree->m, self->bound_min);
    ckd_copy_data_point(dims_max, tree->m, self->bound_max);

    ckd_minkowski_interval_distance_p1(params, point,
                                       self->bound_min, self->bound_max,
                                       tree->m,
                                       &(self->min_distance), &(self->max_distance));

}

static void ckdball_stack_free(CKDball_stack *self)
{
    if (self->_loc_n) {
        free(self->stack);
    }
    free(self->bound_min);
}

static int ckdball_stack_push(CKDball_stack *self, CKDnode *node, int lesser, int m, double *point, CKDHyp *params)
{
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n++;
        self->stack = realloc(self->stack, self->_loc_n * sizeof(CKDball_query));
        if (!self->stack) {
            PyErr_NoMemory();
            return 0;
        }
    }
    CKDball_query *q = self->stack + (self->n - 1);
    q->split_dim = node->split_dim;
    q->min_distance = self->min_distance;
    q->max_distance = self->max_distance;
    q->split_min_distance = *(self->bound_min + node->split_dim);
    q->split_max_distance = *(self->bound_max + node->split_dim);

    if (lesser) {
        *(self->bound_max + node->split_dim) = node->split_val;
    } else {
        *(self->bound_min + node->split_dim) = node->split_val;
    }

    ckd_minkowski_interval_distance_p1(params, point,
                                       self->bound_min, self->bound_max,
                                       m,
                                       &(self->min_distance), &(self->max_distance));
    return 1;
}

static int ckdball_stack_pop(CKDball_stack *self)
{
    if (self->n == 0) {
        return 0;
    }
    self->n--;
    CKDball_query *q = self->stack + self->n;
    self->min_distance = q->min_distance;
    self->max_distance = q->max_distance;
    *(self->bound_min + q->split_dim) = q->split_min_distance;
    *(self->bound_max + q->split_dim) = q->split_max_distance;
    return 1;

}

// -------------------------------  query results  ---------------------------------------------------------------------

static int ckdresults_init(CKDresults *self, int inc_distance)
{
    self->indices = PyList_New(0);
    if (inc_distance) {
        self->distance = PyList_New(0);
    } else {
        self->distance = NULL;
    }
    if (!self->indices || !(!inc_distance || self->distance)) {
        PyErr_NoMemory();
        return 0;
    }
    self->_ind = NULL;
    self->_dist = NULL;
    return 1;
}

static int ckdresults_init_query(CKDresults *self)
{
    self->_ind = PyList_New(0);
    if (self->distance != NULL) {
        self->_dist = PyList_New(0);
    }
    if (!self->_ind || (self->distance != NULL && !self->_dist)) {
        PyErr_NoMemory();
        return 0;
    }
    if (PyList_Append(self->indices, self->_ind) ||
        (self->distance != NULL && PyList_Append(self->distance, self->_dist))) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static int ckdresults_push_indices(CKDresults *self, int i)
{
    if (self->_ind == NULL && !ckdresults_init_query(self)) {
        PyErr_NoMemory();
        return 0;
    }
    if (PyList_Append(self->_ind, Py_BuildValue("i", i))) {
        return 0;
    }
    return 1;
}

static int ckdresults_push_indices_distance(CKDresults *self, int i, double d)
{
    if (!ckdresults_push_indices(self, i) || self->distance == NULL) {
        return 0;
    }
    if (PyList_Append(self->_dist, Py_BuildValue("d", d))) {
        return 0;
    }
    return 1;
}

// #####################################################################################################################
// ----------------------------------  Build new tree  -----------------------------------------------------------------

static int ckd_push_node_to_list(CKDtree *self)
{
    CKDnode *node;
    node = (CKDnode *)CKDnode_Type.tp_alloc(&CKDnode_Type, 0);

    if (node == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Error: CKDnodePy_new");
        return 0;
    }
    node->start_index = 0;
    node->end_index = 0;
    node->children = 0;

    node->split_dim = -1;
    node->split_val = 0;
    node->lesser_index = -1;
    node->higher_index = -1;
    node->level = 0;

    node->lesser = NULL;
    node->higher = NULL;

    if (PyList_Append(self->tree_list, (PyObject *)node)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyList_Append");
        return 0;
    }
    self->size++;
    return 1;
}

static int ckd_build_tree(CKDtree *self, int l, int h, double *dims_min, double *dims_max, int level)
{
    char *data = NULL, *indices = NULL, *ptr_d = NULL, *ptr_i = NULL;
    int m = self->m, curr_pos = self->size, split_dim = 0, ii, jj, nth, pp, node_lesser, node_higher;
    CKDnode *node;
    double dims_delta = 0, split_val, vj;

    if (!ckd_push_node_to_list(self)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: ckd_push_node_to_list");
        return -1;
    }

    node = (CKDnode *)PyList_GetItem(self->tree_list, (Py_ssize_t)curr_pos);

    node->start_index = l;
    node->end_index = h;
    node->children = h - l;
    node->level = level;

    if (node->children <= self->leafsize) {
        node->split_dim = -1;
        return curr_pos;
    }

    ptr_d = data = (void *)PyArray_DATA(self->data);
    ptr_i = indices = (void *)PyArray_DATA(self->indices);

    CKD_PTR_SET_GOTO(ptr_i, l);
    ptr_d = CKD_PTR_GOTO_BY_INDEX(data, ptr_i, m);

    for (jj = 0; jj < m; jj++) {
        *(dims_min + jj) = *(dims_max + jj) = CKD_GET_DOUBLE(ptr_d);
        CKD_PTR_SET_NEXT(ptr_d);
    }

    CKD_PTR_SET_NEXT(ptr_i);

    for (ii = l + 1; ii < h; ii++) {
        ptr_d = CKD_PTR_GOTO_BY_INDEX(data, ptr_i, m);
        for (jj = 0; jj < m; jj++) {
            vj = CKD_GET_DOUBLE(ptr_d);

            if (*(dims_min + jj) > vj) {
                *(dims_min + jj) = vj;
            }
            if (*(dims_max + jj) < vj) {
                *(dims_max + jj) = vj;
            }
            CKD_PTR_SET_NEXT(ptr_d);
        }
        CKD_PTR_SET_NEXT(ptr_i);
    }

    for (jj = 0; jj < m; jj++) {
        if (*(dims_max + jj) - *(dims_min + jj) > dims_delta) {
            dims_delta = *(dims_max + jj) - *(dims_min + jj);
            split_dim = jj;
        }
    }

    if (*(dims_min + split_dim) == *(dims_max + split_dim)) {
        node->split_dim = -1;
        return curr_pos;
    }

    node->split_dim = split_dim;

    ptr_d = CKD_PTR_GOTO(data, split_dim);

    nth = l + (h - l) / 2;

    ckd_nth_element(indices, ptr_d, l, h - 1, nth, m);

    ptr_i = CKD_PTR_GOTO(indices, nth);
    split_val = CKD_GET_DOUBLE_BY_INDEX(ptr_d, ptr_i, m);

    pp = ckd_partition(indices, ptr_d, l, nth, m, split_val);

    if (pp == l) {
        int pp_min = ckd_min_element(indices, ptr_d, l, h, m);
        ptr_i = CKD_PTR_GOTO(indices, pp_min);
        split_val = CKD_GET_DOUBLE_BY_INDEX(ptr_d, ptr_i, m);

        split_val = nextafter(split_val, HUGE_VAL);
        pp = ckd_partition(indices, ptr_d, l, h - 1, m, split_val);
    } else if (pp == h) {
        int pp_max = ckd_max_element(indices, ptr_d, l, h, m);
        ptr_i = CKD_PTR_GOTO(indices, pp_max);
        split_val = CKD_GET_DOUBLE_BY_INDEX(ptr_d, ptr_i, m);

        pp = ckd_partition(indices, ptr_d, l, h - 1, m, split_val);
    }

    node->split_val = split_val;

    node_lesser = ckd_build_tree(self, l, pp, dims_min, dims_max, level + 1);
    node_higher = ckd_build_tree(self, pp, h, dims_min, dims_max, level + 1);

    node = (CKDnode *)PyList_GetItem(self->tree_list, (Py_ssize_t)curr_pos);

    if (!curr_pos) {
        self->tree = node;
    }

    node->lesser_index = node_lesser;
    node->higher_index = node_higher;

    node->lesser = (CKDnode *)PyList_GetItem(self->tree_list, (Py_ssize_t)node_lesser);
    node->higher = (CKDnode *)PyList_GetItem(self->tree_list, (Py_ssize_t)node_higher);
    return curr_pos;
}

// **********************************  queries  ************************************************************************

static int ckdtree_knn_query(CKDtree *self, CKDHyp *params, double *point, int k, CKDresults *output)
{
    CKDheap h1 = CKDHEAP_INIT, h2 = CKDHEAP_INIT, hs = CKDHEAP_INIT;
    CKDknn_stack qs = CKDKNN_STACK_INIT;
    CKDknn_query *q1, *q2, *_qt;
    CKDheap_item h2q1;
    CKDnode *qn;
    int m = self->m, jj, ii, nk;
    double *dp = NULL, *cap, dist, dist_bound = params->bound;
    char *dims_min = NULL, *dims_max = NULL, *data = NULL, *indices = NULL, *d_ptr = NULL, *i_ptr = NULL;

    dims_min = (void *)PyArray_DATA(self->dims_min);
    dims_max = (void *)PyArray_DATA(self->dims_max);
    data = (void *)PyArray_DATA(self->data);
    indices = (void *)PyArray_DATA(self->indices);

    cap = malloc(m * sizeof(double));
    dp = cap;

    if (!cap ||
        !ckdheap_init(&h1, k) ||
        !ckdheap_init(&h2, self->size) ||
        !ckdknn_stack_init(&qs, self->size) ||
        !ckdknn_stack_push(&qs, m)) {
        PyErr_NoMemory();
        goto exit;
    }

    q1 = qs.stack;
    q1->node = self->tree;

    for (jj = 0; jj < m; jj++) {
        dist = ckd_minkowski_distance_min_max_p1(params, *(point + jj), CKD_GET_DOUBLE(dims_min), CKD_GET_DOUBLE(dims_max));
        *(q1->split_distance + jj) = dist;
        if (params->is_inf) {
            q1->min_distance = dist > q1->min_distance ? dist : q1->min_distance;
        } else {
            q1->min_distance += dist;
        }
        CKD_PTR_SET_NEXT(dims_min);
        CKD_PTR_SET_NEXT(dims_max);
    }

    while (1) {
        qn = q1->node;
        if (!(qn->split_dim == -1)) {
            if (q1->min_distance > (dist_bound * params->eps_frac)) {
                break;
            }
            if (!ckdknn_stack_push(&qs, m)) {
                PyErr_NoMemory();
                goto exit;
            }

            q2 = qs.stack + (qs.n - 1);
            if (!ckdknn_query_init_from(q2, q1, m)) {
                PyErr_NoMemory();
                goto exit;
            }

            if (*(point + qn->split_dim) < qn->split_val) {
                q1->node = qn->lesser;
                q2->node = qn->higher;
                dist = qn->split_val - *(point + qn->split_dim);
            } else {
                q1->node = qn->higher;
                q2->node = qn->lesser;
                dist = *(point + qn->split_dim) - qn->split_val;
            }
            dist = ckd_minkowski_distance_p1(params, dist);

            if (params->is_inf) {
                q2->min_distance = q2->min_distance > dist ? q2->min_distance : dist;
            } else {
                q2->min_distance += dist - *(q2->split_distance + qn->split_dim);
            }
            *(q2->split_distance + qn->split_dim) = q2->min_distance;

            if (q1->min_distance > q2->min_distance) {
                _qt = q2;
                q2 = q1;
                q1 = _qt;
            }

            if (q2->min_distance <= (dist_bound * params->eps_frac)) {
                CKDheap_item hh2;
                hh2.priority = q2->min_distance;
                hh2.contents.data_ptr = q2;
                if (!ckdheap_push(&h2, hh2)) {
                    PyErr_SetString(PyExc_RuntimeError, "Error: ckdheap_push");
                    goto exit;
                }
            }
        } else {
            i_ptr = CKD_PTR_GOTO(indices, qn->start_index);

            for (ii = qn->start_index; ii < qn->end_index; ii++) {
                d_ptr = CKD_PTR_GOTO_BY_INDEX(data, i_ptr, m);

                ckd_copy_data_point(d_ptr, m, dp);

                dist = ckd_minkowski_distance_p1p2(params, point, dp, m);

                if (dist < dist_bound) {
                    if (h1.n == k) {
                        ckdheap_pop(&h1);
                    }
                    CKDheap_item hh1;
                    hh1.priority = -dist;
                    hh1.contents.index = CKD_GET_INT(i_ptr);
                    if (!ckdheap_push(&h1, hh1)) {
                        PyErr_SetString(PyExc_RuntimeError, "Error: ckdheap_push");
                        goto exit;
                    }
                    if (h1.n == k) {
                        dist_bound = -(h1.heap)->priority;
                    }
                }
                CKD_PTR_SET_NEXT(i_ptr);
            }
            if (!h2.n) {
                break;
            }

            h2q1 = ckdheap_pop(&h2);
            q1 = h2q1.contents.data_ptr;
        }
    }

    nk = h1.n;
    if (!ckdheap_init(&hs, nk)) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = nk - 1; ii >= 0; ii--) {
        hs.heap[ii] = ckdheap_pop(&h1);
    }

    for (ii = 0; ii < nk; ii++) {
        dist = -hs.heap[ii].priority;
        if (!params->is_inf && params->pnorm > 1) {
            dist = pow(dist, (1 / params->pnorm));
        }
        ckdresults_push_indices_distance(output, hs.heap[ii].contents.index, dist);
    }

    exit:
        ckdknn_stack_free(&qs);
        ckdheap_free(&h1);
        ckdheap_free(&h2);
        ckdheap_free(&hs);
        return PyErr_Occurred() ? 0 : 1;

}

static void ckdtree_ball_get_points(CKDtree *self, CKDnode *node, CKDresults *output)
{
    if (node->split_dim == -1) {
        char *indices = (void *)PyArray_DATA(self->indices);
        int ii;

        CKD_PTR_SET_GOTO(indices, node->start_index);

        for (ii = node->start_index; ii < node->end_index; ii++) {
            ckdresults_push_indices(output, CKD_GET_INT(indices));
            CKD_PTR_SET_NEXT(indices);
        }
    } else {
        ckdtree_ball_get_points(self, node->lesser, output);
        ckdtree_ball_get_points(self, node->higher, output);
    }
}

static int ckdtree_ball_query_traverser(CKDtree *self, CKDnode *node, CKDball_stack *stack, CKDHyp *params,
                                        double *point, double *dp, CKDresults *output)
{
    if (stack->min_distance > (params->bound * params->eps_frac))
        return 1;

    if (stack->max_distance < (params->bound / params->eps_frac)) {
        ckdtree_ball_get_points(self, node, output);
        return 1;
    }

    if (node->split_dim == -1) {

        char *indices = (void *)PyArray_DATA(self->indices), *data = (void *)PyArray_DATA(self->data), *d_ptr = NULL;
        int ii;
        double dist;

        CKD_PTR_SET_GOTO(indices, node->start_index);

        for (ii = node->start_index; ii < node->end_index; ii++) {
            d_ptr = CKD_PTR_GOTO_BY_INDEX(data, indices, self->m);
            ckd_copy_data_point(d_ptr, self->m, dp);

            dist = ckd_minkowski_distance_p1p2(params, point, dp, self->m);

            if (dist <= params->bound) {
                ckdresults_push_indices(output, CKD_GET_INT(indices));
            }
            CKD_PTR_SET_NEXT(indices);
        }
    } else {
        if (!ckdball_stack_push(stack, node, 1, self->m, point, params) ||
            !ckdtree_ball_query_traverser(self, node->lesser, stack, params, point, dp, output) ||
            !ckdball_stack_pop(stack)) {
            return 0;
        }
        if (!ckdball_stack_push(stack, node, 0, self->m, point, params) ||
            !ckdtree_ball_query_traverser(self, node->higher, stack, params, point, dp, output) ||
            !ckdball_stack_pop(stack)) {
            return 0;
        }
    }

    return 1;
}

static int ckdtree_ball_query(CKDtree *self, CKDHyp *params, double *point, CKDresults *output)
{
    CKDball_stack stack = CKDBALL_STACK_INIT;
    double *dp;

    dp = malloc(self->m * sizeof(double));

    if (!dp || !ckdball_stack_init(&stack, self->size, self->m)) {
        PyErr_NoMemory();
        goto exit;
    }

    ckdball_stack_adapt_tree(&stack, self, params, point);

    if (!ckdtree_ball_query_traverser(self, self->tree, &stack, params, point, dp, output)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: ckdtree_ball_query_traverser \n");
        goto exit;
    }

    exit:
        free(dp);
        ckdball_stack_free(&stack);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################
// ----------------------------------  communication  ------------------------------------------------------------------
// **********************************  CKDnode  ************************************************************************

void CKDnodePy_dealloc(CKDnode *self) {Py_TYPE(self)->tp_free((PyObject *)self);}

PyObject *CKDnodePy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CKDnode *self;
    self = (CKDnode *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->start_index = 0;
        self->end_index = 0;
        self->children = 0;

        self->split_dim = -1;
        self->split_val = 0;
        self->lesser_index = -1;
        self->higher_index = -1;
        self->level = 0;

        self->lesser = NULL;
        self->higher = NULL;
    }
    return (PyObject *)self;
}

static void ckdnode_children(CKDnode *self) {self->children = self->end_index - self->start_index;}

static void ckdnode_init(CKDnode *self, int start_index, int end_index)
{
    self->start_index = start_index;
    self->end_index = end_index;
    ckdnode_children(self);
}

int CKDnodePy_init(CKDnode *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"start_index", "end_index", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|ii", kwlist, &(self->start_index), &(self->end_index))) {
        return -1;
    }
    ckdnode_children(self);
    return 0;
}

// **********************************  CKDtree  ************************************************************************

void CKDtreePy_dealloc(CKDtree *self)
{
    Py_XDECREF(self->data);
    Py_XDECREF(self->dims_min);
    Py_XDECREF(self->dims_max);
    Py_XDECREF(self->indices);
    Py_XDECREF(self->tree);
    Py_TYPE(self->tree_list)->tp_free((PyObject *)(self->tree_list));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *CKDtreePy_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    CKDtree *self;
    self = (CKDtree *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->m = 0;
        self->n = 0;
        self->leafsize = 8;
        self->size = 0;

        self->data = NULL;
        self->dims_min = NULL;
        self->dims_max = NULL;
        self->indices = NULL;
        self->tree_list = NULL;

        self->tree = NULL;
    }

    return (PyObject *)self;
}

static int ckdtree_input_data(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    if (*output && PyArray_TYPE(*output) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be float64 \n");
       *output = NULL;
    }
    return *output != NULL;
}

static int ckdtree_input_array(PyObject *object, PyArrayObject **output)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *output = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *output != NULL;
}

static int ckdtree_init(CKDtree *self)
{
    npy_intp dims[1] = {0};
    char *_dims_min = NULL, *_dims_max = NULL, *_indices = NULL, *_data = NULL;
    int ii, jj;
    double *build_dims, *b_dims_min, *b_dims_max;

    *dims = self->m;
    self->dims_min = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
    self->dims_max = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);

    *dims = self->n;
    self->indices = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);

    if (!self->dims_min || !self->dims_max || !self->indices) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_EMPTY");
        return 0;
    }

    _data = (void *)PyArray_DATA(self->data);
    _dims_min = (void *)PyArray_DATA(self->dims_min);
    _dims_max = (void *)PyArray_DATA(self->dims_max);
    _indices = (void *)PyArray_DATA(self->indices);

    for (jj = 0; jj < self->m; jj++) {
        CKD_SET_DOUBLE_PTR((CKD_PTR_GOTO(_dims_min, jj)), (CKD_PTR_GOTO(_data, jj)));
        CKD_SET_DOUBLE_PTR((CKD_PTR_GOTO(_dims_max, jj)), (CKD_PTR_GOTO(_data, jj)));
    }

    for (ii = 0; ii < self->n; ii++) {
        CKD_SET_INT(_indices, ii);
        CKD_PTR_SET_NEXT(_indices);

        for (jj = 0; jj < self->m; jj++) {
            if (CKD_CMP_DOUBLE_LT_PTR(_data, (CKD_PTR_GOTO(_dims_min, jj)))) {
                CKD_SET_DOUBLE_PTR((CKD_PTR_GOTO(_dims_min, jj)), _data);
            }
            if (CKD_CMP_DOUBLE_GT_PTR(_data, (CKD_PTR_GOTO(_dims_max, jj)))) {
                CKD_SET_DOUBLE_PTR((CKD_PTR_GOTO(_dims_max, jj)), _data);
            }
            CKD_PTR_SET_NEXT(_data);
        }
    }

    build_dims = calloc(self->m * 2, sizeof(double));
    if (!build_dims) {
        PyErr_NoMemory();
        return 0;
    }

    b_dims_min = build_dims;
    b_dims_max = build_dims + self->m;

    if (ckd_build_tree(self, 0, self->n, b_dims_min, b_dims_max, 0) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: ckd_build_tree");
        return 0;
    }

    free(build_dims);
    return 1;
}

int CKDtreePy_init(CKDtree *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"", "leafsize", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O&|i", kwlist, ckdtree_input_data, &(self->data), &(self->leafsize))) {
        return -1;
    }

    self->n = (int)PyArray_DIM(self->data, 0);
    self->m = (int)PyArray_DIM(self->data, 1);
    self->size = 0;

    self->tree_list = PyList_New(0);

    if (!ckdtree_init(self)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: ckdtree_init");
        return -1;
    }
    return 0;
}

// **********************************  CKDtree methods  ****************************************************************

PyObject *CKDtree_knn_query(CKDtree *self, PyObject *args)
{
    int pnorm, is_inf;
    double distance_max, epsilon;
    PyArrayObject *query_points, *k;

    CKDresults results = CKDRESULTS_INIT;
    CKDHyp params;
    int np, ii, jj, ki;
    PYCV_ArrayIterator iter_k, iter_p;
    int num_type_k, num_type_p;
    char *k_ptr = NULL, *p_ptr = NULL;
    PyObject *output;
    double *point, pj;

    if (!PyArg_ParseTuple(args, "O&O&iidd",
                          ckdtree_input_array, &query_points,
                          ckdtree_input_array, &k,
                          &pnorm, &is_inf, &distance_max, &epsilon)) {
        goto exit;
    }

    if (PyArray_NDIM(query_points) != 2 || PyArray_DIM(query_points, 1) != self->m) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid points shape \n");
        goto exit;
    }

    point = malloc(self->m * sizeof(double));

    if (!point || !ckdresults_init(&results, 1)) {
        PyErr_NoMemory();
        goto exit;
    }

    ckd_hyperparameter_init(&params, pnorm, is_inf, distance_max, epsilon);

    np = (int)PyArray_DIM(query_points, 0);

    PYCV_ArrayIteratorInit(k, &iter_k);
    PYCV_ArrayIteratorInit(query_points, &iter_p);

    num_type_k = PyArray_TYPE(k);
    num_type_p = PyArray_TYPE(query_points);

    k_ptr = (void *)PyArray_DATA(k);
    p_ptr = (void *)PyArray_DATA(query_points);

    for (ii = 0; ii < np; ii++) {
        PYCV_GET_VALUE(num_type_k, int, k_ptr, ki);

        for (jj = 0; jj < self->m; jj++) {
            PYCV_GET_VALUE(num_type_p, double, p_ptr, pj);
            *(point + jj) = pj;
            PYCV_ARRAY_ITERATOR_NEXT(iter_p, p_ptr);
        }

        if (!ckdresults_init_query(&results)) {
            PyErr_NoMemory();
            goto exit;
        }

        if (!ckdtree_knn_query(self, &params, point, ki, &results)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: ckdtree_knn_query \n");
            goto exit;
        }

        PYCV_ARRAY_ITERATOR_NEXT(iter_k, k_ptr);
    }

    output = Py_BuildValue("(O,O)", results.distance, results.indices);

    exit:
        free(point);
        return PyErr_Occurred() ? Py_BuildValue("") : output;
}

PyObject *CKDtree_ball_point_query(CKDtree *self, PyObject *args)
{
    int pnorm, is_inf;
    double epsilon;
    PyArrayObject *query_points, *r;

    CKDresults results = CKDRESULTS_INIT;
    CKDHyp params;
    int np, ii, jj;
    PYCV_ArrayIterator iter_r, iter_p;
    int num_type_r, num_type_p;
    char *r_ptr = NULL, *p_ptr = NULL;
    double *point, pj, ri;

    if (!PyArg_ParseTuple(args, "O&O&iid",
                          ckdtree_input_array, &query_points,
                          ckdtree_input_array, &r,
                          &pnorm, &is_inf, &epsilon)) {
        goto exit;
    }

    if (PyArray_NDIM(query_points) != 2 || PyArray_DIM(query_points, 1) != self->m) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid points shape \n");
        goto exit;
    }

    point = malloc(self->m * sizeof(double));

    if (!point || !ckdresults_init(&results, 0)) {
        PyErr_NoMemory();
        goto exit;
    }

    np = (int)PyArray_DIM(query_points, 0);

    PYCV_ArrayIteratorInit(r, &iter_r);
    PYCV_ArrayIteratorInit(query_points, &iter_p);

    num_type_r = PyArray_TYPE(r);
    num_type_p = PyArray_TYPE(query_points);

    r_ptr = (void *)PyArray_DATA(r);
    p_ptr = (void *)PyArray_DATA(query_points);

    for (ii = 0; ii < np; ii++) {
        PYCV_GET_VALUE(num_type_r, double, r_ptr, ri);

        for (jj = 0; jj < self->m; jj++) {
            PYCV_GET_VALUE(num_type_p, double, p_ptr, pj);
            *(point + jj) = pj;
            PYCV_ARRAY_ITERATOR_NEXT(iter_p, p_ptr);
        }

        if (!ckdresults_init_query(&results)) {
            PyErr_NoMemory();
            goto exit;
        }

        ckd_hyperparameter_init(&params, pnorm, is_inf, ri, epsilon);

        if (!ckdtree_ball_query(self, &params, point, &results)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: ckdtree_ball_query \n");
            goto exit;
        }

        PYCV_ARRAY_ITERATOR_NEXT(iter_r, r_ptr);
    }

    exit:
        free(point);
        return PyErr_Occurred() ? Py_BuildValue("") : results.indices;
}

// #####################################################################################################################
