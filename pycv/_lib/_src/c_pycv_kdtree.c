#include "c_pycv_base.h"
#include "c_pycv_kdtree.h"

// #####################################################################################################################

// *************************************  KDarray  *********************************************************************

void kdarray_init(KDarray *self, PyArrayObject *array)
{
    self->object = array;
    self->base = self->ptr = (void *)PyArray_DATA(array);
    self->numtype = PyArray_TYPE(array);
    self->itemsize = PyArray_ITEMSIZE(array);
}

static int kdarray_valid_dtype(int numtype, int expect_int)
{
    if (expect_int) {
        switch (numtype) {
            case NPY_INT:
            case NPY_LONG:
            case NPY_LONGLONG:
                break;
            default:
                PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be int \n");
                return 0;
        }
    } else {
        switch (numtype) {
            case NPY_DOUBLE:
                break;
            default:
                PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be float64 \n");
                return 0;
        }
    }
    return 1;
}

static kd_intp kdarray_iget(KDarray *self, char *ptr)
{
    char *ptr_ = ptr == NULL ? self->ptr : ptr;
    kd_intp out = 0;
    switch (self->numtype) {
        case NPY_SHORT:
            out = (kd_intp)(*((npy_short *)ptr_));
            break;
        case NPY_INT:
            out = (kd_intp)(*((npy_int *)ptr_));
            break;
        case NPY_LONG:
            out = (kd_intp)(*((npy_long *)ptr_));
            break;
        case NPY_LONGLONG:
            out = (kd_intp)(*((npy_longlong *)ptr_));
            break;
    }
    return out;
}

static kd_double kdarray_dget(KDarray *self, char *ptr)
{
    char *ptr_ = ptr == NULL ? self->ptr : ptr;
    kd_double out = 0;
    switch (self->numtype) {
        case NPY_DOUBLE:
            out = (kd_double)(*((npy_double *)ptr));
            break;
    }
    return out;
}

static kd_intp kdarray_index_iget(KDarray *self, kd_intp index)
{
    return kdarray_iget(self, self->ptr + index * self->itemsize);
}

static kd_double kdarray_index_dget(KDarray *self, kd_intp index)
{
    return kdarray_dget(self, self->ptr + index * self->itemsize);
}

static void kdarray_allocate_point(KDarray *self, char *ptr, kd_double *point, kd_intp m)
{
    kd_intp ii;
    char *ptr_ = ptr;
    for (ii = 0; ii < m; ii++) {
        *(point + ii) = kdarray_dget(self, ptr_);
        ptr_ += self->itemsize;
    }
}

// *************************************  KDtree  **********************************************************************

static int KDtree_list_push(KDtree *self)
{
    if (!self->size) {
        self->tree_list = malloc(1 * sizeof(KDnode));
    } else {
        self->tree_list = realloc(self->tree_list, (self->size + 1) * sizeof(KDnode));
    }
    if (!self->tree_list) {
        PyErr_NoMemory();
        return 0;
    }
    self->size++;
    return 1;
}

// #####################################################################################################################

// ----------------------------------  Build tree helpers  -------------------------------------------------------------

static void kd_swap(char *i1, char *i2)
{
    char tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static void kd_nth_element(KDarray *indices, KDarray *data, kd_intp l, kd_intp h, kd_intp nth, kd_intp m)
{
    char *ptr_pp, *ptr_ii;
    ptr_pp = ptr_ii = indices->ptr + l * indices->itemsize;
    kd_intp pp = l, ii;
    kd_double v = kdarray_index_dget(data, kdarray_index_iget(indices, h) * m);

    for (ii = l; ii < h; ii++) {
        if (kdarray_index_dget(data, kdarray_iget(indices, ptr_ii) * m) < v) {
            kd_swap(ptr_pp, ptr_ii);
            pp++;
            ptr_pp += indices->itemsize;
        }
        ptr_ii += indices->itemsize;
    }
    kd_swap(ptr_pp, ptr_ii);

    if (nth < pp) {
        kd_nth_element(indices, data, l, pp - 1, nth, m);
    } else if (nth > pp) {
        kd_nth_element(indices, data, pp + 1, h, nth, m);
    }
}

static kd_intp kd_partition(KDarray *indices, KDarray *data, kd_intp l, kd_intp h, kd_intp m, kd_double v)
{
    char *ptr_pp, *ptr_ii;
    ptr_pp = ptr_ii = indices->ptr + l * indices->itemsize;
    kd_intp pp = l, ii;

    for (ii = l; ii < h; ii++) {
        if (kdarray_index_dget(data, kdarray_iget(indices, ptr_ii) * m) < v) {
            kd_swap(ptr_pp, ptr_ii);
            pp++;
            ptr_pp += indices->itemsize;
        }
        ptr_ii += indices->itemsize;
    }
    kd_swap(ptr_pp, ptr_ii);
    return pp;
}

static kd_intp kd_min_element(KDarray *indices, KDarray *data, kd_intp l, kd_intp h, kd_intp m)
{
    char *ptr_ii = indices->ptr + (l + 1) * indices->itemsize;
    kd_double v_min = kdarray_index_dget(data, kdarray_index_iget(indices, l) * m), vi;
    kd_intp min_ = l, ii;

    for (ii = l + 1; ii < h; ii++) {
        vi = kdarray_index_dget(data, kdarray_iget(indices, ptr_ii) * m);
        if (vi < v_min) {
            min_ = ii;
            v_min = vi;
        }
        ptr_ii += indices->itemsize;
    }
    return min_;
}

static kd_intp kd_max_element(KDarray *indices, KDarray *data, kd_intp l, kd_intp h, kd_intp m)
{
    char *ptr_ii = indices->ptr + (l + 1) * indices->itemsize;
    kd_double v_max = kdarray_index_dget(data, kdarray_index_iget(indices, l) * m), vi;
    kd_intp max_ = l, ii;

    for (ii = l + 1; ii < h; ii++) {
        vi = kdarray_index_dget(data, kdarray_iget(indices, ptr_ii) * m);
        if (vi > v_max) {
            max_ = ii;
            v_max = vi;
        }
        ptr_ii += indices->itemsize;
    }
    return max_;
}

// ----------------------------------  Distance  -----------------------------------------------------------------------

static kd_double kd_minkowski_distance_p1(kd_double p1, kd_intp pnorm, int is_inf)
{
    p1 = p1 < 0 ? -p1 : p1;
    if (pnorm == 1 || is_inf) {
        return p1;
    }
    return (kd_double)pow((double)p1, (double)pnorm);
}

static kd_double kd_minkowski_distance_p1p2(kd_double *p1, kd_double *p2, kd_intp m, kd_intp pnorm, int is_inf)
{
    kd_double out = 0, v;
    kd_intp ii;
    for (ii = 0; ii < m; ii++) {
        v = *(p2 + ii) - *(p1 + ii);
        v = kd_minkowski_distance_p1(v, pnorm, is_inf);
        if (is_inf && v > out) {
            out = v;
        } else {
            out += v;
        }
    }
    return out;
}

static kd_double kd_distance_p_min_max(kd_double p1, kd_double l, kd_double h, kd_intp pnorm, int is_inf)
{
    if (p1 > h) {
        return kd_minkowski_distance_p1(p1 - h, pnorm, is_inf);
    } else if (p1 < l) {
        return kd_minkowski_distance_p1(l - p1, pnorm, is_inf);
    }
    return 0;
}

static void kd_distance_p_interval_1d(kd_double p1, kd_double l, kd_double h, kd_double *min_, kd_double *max_)
{
    *min_ = p1 - h > l - p1 ? p1 - h : l - p1;
    if (*min_ < 0) {
        *min_ = 0;
    }
    *max_ = h - p1 > p1 - l ? h - p1 : p1 - l;
}

static void kd_distance_p_interval(kd_double *p1, kd_double *l, kd_double *h, kd_intp m, kd_intp pnorm, int is_inf,
                                   kd_double *l_dist, kd_double *h_dist)
{
    kd_intp ii;
    kd_double min_, max_;
    *l_dist = 0;
    *h_dist = 0;
    for (ii = 0; ii < m; ii++) {
        kd_distance_p_interval_1d(*(p1 + ii), *(l + ii), *(h + ii), &min_, &max_);
        if (is_inf) {
            *l_dist = *l_dist > min_ ? *l_dist : min_;
            *h_dist = *h_dist > max_ ? *h_dist : max_;
        } else {
            *l_dist += kd_minkowski_distance_p1(min_, pnorm, is_inf);
            *h_dist += kd_minkowski_distance_p1(max_, pnorm, is_inf);
        }
    }
}

static void kd_hyperparameter_init(kd_intp pnorm, int is_inf, kd_double *dist_bound, kd_double *epsilon)
{
    kd_double e = *epsilon;
    if (!is_inf && pnorm > 1) {
        *dist_bound = (kd_double)pow((double)(*dist_bound), (double)pnorm);
    }
    if (*epsilon == 0) {
        *epsilon = 1;
    } else if (is_inf) {
        *epsilon = 1 / (1 + *epsilon);
    } else {
        *epsilon = 1 / (kd_double)pow((double)(1 + *epsilon), (double)pnorm);
    }
}

// ----------------------------------  KNN query helpers  --------------------------------------------------------------
// *************************************  Heap  ************************************************************************

static int kdheap_init(KDheap *self, kd_intp init_size)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->heap = malloc(init_size * sizeof(KDheap_item));
    if (!self->heap) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void kdheap_free(KDheap *self)
{
    if (self->_loc_n) {
        free(self->heap);
    }
}

static void kdheap_push_down(KDheap_item *heap, kd_intp pos, kd_intp l)
{
    KDheap_item new_item = *(heap + pos);
    kd_intp parent_pos;
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

static void kdheap_push_up(KDheap_item *heap, kd_intp pos, kd_intp h)
{
    KDheap_item new_item = *(heap + pos);
    kd_intp l = pos, child_pos = 2 * pos + 1;
    while (child_pos < h) {
        if (child_pos + 1 < h && !((heap + child_pos)->priority < (heap + child_pos + 1)->priority)) {
            child_pos += 1;
        }
        *(heap + pos) = *(heap + child_pos);
        pos = child_pos;
        child_pos = 2 * pos + 1;
    }
    *(heap + pos) = new_item;
    kdheap_push_down(heap, pos, l);
}

static int kdheap_push(KDheap *self, KDheap_item new_item)
{
    kd_intp nn = self->n;
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n = 2 * self->_loc_n + 1;
        self->heap = realloc(self->heap, self->_loc_n * sizeof(KDheap_item));
        if (!self->heap) {
            PyErr_NoMemory();
            return 0;
        }
    }
    *(self->heap + nn) = new_item;
    kdheap_push_down(self->heap, nn, 0);
    return 1;
}

static KDheap_item kdheap_pop(KDheap *self)
{
    KDheap_item last = *(self->heap + self->n - 1), out;
    self->n--;
    if (self->n > 0) {
        out = *(self->heap);
        *(self->heap) = last;
        kdheap_push_up(self->heap, 0, self->n);
        return out;
    }
    return last;
}

// **********************************  Knn query stack  ****************************************************************

static int kdknn_query_init(KDknn_query *self, kd_intp m)
{
    self->min_distance = 0;
    self->split_distance = malloc(m * sizeof(kd_double));
    if (!self->split_distance) {
        PyErr_NoMemory();
        return 0;
    }
    self->node = NULL;
    return 1;
}

static int kdknn_query_init_from(KDknn_query *self, KDknn_query *from, kd_intp m)
{
    self->min_distance = from->min_distance;
    memcpy(self->split_distance, from->split_distance, m * sizeof(kd_double));
    if (!self->split_distance) {
        PyErr_NoMemory();
        return 0;
    }
    self->node = NULL;
    return 1;
}

static void kdknn_query_free(KDknn_query *self)
{
    free(self->split_distance);
    self->node = NULL;
}

static int kdknn_stack_init(KDknn_stack *self, kd_intp init_size)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->stack = malloc(init_size * sizeof(KDknn_query));
    if (!self->stack) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void kdknn_stack_free(KDknn_stack *self)
{
    kd_intp ii;
    if (self->_loc_n) {
        for (ii = 0; ii < self->n; ii++) {
            kdknn_query_free(self->stack + ii);
        }
        free(self->stack);
    }
}

static int kdknn_stack_push(KDknn_stack *self, kd_intp m)
{
    kd_intp nn = self->n;
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n++;
        self->stack = realloc(self->stack, self->_loc_n * sizeof(KDknn_query));
        if (!self->stack) {
            PyErr_NoMemory();
            return 0;
        }
    }
    if (!kdknn_query_init(self->stack + nn, m)) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

// ----------------------------------  Ball query helpers  -------------------------------------------------------------
// **********************************  Ball query stack  ***************************************************************

static int kdball_stack_init(KDball_stack *self, kd_intp init_size, kd_intp m)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->stack = malloc(init_size * sizeof(KDball_query));
    self->bound_min = malloc(m * 2 * sizeof(kd_double));
    self->bound_max = self->bound_min + m;
    if (!self->stack || !self->bound_min) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void kdball_stack_free(KDball_stack *self)
{
    if (self->_loc_n) {
        free(self->stack);
    }
    free(self->bound_min);
}

static int kdball_stack_push(KDball_stack *self, KDnode *node, int lesser,
                             kd_intp m, kd_double *point, kd_intp pnorm, int is_inf)
{
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n++;
        self->stack = realloc(self->stack, self->_loc_n * sizeof(KDball_query));
        if (!self->stack) {
            PyErr_NoMemory();
            return 0;
        }
    }
    KDball_query *q = self->stack + (self->n - 1);
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

    kd_distance_p_interval(point, self->bound_min, self->bound_max, m, pnorm, is_inf, &(self->min_distance), &(self->max_distance));
    return 1;
}

static int kdball_stack_pop(KDball_stack *self)
{
    if (self->n == 0) {
        return 0;
    }
    self->n--;
    KDball_query *q = self->stack + self->n;
    self->min_distance = q->min_distance;
    self->max_distance = q->max_distance;
    *(self->bound_min + q->split_dim) = q->split_min_distance;
    *(self->bound_max + q->split_dim) = q->split_max_distance;
    return 1;

}

// ***************************************  query results  *************************************************************

static int kdresults_init(KDresults *self, kd_intp tree_size, int inc_distance)
{
    self->n = 0;
    self->_tree_size = tree_size;
    self->indices = malloc(tree_size * sizeof(kd_intp));
    if (inc_distance) {
        self->distance = malloc(tree_size * sizeof(kd_double));
    } else {
        self->distance = NULL;
    }
    if (!self->indices || !(!inc_distance || self->distance)) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static int kdresults_push1(KDresults *self, kd_intp i)
{
    self->n++;
    if (self->n % self->_tree_size == 0) {
        int has_dist = 0;
        self->_tree_size = self->_tree_size + self->n;
        self->indices = realloc(self->indices, self->_tree_size * sizeof(kd_intp));
        if (self->distance) {
            self->distance = realloc(self->distance, self->_tree_size * sizeof(kd_double));
            has_dist = 1;
        }
        if (!self->indices || !(!has_dist || self->distance)) {
            PyErr_NoMemory();
            return 0;
        }
    }
    *(self->indices + self->n - 1) = i;
    return 1;
}

static int kdresults_push12(KDresults *self, kd_intp i, kd_double d)
{
    if (!kdresults_push1(self, i) || !self->distance) {
        return 0;
    }
    *(self->distance + self->n - 1) = d;
    return 1;
}

static void kdresults_free(KDresults *self)
{
    if (self->_tree_size) {
        free(self->indices);
    }
    if (self->distance) {
        free(self->distance);
    }
}

// #####################################################################################################################

// **********************************  Build new tree  *****************************************************************

static void kd_get_min_max_dims(KDtree *self, kd_intp l, kd_intp h, kd_double *dims_min, kd_double *dims_max)
{
    KDarray *data = &self->data, *indices = &self->indices;
    kd_intp m = self->m, ii, jj;
    char *ind_ptr = indices->ptr + l * indices->itemsize;
    char *data_ptr = data->ptr + kdarray_iget(indices, ind_ptr) * m * data->itemsize;
    kd_double vj;

    for (jj = 0; jj < m; jj++) {
        *(dims_min + jj) = *(dims_max + jj) = kdarray_dget(data, data_ptr);
        data_ptr += data->itemsize;
    }
    ind_ptr += indices->itemsize;
    for (ii = l + 1; ii < h; ii++) {
        data_ptr = data->ptr + kdarray_iget(indices, ind_ptr) * m * data->itemsize;
        for (jj = 0; jj < m; jj++) {
            vj = kdarray_dget(data, data_ptr);
            if (*(dims_min + jj) > vj) {
                *(dims_min + jj) = vj;
            }
            if (*(dims_max + jj) < vj) {
                *(dims_max + jj) = vj;
            }
            data_ptr += data->itemsize;
        }
        ind_ptr += indices->itemsize;
    }
}

static kd_intp kdtree_build(KDtree *self, kd_intp l, kd_intp h, kd_double *dims_min, kd_double *dims_max, kd_intp level)
{
    KDarray *data = &self->data, *indices = &self->indices;
    kd_intp m = self->m, curr_pos = self->size, split_dim, ii, nth, pp, node_lesser, node_higher;
    KDnode *root, *node;
    kd_double dims_delta = 0, split_val;

    if (!KDtree_list_push(self)) {
        PyErr_NoMemory();
        return -1;
    }

    root = self->tree_list;
    node = root + curr_pos;

    node->start_index = l;
    node->end_index = h;
    node->children = h - l;
    node->level = level;

    if (node->children <= self->leafsize) {
        node->split_dim = -1;
        return curr_pos;
    }

    kd_get_min_max_dims(self, l, h, dims_min, dims_max);

    for (ii = 0; ii < m; ii++) {
        if (*(dims_max + ii) - *(dims_min + ii) > dims_delta) {
            dims_delta = *(dims_max + ii) - *(dims_min + ii);
            split_dim = ii;
        }
    }

    if (*(dims_min + split_dim) == *(dims_max + split_dim)) {
        node->split_dim = -1;
        return curr_pos;
    }

    node->split_dim = split_dim;
    data->ptr = data->base + split_dim * data->itemsize;

    nth = l + (h - l) / 2;
    kd_nth_element(indices, data, l, h - 1, nth, m);

    split_val = kdarray_index_dget(data, kdarray_index_iget(indices, nth) * m);
    pp = kd_partition(indices, data, l, nth, m, split_val);

    if (pp == l) {
        kd_intp pp_min = kd_min_element(indices, data, l, h, m);
        split_val = kdarray_index_dget(data, kdarray_index_iget(indices, pp_min) * m);
        split_val = (kd_double)nextafter((double)split_val, HUGE_VAL);
        pp = kd_partition(indices, data, l, h - 1, m, split_val);
    } else if (pp == h) {
        kd_intp pp_max = kd_min_element(indices, data, l, h, m);
        split_val = kdarray_index_dget(data, kdarray_index_iget(indices, pp_max) * m);
        pp = kd_partition(indices, data, l, h - 1, m, split_val);
    }
    data->ptr = data->base;
    node->split_val = split_val;

    node_lesser = kdtree_build(self, l, pp, dims_min, dims_max, level + 1);
    node_higher = kdtree_build(self, pp, h, dims_min, dims_max, level + 1);

    root = self->tree_list;
    node = root + curr_pos;

    if (!curr_pos) {
        self->tree = root;
    }

    node->lesser_index = node_lesser;
    node->higher_index = node_higher;
    node->lesser = root + node_lesser;
    node->higher = root + node_higher;
    return curr_pos;
}

// **********************************  queries  ************************************************************************

static int kdtree_knn_query(KDtree *self, kd_double *point, kd_intp k,
                            kd_intp pnorm, int is_inf, kd_double distance_max, kd_double epsilon, KDresults *output)
{
    KDheap h1 = KDHEAP_INIT, h2 = KDHEAP_INIT, hs = KDHEAP_INIT;
    KDknn_stack qs = KDKNN_STACK_INIT;
    KDknn_query *q1, *q2, *_qt;
    KDheap_item h2q1;
    KDnode *qn;
    char *d_ptr = NULL, *i_ptr = NULL;
    KDarray *data = &(self->data), *indices = &(self->indices);
    kd_intp m = self->m, jj, ii, index, nk;
    kd_double b = distance_max, ef = epsilon, dist;

    kd_double *dims_min = NULL, *dims_max = NULL, *dp = NULL, *cap;


    if (!kdheap_init(&h1, k) || !kdheap_init(&h2, self->size) ||
        !kdknn_stack_init(&qs, self->size) || !kdknn_stack_push(&qs, m)) {
        PyErr_NoMemory();
        goto exit;
    }

    cap = malloc(m * 2 * sizeof(kd_double));
    dp = dims_min = cap;
    dims_max = cap + m;

    kd_hyperparameter_init(pnorm, is_inf, &b, &ef);

    q1 = qs.stack;
    q1->node = self->tree;

    kdarray_allocate_point(&self->dims_min, self->dims_min.ptr, dims_min, m);
    kdarray_allocate_point(&self->dims_max, self->dims_max.ptr, dims_max, m);

    for (jj = 0; jj < m; jj++) {
        dist = kd_distance_p_min_max(*(point + jj), *(dims_min + jj), *(dims_max + jj), pnorm, is_inf);
        *(q1->split_distance + jj) = dist;
        if (is_inf) {
            q1->min_distance = dist > q1->min_distance ? dist : q1->min_distance;
        } else {
            q1->min_distance += dist;
        }
    }

    while (1) {
        qn = q1->node;
        if (!(qn->split_dim == -1)) {
            if (q1->min_distance > b * ef) {
                break;
            }
            if (!kdknn_stack_push(&qs, m)) {
                PyErr_NoMemory();
                goto exit;
            }

            q2 = qs.stack + (qs.n - 1);
            if (!kdknn_query_init_from(q2, q1, m)) {
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
            dist = kd_minkowski_distance_p1(dist, pnorm, is_inf);

            if (is_inf) {
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

            if (q2->min_distance <= (b * ef)) {
                KDheap_item hh2;
                hh2.priority = q2->min_distance;
                hh2.contents.data_ptr = q2;
                if (!kdheap_push(&h2, hh2)) {
                    PyErr_SetString(PyExc_RuntimeError, "Error: kdheap_push");
                    goto exit;
                }
            }

        } else {
            i_ptr = indices->ptr + qn->start_index * indices->itemsize;
            for (ii = qn->start_index; ii < qn->end_index; ii++) {
                index = kdarray_iget(indices, i_ptr);
                i_ptr += indices->itemsize;

                d_ptr = data->ptr + index * m * data->itemsize;
                kdarray_allocate_point(data, d_ptr, dp, m);

                dist = kd_minkowski_distance_p1p2(point, dp, m, pnorm, is_inf);

                if (dist < b) {
                    if (h1.n == k) {
                        kdheap_pop(&h1);
                    }
                    KDheap_item hh1;
                    hh1.priority = -dist;
                    hh1.contents.index = index;
                    if (!kdheap_push(&h1, hh1)) {
                        PyErr_SetString(PyExc_RuntimeError, "Error: kdheap_push");
                        goto exit;
                    }
                    if (h1.n == k) {
                        b = -(h1.heap)->priority;
                    }
                }
            }

            if (!h2.n) {
                break;
            }

            h2q1 = kdheap_pop(&h2);
            q1 = h2q1.contents.data_ptr;
        }
    }

    nk = h1.n;

    if (!kdheap_init(&hs, nk)) {
        PyErr_NoMemory();
        goto exit;
    }

    for (ii = nk - 1; ii >= 0; ii--) {
        hs.heap[ii] = kdheap_pop(&h1);
    }

    for (ii = 0; ii < nk; ii++) {
        dist = -hs.heap[ii].priority;
        if (!is_inf && pnorm > 1) {
            dist = (kd_double)pow((double)dist, (1 / (double)pnorm));
        }
        kdresults_push12(output, hs.heap[ii].contents.index, dist);
    }

    exit:
        kdknn_stack_free(&qs);
        kdheap_free(&h1);
        kdheap_free(&h2);
        kdheap_free(&hs);
        return PyErr_Occurred() ? 0 : 1;
}

static kdtree_ball_get_points(KDtree *self, KDnode *node, KDresults *output)
{
    if (node->split_dim == -1) {
        KDarray *indices = &self->indices;
        char *i_ptr = indices->ptr + node->start_index * indices->itemsize;
        kd_intp ii;
        for (ii = node->start_index; ii < node->end_index; ii++) {
            kdresults_push1(output, kdarray_iget(indices, i_ptr));
            i_ptr += indices->itemsize;
        }
    } else {
        kdtree_ball_get_points(self, node->lesser, output);
        kdtree_ball_get_points(self, node->higher, output);
    }
}

static int kdtree_ball_query_traverser(KDtree *self, KDball_stack *stack, KDnode *node, kd_double r,
                                       kd_intp pnorm, int is_inf, kd_double epsilon,
                                       kd_double *point, kd_double *dp, KDresults *output)
{
    if (stack->min_distance > (r * epsilon)) {
        return 1;
    }
    if (stack->max_distance < (r / epsilon)) {
        kdtree_ball_get_points(self, node, output);
        return 1;
    }

    if (node->split_dim == -1) {
        KDarray *indices = &(self->indices), *data = &(self->data);
        char *i_ptr = indices->ptr + node->start_index * indices->itemsize, *d_ptr = NULL;
        kd_intp ii, index;
        kd_double dist;

        for (ii = node->start_index; ii < node->end_index; ii++) {
            index = kdarray_iget(indices, i_ptr);
            i_ptr += indices->itemsize;

            d_ptr = data->ptr + index * self->m * data->itemsize;
            kdarray_allocate_point(data, d_ptr, dp, self->m);

            dist = kd_minkowski_distance_p1p2(point, dp, self->m, pnorm, is_inf);

            if (dist <= r) {
                kdresults_push1(output, index);
            }
        }
    } else {
        if (!kdball_stack_push(stack, node, 1, self->m, point, pnorm, is_inf) ||
            !kdtree_ball_query_traverser(self, stack, node->lesser, r, pnorm, is_inf, epsilon, point, dp, output) ||
            !kdball_stack_pop(stack)) {
            return 0;
        }
        if (!kdball_stack_push(stack, node, 0, self->m, point, pnorm, is_inf) ||
            !kdtree_ball_query_traverser(self, stack, node->higher, r, pnorm, is_inf, epsilon, point, dp, output) ||
            !kdball_stack_pop(stack)) {
            return 0;
        }
    }
    return 1;
}

static int kdtree_ball_query(KDtree *self, kd_double *point, kd_double r,
                             kd_intp pnorm, int is_inf, kd_double epsilon, KDresults *output)
{
    KDball_stack stack = KDBALL_STACK_INIT;
    kd_intp m = self->m;
    kd_double b = r, ef = epsilon, *dp;

    dp = malloc(m * sizeof(kd_double));
    if (!dp || !kdball_stack_init(&stack, self->size, m)) {
        PyErr_NoMemory();
        goto exit;
    }

    kdarray_allocate_point(&self->dims_min, self->dims_min.ptr, stack.bound_min, m);
    kdarray_allocate_point(&self->dims_max, self->dims_max.ptr, stack.bound_max, m);

    kd_distance_p_interval(point, stack.bound_min, stack.bound_max, m, pnorm, is_inf, &stack.min_distance, &stack.max_distance);

    kd_hyperparameter_init(pnorm, is_inf, &b, &ef);

    if (!kdtree_ball_query_traverser(self, &stack, self->tree, b, pnorm, is_inf, ef, point, dp, output)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: kdtree_ball_query_traverser \n");
        goto exit;
    }

    exit:
        free(dp);
        kdball_stack_free(&stack);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################
// **********************************  communication  ******************************************************************

static int kdtree_attr_to_int(PyObject *attr, kd_intp *attr_out)
{
    int output;
    if (attr == NULL || !PyArg_Parse(attr, "i", &output)) {
        return 0;
    }
    *attr_out = (kd_intp)output;
    return 1;
}

static int kdtree_attr_to_double(PyObject *attr, kd_double *attr_out)
{
    double output;
    if (attr == NULL || !PyArg_Parse(attr, "d", &output)) {
        return 0;
    }
    *attr_out = (kd_double)output;
    return 1;
}

static int kdtree_attr_to_kdarray(PyObject *attr, KDarray *attr_out, int dtype_int)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    PyArrayObject *arr;

    if (attr == NULL) {
        return 0;
    }

    arr = (PyArrayObject *)PyArray_CheckFromAny(attr, NULL, 0, 0, flags, NULL);
    if (arr == NULL || !kdarray_valid_dtype(PyArray_TYPE(arr), dtype_int)) {
        return 0;
    }
    kdarray_init(attr_out, arr);
    return 1;
}

static int kdtree_build_traverser(PyObject *pynode, KDtree *tree, KDnode *node)
{
    if (pynode == NULL) {
        return 1;
    }
    if (!kdtree_attr_to_int(PyObject_GetAttrString(pynode, "start_index"), &node->start_index) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pynode, "end_index"), &node->end_index) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pynode, "children"), &node->children) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pynode, "split_dim"), &node->split_dim) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pynode, "lesser_index"), &node->lesser_index) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pynode, "higher_index"), &node->higher_index) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pynode, "level"), &node->level) ||
        !kdtree_attr_to_double(PyObject_GetAttrString(pynode, "split_val"), &node->split_val)) {
        return 0;
    }
    if (node->split_dim != -1) {
        if (kdtree_build_traverser(PyObject_GetAttrString(pynode, "lesser"), tree, tree->tree_list + node->lesser_index) &&
            kdtree_build_traverser(PyObject_GetAttrString(pynode, "higher"), tree, tree->tree_list + node->higher_index)) {
            node->lesser = tree->tree_list + node->lesser_index;
            node->higher = tree->tree_list + node->higher_index;
            return 1;
        }
        return 0;
    }
    return 1;
}

int PYCV_input_to_KDtree(PyObject *pytree, KDtree *tree)
{
    if (pytree == NULL) {
        return 0;
    }

    if (!kdtree_attr_to_int(PyObject_GetAttrString(pytree, "m"), &tree->m) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pytree, "n"), &tree->n) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pytree, "leafsize"), &tree->leafsize) ||
        !kdtree_attr_to_int(PyObject_GetAttrString(pytree, "size"), &tree->size) ||
        !kdtree_attr_to_kdarray(PyObject_GetAttrString(pytree, "data"), &tree->data, 0) ||
        !kdtree_attr_to_kdarray(PyObject_GetAttrString(pytree, "dims_min"), &tree->dims_min, 0) ||
        !kdtree_attr_to_kdarray(PyObject_GetAttrString(pytree, "dims_max"), &tree->dims_max, 0) ||
        !kdtree_attr_to_kdarray(PyObject_GetAttrString(pytree, "indices"), &tree->indices, 1)) {
        return 0;
    }

    if (!tree->n) {
        tree->tree_list = NULL;
        tree->tree = NULL;
        return 1;
    }

    tree->tree_list = malloc(tree->size * sizeof(KDnode));
    if (!tree->tree_list) {
        return 0;
    }
    if (!kdtree_build_traverser(PyObject_GetAttrString(pytree, "tree"), tree, tree->tree_list)) {
        return 0;
    }
    tree->tree = tree->tree_list;
    return 1;
}

void PYCV_KDtree_free(KDtree *self)
{
    if (self->size) {
        free(self->tree_list);
    }
    self->n = 0;
    self->m = 0;
    self->leafsize = 0;
    self->size = 0;
    self->tree_list = NULL;
    self->tree = NULL;
    Py_XDECREF(self->data.object);
    Py_XDECREF(self->dims_min.object);
    Py_XDECREF(self->dims_max.object);
    Py_XDECREF(self->indices.object);
}

// #####################################################################################################################
// **********************************  python calls  *******************************************************************

int PYCV_KDtree_build(KDtree *self, PyArrayObject *data, PyArrayObject *dims_min, PyArrayObject *dims_max,
                      PyArrayObject *indices, kd_intp leafsize, PyObject **output_list)
{
    kd_double *dims_min_l, *dims_max_l;
    kd_intp ii;
    KDnode *node;
    PyObject *node_dict;

    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Error: array need to be 2 dimensional \n");
        return 0;
    }

    if (!kdarray_valid_dtype(PyArray_TYPE(data), 0) ||
        !kdarray_valid_dtype(PyArray_TYPE(dims_min), 0) ||
        !kdarray_valid_dtype(PyArray_TYPE(dims_max), 0) ||
        !kdarray_valid_dtype(PyArray_TYPE(indices), 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid array dtype \n");
        return 0;
    }

    self->n = (kd_intp)PyArray_DIM(data, 0);
    self->m = (kd_intp)PyArray_DIM(data, 1);
    self->leafsize = leafsize;
    self->size = 0;
    kdarray_init(&self->data, data);
    kdarray_init(&self->dims_min, dims_min);
    kdarray_init(&self->dims_max, dims_max);
    kdarray_init(&self->indices, indices);

    self->tree_list = NULL;
    self->tree = NULL;

    dims_min_l = calloc(self->m, sizeof(kd_double));
    dims_max_l = calloc(self->m, sizeof(kd_double));

    if (!dims_min_l || !dims_max_l) {
        PyErr_NoMemory();
        return 0;
    }

    if (kdtree_build(self, 0, self->n, dims_min_l, dims_max_l, 0) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: kdtree_build \n");
        goto exit;
    }

    *output_list = PyList_New(self->size);
    for (ii = 0; ii < self->size; ii++) {
        node = self->tree_list + ii;
        node_dict = Py_BuildValue("{s:i,s:i,s:i,s:f,s:i,s:i,s:i}",
                                  "start_index", node->start_index, "end_index", node->end_index,
                                  "split_dim", node->split_dim, "split_val", node->split_val,
                                  "lesser_index", node->lesser_index,  "higher_index", node->higher_index,
                                  "level", node->level);
        PyList_SET_ITEM(*output_list, ii, node_dict);
    }

    exit:
        free(dims_min_l);
        free(dims_max_l);
        return PyErr_Occurred() ? 0 : 1;
}

int PYCV_KDtree_knn_query(KDtree *self, PyArrayObject *points, PyArrayObject *k,
                          kd_intp pnorm, int is_inf, kd_double epsilon, kd_double distance_max,
                          PyObject **output)
{
    KDresults results;
    kd_intp np, ii, ki;
    PYCV_ArrayIterator iter_s, iter_k, iter_d, iter_i;
    int num_type_s, num_type_k, num_type_d, num_type_i;
    KDarray kd_p;
    char *k_ptr = NULL, *p_ptr = NULL, *s_ptr = NULL, *d_ptr = NULL, *i_ptr = NULL;
    npy_intp slice_dims[1] = {0}, knn_dims[1] = {0};
    PyArrayObject *slice, *indices, *dist;
    kd_double *point;

    if (PyArray_NDIM(points) != 2 || PyArray_DIM(points, 1) != self->m || !kdarray_valid_dtype(PyArray_TYPE(points), 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid points shape or dtype \n");
        return 0;
    }

    if (!kdresults_init(&results, self->size, 1)) {
        PyErr_NoMemory();
        return 0;
    }

    point = malloc(self->m * sizeof(kd_double));
    if (!point) {
        kdresults_free(&results);
        PyErr_NoMemory();
        return 0;
    }

    np = (kd_intp)PyArray_DIM(points, 0);

    slice_dims[0] = np + 1;
    slice = (PyArrayObject *)PyArray_EMPTY(1, slice_dims, NPY_INT64, 0);

    PYCV_ArrayIteratorInit(slice, &iter_s);
    PYCV_ArrayIteratorInit(k, &iter_k);

    num_type_s = PyArray_TYPE(slice);
    num_type_k = PyArray_TYPE(k);

    k_ptr = (void *)PyArray_DATA(k);
    s_ptr = (void *)PyArray_DATA(slice);
    kdarray_init(&kd_p, points);
    p_ptr = kd_p.ptr;

    PYCV_SET_VALUE(num_type_s, s_ptr, 0);
    PYCV_ARRAY_ITERATOR_NEXT(iter_s, s_ptr);

    for (ii = 0; ii < np; ii++) {
        PYCV_GET_VALUE(num_type_k, kd_intp, k_ptr, ki);
        kdarray_allocate_point(&kd_p, p_ptr, point, self->m);

        if (!kdtree_knn_query(self, point, ki, pnorm, is_inf, distance_max, epsilon, &results)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: kdtree_knn_query \n");
            goto exit;
        }

        PYCV_SET_VALUE(num_type_s, s_ptr, results.n);
        PYCV_ARRAY_ITERATOR_NEXT2(iter_s, s_ptr, iter_k, k_ptr);
        p_ptr += kd_p.itemsize * self->m;
    }

    knn_dims[0] = results.n;
    indices = (PyArrayObject *)PyArray_EMPTY(1, knn_dims, NPY_INT64, 0);
    dist = (PyArrayObject *)PyArray_EMPTY(1, knn_dims, NPY_DOUBLE, 0);

    PYCV_ArrayIteratorInit(indices, &iter_i);
    PYCV_ArrayIteratorInit(dist, &iter_d);

    num_type_i = PyArray_TYPE(indices);
    num_type_d = PyArray_TYPE(dist);

    i_ptr = (void *)PyArray_DATA(indices);
    d_ptr = (void *)PyArray_DATA(dist);

    for (ii = 0; ii < results.n; ii++) {
        PYCV_SET_VALUE(num_type_i, i_ptr, *(results.indices + ii));
        PYCV_SET_VALUE_F2A(num_type_d, d_ptr, *(results.distance + ii));
        PYCV_ARRAY_ITERATOR_NEXT2(iter_i, i_ptr, iter_d, d_ptr);
    }

    *output = Py_BuildValue("(O,O,O)",
                            (PyObject *)dist,
                            (PyObject *)indices,
                            (PyObject *)slice);

    exit:
        free(point);
        kdresults_free(&results);
        return PyErr_Occurred() ? 0 : 1;
}

int PYCV_query_ball_points(KDtree *self, PyArrayObject *points, PyArrayObject *radius, kd_intp pnorm,
                           int is_inf, kd_double epsilon, PyObject **output)
{
    KDresults results;
    kd_intp np, ii;
    PYCV_ArrayIterator iter_s, iter_r, iter_d, iter_i;
    int num_type_s, num_type_r, num_type_d, num_type_i;
    KDarray kd_p;
    char *r_ptr = NULL, *p_ptr = NULL, *s_ptr = NULL, *d_ptr = NULL, *i_ptr = NULL;
    npy_intp slice_dims[1] = {0}, ind_dims[1] = {0};
    PyArrayObject *slice, *indices;
    kd_double *point, ri;

    if (PyArray_NDIM(points) != 2 || PyArray_DIM(points, 1) != self->m || !kdarray_valid_dtype(PyArray_TYPE(points), 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid points shape or dtype \n");
        return 0;
    }

    if (!kdresults_init(&results, self->size, 0)) {
        PyErr_NoMemory();
        return 0;
    }

    point = malloc(self->m * sizeof(kd_double));
    if (!point) {
        kdresults_free(&results);
        PyErr_NoMemory();
        return 0;
    }

    np = (kd_intp)PyArray_DIM(points, 0);

    slice_dims[0] = np + 1;
    slice = (PyArrayObject *)PyArray_EMPTY(1, slice_dims, NPY_INT64, 0);

    PYCV_ArrayIteratorInit(slice, &iter_s);
    PYCV_ArrayIteratorInit(radius, &iter_r);

    num_type_s = PyArray_TYPE(slice);
    num_type_r = PyArray_TYPE(radius);

    r_ptr = (void *)PyArray_DATA(radius);
    s_ptr = (void *)PyArray_DATA(slice);
    kdarray_init(&kd_p, points);
    p_ptr = kd_p.ptr;

    PYCV_SET_VALUE(num_type_s, s_ptr, 0);
    PYCV_ARRAY_ITERATOR_NEXT(iter_s, s_ptr);

    for (ii = 0; ii < np; ii++) {
        PYCV_GET_VALUE(num_type_r, kd_double, r_ptr, ri);
        kdarray_allocate_point(&kd_p, p_ptr, point, self->m);

        if (!kdtree_ball_query(self, point, ri, pnorm, is_inf, epsilon, &results)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: kdtree_ball_query \n");
            goto exit;
        }
        PYCV_SET_VALUE(num_type_s, s_ptr, results.n);
        PYCV_ARRAY_ITERATOR_NEXT2(iter_s, s_ptr, iter_r, r_ptr);
        p_ptr += kd_p.itemsize * self->m;
    }

    ind_dims[0] = results.n;
    indices = (PyArrayObject *)PyArray_EMPTY(1, ind_dims, NPY_INT64, 0);

    PYCV_ArrayIteratorInit(indices, &iter_i);
    num_type_i = PyArray_TYPE(indices);
    i_ptr = (void *)PyArray_DATA(indices);

    for (ii = 0; ii < results.n; ii++) {
        PYCV_SET_VALUE(num_type_i, i_ptr, *(results.indices + ii));
        PYCV_ARRAY_ITERATOR_NEXT(iter_i, i_ptr);
    }

    *output = Py_BuildValue("(O,O)",
                            (PyObject *)indices,
                            (PyObject *)slice);

    exit:
        free(point);
        kdresults_free(&results);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################









