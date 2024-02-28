#include "c_pycv_base.h"
#include "c_pycv_kd_tree.h"

// #####################################################################################################################

static int kdtree_valid_double_type(int num_type)
{
    switch (num_type) {
        case NPY_DOUBLE:
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be float64 \n");
            return 0;
    }
    return 1;
}

static int kdtree_valid_int_type(int num_type)
{
    switch (num_type) {
        case NPY_INT:
        case NPY_LONG:
        case NPY_LONGLONG:
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Error: data type need to be int \n");
            return 0;
    }
    return 1;
}

void PYCV_KDarray_init(KDarray *self, PyArrayObject *array)
{
    self->base = self->ptr = (void *)PyArray_DATA(array);
    self->numtype = PyArray_TYPE(array);
    self->itemsize = PyArray_ITEMSIZE(array);
    self->object = array;
}

static kdtree_intp kdarray_get_ivalue(char *ptr, int numtype)
{
    kdtree_intp out = 0;
    switch (numtype) {
        case NPY_SHORT:
            out = (kdtree_intp)(*((npy_short *)ptr));
            break;
        case NPY_INT:
            out = (kdtree_intp)(*((npy_int *)ptr));
            break;
        case NPY_LONG:
            out = (kdtree_intp)(*((npy_long *)ptr));
            break;
        case NPY_LONGLONG:
            out = (kdtree_intp)(*((npy_longlong *)ptr));
            break;
    }
    return out;
}

static npy_double kdarray_get_dvalue(char *ptr, int numtype)
{
    kdtree_double out;
    switch (numtype) {
        case NPY_DOUBLE:
            out = (kdtree_double)(*((npy_double *)ptr));
            break;
    }
    return out;
}

// #####################################################################################################################

static int KDtree_list_push_back(KDtree *self)
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
    memset(&self->tree_list[self->size], 0, sizeof(KDnode));
    self->size++;
    return 1;
}

static void kdtree_swap(char *i1, char *i2)
{
    char tmp = *i1;
    *i1 = *i2;
    *i2 = tmp;
}

static kdtree_intp kdtree_partition_index_compare(KDarray indices, kdtree_intp low, kdtree_intp high,
                                                  KDarray data, kdtree_intp m)
{
    char *p_ind_ii = indices.ptr + low * indices.itemsize;
    char *p_ind_jj = p_ind_ii;
    kdtree_intp ii = low, jj;
    kdtree_intp ind = kdarray_get_ivalue(indices.ptr + high * indices.itemsize, indices.numtype);
    kdtree_double pivot = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype);

    for (jj = low; jj < high; jj++) {
        ind = kdarray_get_ivalue(p_ind_jj, indices.numtype);

        if (kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype) < pivot) {
            kdtree_swap(p_ind_ii, p_ind_jj);
            p_ind_ii += indices.itemsize;
            ii++;
        }
        p_ind_jj += indices.itemsize;
    }
    kdtree_swap(p_ind_ii, p_ind_jj);
    return ii;
}

static kdtree_intp kdtree_partition_pivot(KDarray indices, kdtree_intp low, kdtree_intp high, KDarray data,
                                          kdtree_intp m, kdtree_double pivot)
{
    kdtree_intp ii = low, jj, ind;
    char *p_ind_ii = indices.ptr + low * indices.itemsize;
    char *p_ind_jj = p_ind_ii;

    for (jj = low; jj < high; jj++) {
        ind = kdarray_get_ivalue(p_ind_jj, indices.numtype);
        if (kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype) < pivot) {
            kdtree_swap(p_ind_ii, p_ind_jj);
            p_ind_ii += indices.itemsize;
            ii++;
        }
        p_ind_jj += indices.itemsize;
    }
    kdtree_swap(p_ind_ii, p_ind_jj);
    return ii;
}

static void kdtree_nth_element(KDarray indices, kdtree_intp low, kdtree_intp high, kdtree_intp nth,
                               KDarray data, kdtree_intp m)
{
    kdtree_intp ii = kdtree_partition_index_compare(indices, low, high, data, m);
    if (nth < ii) {
        kdtree_nth_element(indices, low, ii - 1, nth, data, m);
    } else if (nth > ii) {
        kdtree_nth_element(indices, ii + 1, high, nth, data, m);
    }
}

static kdtree_intp kdtree_min_element(KDarray indices, kdtree_intp low, kdtree_intp high, KDarray data, kdtree_intp m)
{
    kdtree_intp jj, min_element = low;
    char *p_ind_ii = indices.ptr + low * indices.itemsize;
    char *p_ind_jj = p_ind_ii + indices.itemsize;
    kdtree_intp ind = kdarray_get_ivalue(p_ind_ii, indices.numtype);
    kdtree_double min_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype), tmp_val;

    for (jj = low + 1; jj < high; jj++) {
        ind = kdarray_get_ivalue(p_ind_jj, indices.numtype);
        tmp_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype);
        if (tmp_val < min_val) {
            min_element = jj;
            min_val = tmp_val;
        }
    }
    return min_element;
}

static kdtree_intp kdtree_max_element(KDarray indices, kdtree_intp low, kdtree_intp high, KDarray data, kdtree_intp m)
{
    kdtree_intp jj, max_element = low;
    char *p_ind_ii = indices.ptr + low * indices.itemsize;
    char *p_ind_jj = p_ind_ii + indices.itemsize;
    kdtree_intp ind = kdarray_get_ivalue(p_ind_ii, indices.numtype);
    kdtree_double max_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype), tmp_val;

    for (jj = low + 1; jj < high; jj++) {
        ind = kdarray_get_ivalue(p_ind_jj, indices.numtype);
        tmp_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype);
        if (tmp_val > max_val) {
            max_element = jj;
            max_val = tmp_val;
        }
    }
    return max_element;
}

static void kdtree_fill_min_max_dims(KDarray data, KDarray indices, kdtree_double *dims_min, kdtree_double *dims_max,
                                     kdtree_intp start_index, kdtree_intp end_index, kdtree_intp m)
{
    char *p_ind_ii = indices.ptr + start_index * indices.itemsize;
    kdtree_intp jj, ii, ind = kdarray_get_ivalue(p_ind_ii, indices.numtype);
    char *p_data = data.ptr + ind * m * data.itemsize;
    kdtree_double tmp;

    for (jj = 0; jj < m; jj++) {
        dims_min[jj] = kdarray_get_dvalue(p_data, data.numtype);
        dims_max[jj] = kdarray_get_dvalue(p_data, data.numtype);
        p_data += data.itemsize;
    }
    p_ind_ii += indices.itemsize;
    for (ii = start_index + 1; ii < end_index; ii++) {
        ind = kdarray_get_ivalue(p_ind_ii, indices.numtype);
        p_data = data.ptr + ind * m * data.itemsize;
        for (jj = 0; jj < m; jj++) {
            tmp = kdarray_get_dvalue(p_data, data.numtype);
            dims_min[jj] = dims_min[jj] < tmp ? dims_min[jj] : tmp;
            dims_max[jj] = dims_max[jj] > tmp ? dims_max[jj] : tmp;
            p_data += data.itemsize;
        }
        p_ind_ii += indices.itemsize;
    }
}

// *********************************************************************************************************************

static kdtree_intp KDtree_build(KDtree *self, kdtree_intp start_index, kdtree_intp end_index,
                                kdtree_double *dims_min, kdtree_double *dims_max, kdtree_intp level)
{
    const kdtree_intp m = self->m;
    KDarray data = self->data, indices = self->indices;
    kdtree_double dims_delta, split_val;
    kdtree_intp node_index = self->size, split_dim, pp, _lesser, _higher;
    KDnode *root, *node;
    kdtree_intp jj, ind;

    data.ptr = data.base;

    if (!KDtree_list_push_back(self)) {
        PyErr_NoMemory();
        return -1;
    }

    root = self->tree_list;
    node = &root[node_index];

    node->start_index = start_index;
    node->end_index = end_index;
    node->children = end_index - start_index;
    node->level = level;

    if (node->children <= self->leafsize) {
        node->split_dim = -1;
        return node_index;
    }

    kdtree_fill_min_max_dims(data, indices, dims_min, dims_max, start_index, end_index, m);
    dims_delta = 0;
    split_dim = 0;

    for (jj = 0; jj < m; jj++) {
        if (dims_delta < dims_max[jj] - dims_min[jj]) {
            dims_delta = dims_max[jj] - dims_min[jj];
            split_dim = jj;
        }
    }

    if (dims_min[split_dim] == dims_max[split_dim]) {
        node->split_dim = -1;
        return node_index;
    }

    node->split_dim = split_dim;
    data.ptr += split_dim * data.itemsize;

    kdtree_intp nth = start_index + node->children / 2;
    kdtree_nth_element(indices, start_index, end_index - 1, nth, data, m);

    ind = kdarray_get_ivalue(indices.ptr + nth * indices.itemsize, indices.numtype);
    split_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype);

    pp = kdtree_partition_pivot(indices, start_index, nth, data, m, split_val);

    if (pp == start_index) {
        kdtree_intp min_index = kdtree_min_element(indices, start_index, end_index, data, m);

        ind = kdarray_get_ivalue(indices.ptr + min_index * indices.itemsize, indices.numtype);
        split_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype);
        split_val = (kdtree_double)nextafter((double)split_val, HUGE_VAL);

        pp = kdtree_partition_pivot(indices, start_index, end_index - 1, data, m, split_val);
    } else if (pp == end_index) {
        kdtree_intp max_index = kdtree_max_element(indices, start_index, end_index, data, m);

        ind = kdarray_get_ivalue(indices.ptr + max_index * indices.itemsize, indices.numtype);
        split_val = kdarray_get_dvalue(data.ptr + ind * m * data.itemsize, data.numtype);

        pp = kdtree_partition_pivot(indices, start_index, end_index - 1, data, m, split_val);
    }

    node->split_val = split_val;

    _lesser = KDtree_build(self, start_index, pp, dims_min, dims_max, level + 1);
    _higher = KDtree_build(self, pp, end_index, dims_min, dims_max, level + 1);

    root = self->tree_list;
    node = &root[node_index];

    if (!node_index) {
        self->tree = root;
    }

    node->lesser_index = _lesser;
    node->higher_index = _higher;
    node->lesser = root + _lesser;
    node->higher = root + _higher;

    return node_index;
}

// *********************************************************************************************************************

int PYCV_KDtree_build(KDtree *self,
                      PyArrayObject *data,
                      PyArrayObject *dims_min,
                      PyArrayObject *dims_max,
                      PyArrayObject *indices,
                      kdtree_intp leafsize)
{
    kdtree_double *dims_min_l, *dims_max_l;

    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Error: array need to be 2 dimensional \n");
        return 0;
    }

    if (!kdtree_valid_double_type(PyArray_TYPE(data)) ||
        !kdtree_valid_double_type(PyArray_TYPE(dims_min)) ||
        !kdtree_valid_double_type(PyArray_TYPE(dims_max)) ||
        !kdtree_valid_int_type(PyArray_TYPE(indices))) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid array dtype \n");
        return 0;
    }

    self->n = (kdtree_intp)PyArray_DIM(data, 0);
    self->m = (kdtree_intp)PyArray_DIM(data, 1);
    self->leafsize = leafsize;
    self->size = 0;
    PYCV_KDarray_init(&self->data, data);
    PYCV_KDarray_init(&self->dims_min, dims_min);
    PYCV_KDarray_init(&self->dims_max, dims_max);
    PYCV_KDarray_init(&self->indices, indices);

    self->tree_list = NULL;
    self->tree = NULL;

    dims_min_l = calloc(self->m, sizeof(kdtree_double));
    dims_max_l = calloc(self->m, sizeof(kdtree_double));

    if (!dims_min_l || !dims_max_l) {
        PyErr_NoMemory();
        return 0;
    }

    if (KDtree_build(self, 0, self->n, dims_min_l, dims_max_l, 0) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Error: KDtree_build \n");
        goto exit;
    }

    exit:
        free(dims_min_l);
        free(dims_max_l);
        return PyErr_Occurred() ? 0 : 1;
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
}

// #####################################################################################################################

static int kdheap_init(KDheap *self, kdtree_intp init_size)
{
    self->n = 0;
    self->_has_n = init_size;
    self->heap = malloc(init_size * sizeof(KDheap_item));
    if (!self->heap) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static int kdheap_allocate(KDheap *self)
{
    self->n++;
    if (self->n > self->_has_n) {
        self->_has_n = 2 * self->_has_n + 1;
        self->heap = realloc(self->heap, self->_has_n * sizeof(KDheap_item));
        if (!self->heap) {
            PyErr_NoMemory();
            return 0;
        }
    }
    return 1;
}

static void kdheap_push_down(KDheap_item *heap, kdtree_intp start_pos, kdtree_intp pos)
{
    KDheap_item new_item = *(heap + pos), parent;
    kdtree_intp parent_pp, pp = pos;
    while (pp > start_pos) {
        parent_pp = (pp - 1) >> 1;
        parent = *(heap + parent_pp);
        if (new_item.priority < parent.priority) {
            *(heap + pp) = parent;
            pp = parent_pp;
        } else {
            break;
        }
    }
    *(heap + pp) = new_item;
}

static void kdheap_push_up(KDheap_item *heap, kdtree_intp start_pos, kdtree_intp end_pos)
{
    KDheap_item new_item = *(heap + start_pos), child;
    kdtree_intp child_pp = 2 * start_pos + 1, pp = start_pos;
    while (child_pp < end_pos) {
        child = *(heap + child_pp);
        if (child_pp + 1 < end_pos && !(child.priority < (*(heap + child_pp + 1)).priority)) {
            child_pp += 1;
        }
        *(heap + pp) = *(heap + child_pp);
        pp = child_pp;
        child_pp = 2 * pp + 1;
    }
    *(heap + pp) = new_item;
    kdheap_push_down(heap, start_pos, pp);
}

static int kdheap_push(KDheap *self, KDheap_item item)
{
    kdtree_intp nn = self->n;
    if (!kdheap_allocate(self)) {
        return 0;
    }
    self->heap[nn] = item;
    kdheap_push_down(self->heap, 0, nn);
    return 1;
}

static KDheap_item kdheap_pop(KDheap *self)
{
    KDheap_item last = *(self->heap + self->n - 1), out;
    self->n -= 1;
    if (self->n > 0) {
        out = *(self->heap);
        *(self->heap) = last;
        kdheap_push_up(self->heap, 0, self->n);
        return out;
    }
    return last;
}

static void kdheap_heapify(KDheap *self)
{
    kdtree_intp nn = self->n, ii = self->n / 2;
    while (ii) {
        kdheap_push_up(self->heap, ii, self->n);
        ii--;
    }
}

static void kdheap_free(KDheap *self)
{
    if (self->_has_n) {
        free(self->heap);
    }
}

static int kdquery_item_init(KDquery_item *self, kdtree_intp m)
{
    self->min_distance = 0;
    self->split_distance = malloc(m * sizeof(kdtree_double));
    if (!self->split_distance) {
        PyErr_NoMemory();
        return 0;
    }
    self->node = NULL;
    return 1;
}

static int kdquery_item_init_from(KDquery_item *self, KDquery_item *from, kdtree_intp m)
{
    self->min_distance = from->min_distance;
    memcpy(self->split_distance, from->split_distance, m * sizeof(kdtree_double));
    if (!self->split_distance) {
        PyErr_NoMemory();
        return 0;
    }
    self->node = NULL;
    return 1;
}

static void kdquery_item_free(KDquery_item *self)
{
    free(self->split_distance);
    self->node = NULL;
}

// *********************************************************************************************************************

static kdtree_double kdtree_minkowski_distance_p1p2(KDarray p1, KDarray p2, kdtree_intp m, kdtree_intp p, int is_inf)
{
    kdtree_double out = 0, pp1, pp2, tmp;
    kdtree_intp ii;
    char *p1_data = p1.ptr, *p2_data = p2.ptr;

    for (ii = 0; ii < m; ii++) {
        pp1 = kdarray_get_dvalue(p1_data, p1.numtype);
        pp2 = kdarray_get_dvalue(p2_data, p2.numtype);
        tmp = pp2 > pp1 ? pp2 - pp1 : pp1 - pp2;

        if (is_inf) {
            out = tmp > out ? tmp : out;
        } else if (p == 1) {
            out += tmp;
        } else {
            out += (kdtree_double)pow((double)tmp, (double)p);
        }
        p1_data += p1.itemsize;
        p2_data += p2.itemsize;
    }
    return out;
}

static kdtree_double kdtree_minkowski_distance_p1(kdtree_double p1, kdtree_intp p, int is_inf)
{
    if (is_inf || p == 1) {
        return p1 < 0 ? -p1 : p1;
    }
    return (kdtree_double)pow((double)p1, (double)p);
}

static kdtree_double kdtree_split_distance(KDarray p, KDarray dims_min, KDarray dims_max)
{
    kdtree_double pp = kdarray_get_dvalue(p.ptr, p.numtype);
    kdtree_double p_min = kdarray_get_dvalue(dims_min.ptr, dims_min.numtype);
    kdtree_double p_max = kdarray_get_dvalue(dims_max.ptr, dims_max.numtype);
    if (pp > p_max) {
        return pp - p_max;
    } else if (pp < p_min) {
        return p_min - pp;
    }
    return 0; // In the range;
}

// *********************************************************************************************************************

static int kdquery_buffer_init(KDquery_buffer *self, kdtree_intp init_size)
{
    self->n = 0;
    self->_has_n = init_size;
    self->buffer = malloc(init_size * sizeof(KDquery_item));
    if (!self->buffer) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void kdquery_buffer_free(KDquery_buffer *self)
{
    kdtree_intp ii;
    if (self->_has_n) {
        for (ii = 0; ii < self->n; ii++) {
            kdquery_item_free(&self->buffer[ii]);
        }
        free(self->buffer);
    }
}

static int kdquery_buffer_allocate(KDquery_buffer *self, kdtree_intp m)
{
    self->n++;
    if (self->n > self->_has_n) {
        self->_has_n++;
        self->buffer = realloc(self->buffer, self->_has_n * sizeof(KDquery_item));
        if (!self->buffer) {
            PyErr_NoMemory();
            return 0;
        }
    }
    memset(&self->buffer[self->n - 1], 0, sizeof(KDquery_item));
    if (!kdquery_item_init(&self->buffer[self->n - 1], m)) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

// *********************************************************************************************************************

static kdtree_intp kdtree_query_point(KDtree tree,
                                      KDarray point,
                                      kdtree_intp k,
                                      kdtree_intp p,
                                      int is_inf,
                                      kdtree_double distance_max,
                                      kdtree_double epsilon,
                                      KDarray dist_out,
                                      KDarray indices_out)
{
    KDheap neighbors, queue;
    KDquery_buffer buffer;
    KDquery_item *near, *far;
    KDnode *neighbor, *c_node;
    KDheap_item next;

    KDarray data = tree.data, indices = tree.indices;
    kdtree_intp m = tree.m, out = 0;

    kdtree_double dist_bound, eps, dist, split_dist, tmp;
    kdtree_intp ii, ind;

    char *hold_ptr = point.ptr;

    if (!kdheap_init(&neighbors, k)) {
        PyErr_NoMemory();
        return 0;
    }

    if (!kdheap_init(&queue, tree.n)) {
        kdheap_free(&neighbors);
        PyErr_NoMemory();
        return 0;
    }

    if (!kdquery_buffer_init(&buffer, tree.size)) {
        kdheap_free(&neighbors);
        kdheap_free(&queue);
        PyErr_NoMemory();
        return 0;
    }

    if (!kdquery_buffer_allocate(&buffer, m)) {
        PyErr_NoMemory();
        goto exit;
    }

    near = (KDquery_item *)(buffer.buffer + (buffer.n - 1));
    near->node = tree.tree;

    for (ii = 0; ii < tree.m; ii++) {
        dist = kdtree_split_distance(point, tree.dims_min, tree.dims_max);
        dist = kdtree_minkowski_distance_p1(dist, p, is_inf);

        *(near->split_distance + ii) = dist;
        if (is_inf) {
            near->min_distance = dist > near->min_distance ? dist : near->min_distance;
        } else {
            near->min_distance += dist;
        }
        point.ptr += point.itemsize;
        tree.dims_min.ptr += tree.dims_min.itemsize;
        tree.dims_max.ptr += tree.dims_max.itemsize;
    }
    point.ptr = hold_ptr;
    tree.dims_min.ptr = tree.dims_min.base;
    tree.dims_max.ptr = tree.dims_max.base;

    dist_bound = distance_max;
    eps = epsilon;
    if (!is_inf && p > 1) {
        dist_bound = (kdtree_double)pow((double)dist_bound, (double)p);
    }

    if (eps == 0) {
        eps = 1;
    } else if (is_inf) {
        eps = 1 / (1 + eps);
    } else {
        eps = 1 / (kdtree_double)pow((double)(1 + eps), (double)p);
    }

    while (1) {

        if (near->node->split_dim == -1) {
            neighbor = near->node;
            for (ii = neighbor->start_index; ii < neighbor->end_index; ii++) {
                ind = kdarray_get_ivalue(indices.base + ii * indices.itemsize, indices.numtype);
                data.ptr = data.base + ind * m * data.itemsize;
                dist = kdtree_minkowski_distance_p1p2(data, point, m, p, is_inf);
                if (dist < dist_bound) {
                    if (neighbors.n == k) {
                        kdheap_pop(&neighbors);
                    }
                    KDheap_item item1, *tmp1;
                    tmp1 = &item1;
                    tmp1->priority = -dist;
                    tmp1->contents.index = ind;
                    if (!kdheap_push(&neighbors, item1)) {
                        PyErr_SetString(PyExc_RuntimeError, "Error: kdheap_push");
                        goto exit;
                    }
                    if (neighbors.n == k) {
                        dist_bound = -(*neighbors.heap).priority;
                    }
                }
            }
            if (!queue.n) {
                break;
            }
            next = kdheap_pop(&queue);
            near = (KDquery_item *)next.contents.data_ptr;
        } else {
            if (near->min_distance > (dist_bound * eps)) {
                break;
            }
            if (!kdquery_buffer_allocate(&buffer, m)) {
                PyErr_NoMemory();
                goto exit;
            }
            far = (KDquery_item *)(buffer.buffer + (buffer.n - 1));

            if (!kdquery_item_init_from(far, near, m)) {
                PyErr_NoMemory();
                goto exit;
            }

            c_node = near->node;
            tmp = kdarray_get_dvalue(point.ptr + c_node->split_dim * point.itemsize, point.numtype);

            if (tmp < c_node->split_val) {
                near->node = c_node->lesser;
                far->node = c_node->higher;
                split_dist = c_node->split_val - tmp;
            } else {
                near->node = c_node->higher;
                far->node = c_node->lesser;
                split_dist = tmp - c_node->split_val;
            }
            split_dist = kdtree_minkowski_distance_p1(split_dist, p, is_inf);

            if (is_inf) {
                far->min_distance = far->min_distance > split_dist ? far->min_distance : split_dist;
            } else {
                far->min_distance += split_dist - far->split_distance[c_node->split_dim];
            }

            far->split_distance[c_node->split_dim] = split_dist;

            if (near->min_distance > far->min_distance) {
                KDquery_item *tmp = far;
                far = near;
                near = tmp;
            }

            if (far->min_distance <= (dist_bound * eps)) {
                KDheap_item item2, *tmp2;
                tmp2 = &item2;
                tmp2->priority = far->min_distance;
                tmp2->contents.data_ptr = far;
                if (!kdheap_push(&queue, item2)) {
                    PyErr_SetString(PyExc_RuntimeError, "Error: kdheap_push");
                    goto exit;
                }
            }
        }
    }

    out = neighbors.n;

    for (ii = out - 1; ii >= 0; ii--) {
        KDheap_item tmp = kdheap_pop(&neighbors);
        dist = -tmp.priority;
        if (!is_inf && p > 1) {
            dist = (kdtree_double)pow((double)dist, (1 / (double)p));
        }
        PYCV_SET_VALUE_F2A(dist_out.numtype, (dist_out.ptr + ii * dist_out.itemsize), dist);
        PYCV_SET_VALUE(indices_out.numtype, (indices_out.ptr + ii * indices_out.itemsize), tmp.contents.index);
    }

    exit:
        kdheap_free(&neighbors);
        kdheap_free(&queue);
        kdquery_buffer_free(&buffer);
        return out;
}

int PYCV_KDtree_query_knn(KDtree tree,
                          PyArrayObject *points,
                          KDarray k,
                          kdtree_intp p,
                          int is_inf,
                          kdtree_double distance_max,
                          kdtree_double epsilon,
                          KDarray dist,
                          KDarray indices,
                          KDarray slice)
{
    kdtree_intp ii, nn, m = tree.m, np, cumsum_nn = 0, ki;
    KDarray pp;

    if (PyArray_NDIM(points) != 2 || PyArray_DIM(points, 1) != m || !kdtree_valid_double_type(PyArray_TYPE(points))) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid points shape or dtype \n");
        return 0;
    }

    np = (kdtree_intp)PyArray_DIM(points, 0);

    PYCV_KDarray_init(&pp, points);

    PYCV_SET_VALUE(slice.numtype, slice.ptr, cumsum_nn);
    slice.ptr += slice.itemsize;

    for (ii = 0; ii < np; ii++) {
        ki = kdarray_get_ivalue(k.ptr, k.numtype);
        nn = kdtree_query_point(tree, pp, ki, p, is_inf, distance_max, epsilon, dist, indices);

        cumsum_nn += nn;
        PYCV_SET_VALUE(slice.numtype, slice.ptr, cumsum_nn);

        pp.ptr += pp.itemsize * m;
        dist.ptr += dist.itemsize * nn;
        indices.ptr += indices.itemsize * nn;
        slice.ptr += slice.itemsize;
        k.ptr += k.itemsize;
    }

    return 1;
}

// #####################################################################################################################

static void kdtree_distance_p_interval_1d(kdtree_double p1, kdtree_double inter_min, kdtree_double inter_max,
                                          kdtree_double *min_dist, kdtree_double *max_dist)
{
    kdtree_double tmp = p1 - inter_max > inter_min - p1 ? p1 - inter_max : inter_min - p1;
    *min_dist = tmp > 0 ? tmp : 0;
    *max_dist = inter_max - p1 > p1 - inter_min ? inter_max - p1 : p1 - inter_min;
}

static void kdtree_distance_p_interval(KDarray p1, kdtree_double *inter_min, kdtree_double *inter_max,
                                       kdtree_double *min_dist, kdtree_double *max_dist, kdtree_intp m,
                                       kdtree_intp p, int is_inf)
{
    kdtree_double min_, max_, pp;
    kdtree_intp ii;
    char *p_ptr = p1.ptr;
    *min_dist = 0;
    *max_dist = 0;
    for (ii = 0; ii < m; ii++) {
        pp = kdarray_get_dvalue(p_ptr, p1.numtype);
        kdtree_distance_p_interval_1d(pp, *(inter_min + ii), *(inter_max + ii), &min_, &max_);
        if (is_inf) {
            *min_dist = *min_dist > min_ ? *min_dist : min_;
            *max_dist = *max_dist > max_ ? *max_dist : max_;
        } else {
            *min_dist += (kdtree_double)pow((double)min_, (double)p);
            *max_dist += (kdtree_double)pow((double)max_, (double)p);
        }
        p_ptr += p1.itemsize;
    }
}


// *********************************************************************************************************************

static int kdball_results_push(KDball_results *self, kdtree_intp index)
{
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n += 1;
        self->list = realloc(self->list, self->_loc_n * sizeof(kdtree_intp));
        if (!self->list) {
            PyErr_NoMemory();
            return 0;
        }
    }
    *(self->list + (self->n - 1)) = index;
    return 1;
}

// *********************************************************************************************************************

static void kdball_tracking_adapt(KDball_tracking *self, KDtree tree, KDarray point)
{
    kdtree_intp ii;
    char *p_min, *p_max;
    self->n = 0;
    p_min = tree.dims_min.ptr;
    p_max = tree.dims_max.ptr;
    for (ii = 0; ii < self->m; ii++) {
        *(self->bound_min + ii) = kdarray_get_dvalue(p_min, tree.dims_min.numtype);
        *(self->bound_max + ii) = kdarray_get_dvalue(p_max, tree.dims_max.numtype);
        p_min += tree.dims_min.itemsize;
        p_max += tree.dims_max.itemsize;
    }
    kdtree_distance_p_interval(point, self->bound_min, self->bound_max, &(self->min_distance), &(self->max_distance),
                               self->m, self->p, self->is_inf);
}

static int kdball_tracking_init(KDball_tracking *self, KDtree tree, KDarray point, kdtree_intp p, int is_inf)
{
    self->m = tree.m;
    self->p = p;
    self->is_inf = is_inf;
    self->_loc_n = tree.size / 2 + 1;
    self->n = 0;
    self->stack = malloc(self->_loc_n * sizeof(KDtracking_item));
    self->bound_min = malloc(self->m * 2 * sizeof(kdtree_double));
    if (!self->stack || !self->bound_min) {
        PyErr_NoMemory();
        return 0;
    }
    self->bound_max = self->bound_min + self->m;
    kdball_tracking_adapt(self, tree, point);
    return 1;
}

static void kdball_tracking_free(KDball_tracking *self)
{
    free(self->bound_min);
    free(self->stack);
}

static int kdball_tracking_allocate(KDball_tracking *self)
{
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n += 1;
        self->stack = realloc(self->stack, self->_loc_n * sizeof(KDtracking_item));
        if (!self->stack) {
            PyErr_NoMemory();
            return 0;
        }
    }
    return 1;
}

static int kdball_tracking_push(KDball_tracking *self, KDarray point, int change_lesser,
                                kdtree_intp split_dim, kdtree_double split_val)
{
    kdtree_intp nn = self->n;
    KDtracking_item *item;

    if (!kdball_tracking_allocate(self)) {
        return 0;
    }

    item = self->stack + nn;
    item->split_dim = split_dim;
    item->min_distance = self->min_distance;
    item->max_distance = self->max_distance;
    item->split_min_distance = *(self->bound_min + split_dim);
    item->split_max_distance = *(self->bound_max + split_dim);

    if (change_lesser) {
        *(self->bound_max + split_dim) = split_val;
    } else {
        *(self->bound_min + split_dim) = split_val;
    }

    kdtree_distance_p_interval(point, self->bound_min, self->bound_max, &(self->min_distance), &(self->max_distance),
                               self->m, self->p, self->is_inf);
    return 1;
}

static int kdball_tracking_pop(KDball_tracking *self)
{
    KDtracking_item *item;
    if (self->n == 0) {
        return 0;
    }
    self->n--;
    item = self->stack + self->n;
    self->min_distance = item->min_distance;
    self->max_distance = item->max_distance;
    *(self->bound_min + item->split_dim) = item->split_min_distance;
    *(self->bound_max + item->split_dim) = item->split_max_distance;
    return 1;
}

static int kdball_traverser_collect(KDtree tree, KDnode *node, KDarray point, KDball_results *output)
{
    KDarray indices;
    kdtree_intp ii, ind;
    if (node->split_dim == -1) {
        indices = tree.indices;
        for (ii = node->start_index; ii < node->end_index; ii++) {
            ind = kdarray_get_ivalue(indices.base + ii * indices.itemsize, indices.numtype);
            if (!kdball_results_push(output, ind)) {
                return 0;
            }
        }
    } else {
        if (!kdball_traverser_collect(tree, node->lesser, point, output) ||
            !kdball_traverser_collect(tree, node->higher, point, output)) {
            return 0;
        }
    }
    return 1;
}

static int kdball_traverser_tracking(KDball_tracking *self, KDtree tree, KDnode *node, KDarray point,
                                             kdtree_double r, kdtree_double eps, KDball_results *output)
{
    kdtree_intp ii, ind, m = self->m;
    KDarray data, indices;
    kdtree_double dist;
    if (self->min_distance > (r * eps)) {
        return 1;
    }
    if (self->max_distance < (r / eps)) {
        return kdball_traverser_collect(tree, node, point, output);
    }

    if (node->split_dim == -1) {
        data = tree.data;
        indices = tree.indices;
        for (ii = node->start_index; ii < node->end_index; ii++) {
            ind = kdarray_get_ivalue(indices.base + ii * indices.itemsize, indices.numtype);
            data.ptr = data.base + ind * m * data.itemsize;
            dist = kdtree_minkowski_distance_p1p2(data, point, self->m, self->p, self->is_inf);
            if (dist <= r) {
                if (!kdball_results_push(output, ind)) {
                    return 0;
                }
            }
        }
    } else {
        if (!kdball_tracking_push(self, point, 1, node->split_dim, node->split_val) ||
            !kdball_traverser_tracking(self, tree, node->lesser, point, r, eps, output)) {
            return 0;
        }
        kdball_tracking_pop(self);

        if (!kdball_tracking_push(self, point, 0, node->split_dim, node->split_val) ||
            !kdball_traverser_tracking(self, tree, node->higher, point, r, eps, output)) {
            return 0;
        }
        kdball_tracking_pop(self);
    }
    return 1;
}

// *********************************************************************************************************************

int PYCV_query_ball_points(KDtree tree,
                           PyArrayObject *points,
                           PyArrayObject *radius,
                           kdtree_intp p,
                           int is_inf,
                           kdtree_double epsilon,
                           KDball_results *indices,
                           KDarray slice)
{
    kdtree_intp ii, jj, nn, m = tree.m, np, cumsum_nn = 0, ni;
    kdtree_double ri, eps = epsilon;
    KDarray point, rad;
    KDball_tracking tracking;

    if (PyArray_NDIM(points) != 2 || PyArray_DIM(points, 1) != m || !kdtree_valid_double_type(PyArray_TYPE(points))) {
        PyErr_SetString(PyExc_RuntimeError, "Error: invalid points shape or dtype \n");
        goto exit;
    }

    np = PyArray_DIM(points, 0);
    PYCV_KDarray_init(&point, points);
    PYCV_KDarray_init(&rad, radius);

    indices->n = 0;
    indices->_loc_n = tree.n;
    indices->list = malloc(tree.n * sizeof(kdtree_intp));
    if (!indices->list) {
        PyErr_NoMemory();
        return 0;
    }

    if (!kdball_tracking_init(&tracking, tree, point, p, is_inf)) {
        free(indices->list);
        PyErr_NoMemory();
        return 0;
    }

    if (eps == 0) {
        eps = 1;
    } else if (is_inf) {
        eps = 1 / (1 + eps);
    } else {
        eps = 1 / (kdtree_double)pow((double)(1 + eps), (double)p);
    }

    PYCV_SET_VALUE(slice.numtype, slice.ptr, cumsum_nn);
    slice.ptr += slice.itemsize;
    PYCV_SET_VALUE(slice.numtype, slice.ptr, cumsum_nn);

    for (ii = 0; ii < np; ii++) {
        ri = kdarray_get_dvalue(rad.ptr, rad.numtype);

        if (!is_inf && p > 1) {
            ri = (kdtree_double)pow((double)ri, (double)p);
        }

        if (!kdball_traverser_tracking(&tracking, tree, tree.tree, point, ri, eps, indices)) {
            PyErr_SetString(PyExc_RuntimeError, "Error: kdball_traverser_tracking \n");
            goto exit;
        }

        PYCV_SET_VALUE(slice.numtype, slice.ptr, indices->n);

        point.ptr += point.itemsize * m;
        rad.ptr += rad.itemsize;
        slice.ptr += slice.itemsize;

        if (np > 1 && ii + 1 < np) {
            indices->list = realloc(indices->list, (tree.n + indices->n) * sizeof(kdtree_intp));
            if (!indices->list) {
                kdball_tracking_free(&tracking);
                PyErr_NoMemory();
                return 0;
            }
            kdball_tracking_adapt(&tracking, tree, point);
        }
    }

    exit:
        kdball_tracking_free(&tracking);
        if (PyErr_Occurred()) {
            free(indices->list);
            return 0;
        }
        return 1;
}

// #####################################################################################################################
