#include "c_pycv_base.h"
#include "c_pycv_peaks.h"

// #####################################################################################################################

#define p_type npy_longlong

#define p_itype NPY_INT64

#define p_type_stride (int)NPY_SIZEOF_LONGLONG

// #####################################################################################################################

#define CASE_IS_LOCAL_MAX(_ntype, _dtype, _x, _n, _neigh, _flag, _cval, _is_max)                                       \
case NPY_##_ntype:                                                                                                     \
{                                                                                                                      \
    double _root = (double)(*(_dtype *)_x);                                                                            \
    double _val;                                                                                                       \
    _is_max = 1;                                                                                                       \
    for (int _ii = 0; _ii < _n; _ii++) {                                                                               \
        if (!*(_neigh + _ii)) {                                                                                        \
            continue;                                                                                                  \
        } else if (*(_neigh + _ii) == _flag) {                                                                         \
            _val = _cval;                                                                                              \
        } else {                                                                                                       \
            _val = (double)(*(_dtype *)(_x + *(_neigh + _ii)));                                                        \
        }                                                                                                              \
        if (_val > _root) {                                                                                            \
            _is_max = 0;                                                                                               \
            break;                                                                                                     \
        }                                                                                                              \
    }                                                                                                                  \
}                                                                                                                      \
break

#define IS_LOCAL_MAX(_ntype, _x, _n, _neigh, _flag, _cval, _is_max)                                                    \
{                                                                                                                      \
    switch (_ntype) {                                                                                                  \
        CASE_IS_LOCAL_MAX(BOOL, npy_bool, _x, _n, _neigh, _flag, _cval, _is_max);                                      \
        CASE_IS_LOCAL_MAX(UBYTE, npy_ubyte, _x, _n, _neigh, _flag, _cval, _is_max);                                    \
        CASE_IS_LOCAL_MAX(USHORT, npy_ushort, _x, _n, _neigh, _flag, _cval, _is_max);                                  \
        CASE_IS_LOCAL_MAX(UINT, npy_uint, _x, _n, _neigh, _flag, _cval, _is_max);                                      \
        CASE_IS_LOCAL_MAX(ULONG, npy_ulong, _x, _n, _neigh, _flag, _cval, _is_max);                                    \
        CASE_IS_LOCAL_MAX(ULONGLONG, npy_ulonglong, _x, _n, _neigh, _flag, _cval, _is_max);                            \
        CASE_IS_LOCAL_MAX(BYTE, npy_byte, _x, _n, _neigh, _flag, _cval, _is_max);                                      \
        CASE_IS_LOCAL_MAX(SHORT, npy_short, _x, _n, _neigh, _flag, _cval, _is_max);                                    \
        CASE_IS_LOCAL_MAX(INT, npy_int, _x, _n, _neigh, _flag, _cval, _is_max);                                        \
        CASE_IS_LOCAL_MAX(LONG, npy_long, _x, _n, _neigh, _flag, _cval, _is_max);                                      \
        CASE_IS_LOCAL_MAX(LONGLONG, npy_longlong, _x, _n, _neigh, _flag, _cval, _is_max);                              \
        CASE_IS_LOCAL_MAX(FLOAT, npy_float, _x, _n, _neigh, _flag, _cval, _is_max);                                    \
        CASE_IS_LOCAL_MAX(DOUBLE, npy_double, _x, _n, _neigh, _flag, _cval, _is_max);                                  \
    }                                                                                                                  \
}

// #####################################################################################################################

typedef struct {
    double priority;
    int index;
    int inp_shift;
    int out_shift;
    int neighbor_shift;
} HeapItem;

typedef struct {
    HeapItem *heap;
    int n;
    int _loc_n;
} Heap;

#define HEAP_NEW {NULL, 0, 0}

static int heap_init(Heap *self, int init_size)
{
    self->n = 0;
    self->_loc_n = init_size;
    self->heap = malloc(init_size * sizeof(HeapItem));
    if (!self->heap) {
        self->_loc_n = 0;
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

static void heap_free(Heap *self)
{
    if (self->_loc_n) {
        free(self->heap);
    }
}

static void heap_push_down(HeapItem *heap, int pos, int l)
{
    HeapItem new_item = *(heap + pos);
    int parent_pos;
    while (pos > l) {
        parent_pos = (pos - 1) >> 1;
        if ((new_item.priority < (heap + parent_pos)->priority) ||
            (new_item.priority == (heap + parent_pos)->priority && new_item.index < (heap + parent_pos)->index)) {
            *(heap + pos) = *(heap + parent_pos);
            pos = parent_pos;
        } else {
            break;
        }
    }
    *(heap + pos) = new_item;
}

static void heap_push_up(HeapItem *heap, int pos, int h)
{
    HeapItem new_item = *(heap + pos);
    int l = pos, child_pos = 2 * pos + 1;
    while (child_pos < h) {
        if (child_pos + 1 < h &&
            (((heap + child_pos)->priority > (heap + child_pos + 1)->priority) ||
              ((heap + child_pos)->priority == (heap + child_pos + 1)->priority &&
               (heap + child_pos)->index > (heap + child_pos + 1)->index))) {
            child_pos += 1;
        }
        *(heap + pos) = *(heap + child_pos);
        pos = child_pos;
        child_pos = 2 * pos + 1;
    }
    *(heap + pos) = new_item;
    heap_push_down(heap, pos, l);
}

static int heap_push(Heap *self, HeapItem new_item)
{
    int nn = self->n;
    self->n++;
    if (self->n > self->_loc_n) {
        self->_loc_n = 2 * self->_loc_n + 1;
        self->heap = realloc(self->heap, self->_loc_n * sizeof(new_item));
        if (!self->heap) {
            PyErr_NoMemory();
            return 0;
        }
    }
    *(self->heap + nn) = new_item;
    heap_push_down(self->heap, nn, 0);
    return 1;
}

static HeapItem heap_pop(Heap *self)
{
    HeapItem last = *(self->heap + self->n - 1), out;
    self->n--;
    if (self->n > 0) {
        out = *(self->heap);
        *(self->heap) = last;
        heap_push_up(self->heap, 0, self->n);
        return out;
    }
    return last;
}

// #####################################################################################################################

typedef struct {
    int ndim;
    int neighbors_size;
    npy_intp flag;
    npy_intp *offsets;
    npy_intp *offsets_norm;
    npy_intp neighbors_dims[NPY_MAXDIMS];
    npy_intp norm_strides[NPY_MAXDIMS];
    npy_intp input_dims[NPY_MAXDIMS];
    int n_dfs;
    int *dfs_ptr;
} Neighborhood;

static int Neighborhood_init(Neighborhood *self, PyArrayObject *input, npy_intp *min_distance,
                             PYCV_ExtendBorder mode, int include_dfs)
{
    int ii, jj, cc_size = 1, cum_sum, mid, con;
    PYCV_CoordinatesIterator iterator;
    npy_intp ravel, cc_dims[NPY_MAXDIMS], cc_strides[NPY_MAXDIMS];

    self->ndim = (int)PyArray_NDIM(input);
    self->neighbors_size = 1;
    self->n_dfs = 0;

    con = self->ndim - 1;

    for (ii = 0; ii < self->ndim; ii++) {
        *(self->neighbors_dims + ii) = (2 * *(min_distance + ii)) + 1;
        self->neighbors_size *= (int)(*(self->neighbors_dims + ii));
        *(self->input_dims + ii) = (npy_intp)PyArray_DIM(input, ii);
        if (include_dfs) {
            *(cc_dims + ii) = 3 > *(self->neighbors_dims + ii) ? *(self->neighbors_dims + ii) : 3;
            cc_size *= (int)*(cc_dims + ii);
            con = *(cc_dims + ii) < 3 ? con - 1: con + 1;
        }
    }

    con = con > 0 ? con : 1;

    *(self->norm_strides + self->ndim - 1) = 1;
    *(cc_strides + self->ndim - 1) = 1;
    for (ii = self->ndim - 2; ii >= 0; ii--) {
        *(self->norm_strides + ii) = *(self->norm_strides + ii + 1) * *(self->input_dims + ii + 1);
        *(cc_strides + ii) = *(cc_strides + ii + 1) * *(self->neighbors_dims + ii + 1);
    }

    if (!PYCV_InitNeighborhoodOffsets(input, self->neighbors_dims, NULL, NULL,
                                      &(self->offsets), NULL, &(self->flag), mode)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PYCV_InitNeighborhoodOffsets \n");
        self->neighbors_size = 0;
        return 0;
    }

    self->offsets_norm = malloc(self->neighbors_size * sizeof(npy_intp));
    if (!self->offsets_norm) {
        free(self->offsets);
        self->neighbors_size = 0;
        PyErr_NoMemory();
        return 0;
    }

    PYCV_CoordinatesIteratorInit((npy_intp)(self->ndim), self->neighbors_dims, &iterator);

    for (ii = 0; ii < self->neighbors_size; ii++) {
        ravel = 0;
        for (jj = 0; jj < self->ndim; jj++) {
            ravel += (*(iterator.coordinates + jj) - (*(self->neighbors_dims + jj) / 2)) * *(self->norm_strides + jj);
        }
        *(self->offsets_norm + ii) = ravel;
        PYCV_COORDINATES_ITERATOR_NEXT(iterator);
    }

    if (!include_dfs) {
        self->dfs_ptr = NULL;
        return 1;
    }

    PYCV_CoordinatesIteratorInit((npy_intp)(self->ndim), cc_dims, &iterator);

    self->dfs_ptr = malloc(cc_size * sizeof(int));

    if (!self->dfs_ptr) {
        PyErr_NoMemory();
        return 0;
    }

    mid = self->neighbors_size / 2;

    for (ii = 0; ii < cc_size; ii++) {
        cum_sum = 0;
        ravel = 0;
        for (jj = 0; jj < self->ndim; jj++) {
            ravel += (*(iterator.coordinates + jj) - (*(cc_dims + jj) / 2)) * *(cc_strides + jj);
            cum_sum += abs((int)(*(iterator.coordinates + jj)) - 1);
        }
        if (ravel && cum_sum <= (self->ndim - 1)) {
            *(self->dfs_ptr + self->n_dfs) = mid + (int)ravel;
            self->n_dfs++;
        }
        PYCV_COORDINATES_ITERATOR_NEXT(iterator);
    }

    return 1;
}

static void Neighborhood_free(Neighborhood *self)
{
    if (self->neighbors_size) {
        free(self->offsets);
        free(self->offsets_norm);
    }
    if (self->n_dfs) {
        free(self->dfs_ptr);
    }
}

// #####################################################################################################################

static int peaks_prepare_heap(Heap *heap, Neighborhood *nn, PyArrayObject *input, double threshold, double c_val)
{
    NeighborhoodIterator iterator;
    int array_size, ii, jj, neighbor_shift = 0, inp_shift = 0, out_shift = 0, is_max;
    npy_intp *n1 = NULL;
    char *pi = NULL;
    double v;

    array_size = (int)PyArray_SIZE(input);
    pi = (void *)PyArray_DATA(input);

    PYCV_NeighborhoodIteratorInit(input, nn->neighbors_dims, NULL, (npy_intp)(nn->neighbors_size), &iterator);

    if (!heap_init(heap, array_size)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: heap_init.\n");
        return 0;
    }

    n1 = nn->offsets;

    for (ii = 0; ii < array_size; ii++) {
        PYCV_GET_VALUE(iterator.numtype, double, pi, v);

        if (v > threshold) {
            IS_LOCAL_MAX(iterator.numtype, pi, nn->neighbors_size, n1, nn->flag, c_val, is_max);
            if (is_max) {
                HeapItem new_item;
                new_item.priority = -v;
                new_item.index = ii;
                new_item.out_shift = out_shift;
                new_item.inp_shift = inp_shift;
                new_item.neighbor_shift = neighbor_shift;
                if (!heap_push(heap, new_item)) {
                    PyErr_SetString(PyExc_RuntimeError, "Error: heap_push.\n");
                    return 0;
                }
            }
        }
        out_shift += p_type_stride;
        for (jj = (int)iterator.nd_m1; jj >= 0; jj--) {
            if (*(iterator.coordinates + jj) < *(iterator.dims_m1 + jj)) {
                if ((*(iterator.coordinates + jj) < *(iterator.boundary_low + jj)) ||
                    (*(iterator.coordinates + jj) >= *(iterator.boundary_high + jj))) {
                    neighbor_shift += (int)*(iterator.nn_strides + jj);
                    n1 += *(iterator.nn_strides + jj);
                }
                *(iterator.coordinates + jj) += 1;
                pi += *(iterator.strides + jj);
                inp_shift += (int)*(iterator.strides + jj);
                break;
            } else {
                *(iterator.coordinates + jj) = 0;
                neighbor_shift -= (int)*(iterator.nn_strides_back + jj);
                n1 -= *(iterator.nn_strides_back + jj);
                pi -= *(iterator.strides_back + jj);
                inp_shift -= (int)*(iterator.strides_back + jj);
            }
        }
    }
    return 1;
}

// #####################################################################################################################

static int peaks_nonmaximum_suppression(Heap *heap, Neighborhood *nn, int size, char *output)
{
    HeapItem heap_top, *map_item = NULL;
    Heap map = HEAP_NEW;
    char *ptr_i = NULL, *ptr_o = NULL;
    npy_intp *n1 = NULL, *n2 = nn->offsets_norm, *n1_dfs = NULL;
    int ii, n_index, *lut = NULL, *stack = NULL, stack_n, stack_i, n_peaks = 0;

    lut = calloc(size, sizeof(int));
    stack = malloc(heap->n * sizeof(int));
    if (!lut || !stack) {
        PyErr_NoMemory();
        return 0;
    }

    if (!heap_init(&map, heap->n)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: heap_init.\n");
        free(lut);
        free(stack);
        return 0;
    }

    for (ii = 0; ii < heap->n; ii++) {
        map_item = heap->heap + ii;
        *(map.heap + map.n) = *map_item;
        *(lut + map_item->index) = map.n + 1;
        map.n++;
    }


    while (heap->n) {
        heap_top = heap_pop(heap);

        if (!*(lut + heap_top.index)) {
            continue;
        }

        stack_i = 0;
        stack_n = 1;
        *stack = *(lut + heap_top.index) - 1;
        *(lut + heap_top.index) = 0;

        ptr_o = output + heap_top.out_shift;
        n_peaks++;
        *(p_type *)ptr_o = (p_type)n_peaks;

        while (stack_i < stack_n) {
            map_item = map.heap + *(stack + stack_i);
            n1 = nn->offsets + map_item->neighbor_shift;

            for (ii = 0; ii < nn->n_dfs; ii++) {
                n1_dfs = n1 + *(nn->dfs_ptr + ii);
                n_index = map_item->index + (int)*(n2 + *(nn->dfs_ptr + ii));

                if (*n1_dfs == nn->flag || !*(lut + n_index)) {
                    continue;
                }

                ptr_o = output + (map.heap + *(lut + n_index) - 1)->out_shift;
                *(p_type *)ptr_o = (p_type)n_peaks;

                *(stack + stack_n) = *(lut + n_index) - 1;
                stack_n++;

                *(lut + n_index) = 0;
            }
            stack_i++;
        }

        n1 = nn->offsets + heap_top.neighbor_shift;
        for (ii = 0; ii < nn->neighbors_size; ii++) {
            if (*(n1 + ii) == nn->flag) {
                continue;
            }
            *(lut + heap_top.index + *(n2 + ii)) = 0;
        }
    }

    free(lut);
    free(stack);
    heap_free(&map);
    return 1;
}

// #####################################################################################################################

int PYCV_peaks_nonmaximum_suppression(PyArrayObject *input,
                                      npy_intp *min_distance,
                                      double threshold,
                                      PYCV_ExtendBorder mode,
                                      double c_val,
                                      PyArrayObject **output)
{
    Neighborhood nn;
    Heap heap = HEAP_NEW;
    char *ptr_o = NULL;
    int size;

    nn.neighbors_size = 0;
    nn.n_dfs = 0;

    size = (int)PyArray_SIZE(input);

    if (!Neighborhood_init(&nn, input, min_distance, mode, 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: Neighborhood_init.\n");
        goto exit;
    }

    if (!peaks_prepare_heap(&heap, &nn, input, threshold, c_val)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: peaks_prepare_heap.\n");
        goto exit;
    }

    *output = (PyArrayObject *)PyArray_ZEROS(nn.ndim, nn.input_dims, p_itype, 0);
    if (!*output) {
        PyErr_SetString(PyExc_RuntimeError, "Error: PyArray_ZEROS.\n");
        goto exit;
    }

    ptr_o = (void *)PyArray_DATA(*output);

    if (!peaks_nonmaximum_suppression(&heap, &nn, size, ptr_o)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: peaks_nonmaximum_suppression.\n");
        goto exit;
    }

    exit:
        Neighborhood_free(&nn);
        heap_free(&heap);
        return PyErr_Occurred() ? 0 : 1;
}

// #####################################################################################################################











