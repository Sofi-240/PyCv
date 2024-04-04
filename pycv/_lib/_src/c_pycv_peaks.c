#include "c_pycv_base.h"
#include "c_pycv_peaks.h"

// #####################################################################################################################

#define p_type npy_ulonglong

#define p_itype NPY_UINT64

#define p_type_stride (int)NPY_SIZEOF_ULONGLONG

// #####################################################################################################################
// *************************************  Heap  ************************************************************************

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

static int heap_build_from_array(Heap *self, PyArrayObject *input, double threshold, int *neigh_shape, int neigh_size)
{
    int array_size, ii, jj, neighbor_shift = 0, inp_shift = 0, out_shift = 0;
    NeighborhoodIterator iter;
    char *pi = NULL;
    double v;

    array_size = (int)PyArray_SIZE(input);

    PYCV_NeighborhoodIteratorInit(input, neigh_shape, NULL, neigh_size, &iter);

    if (!heap_init(self, array_size)) {
        PyErr_SetString(PyExc_RuntimeError, "Error: heap_init.\n");
        return 0;
    }

    pi = (void *)PyArray_DATA(input);

    for (ii = 0; ii < array_size; ii++) {
        out_shift += p_type_stride;
        PYCV_GET_VALUE(iter.numtype, double, pi, v);

        if (v >= threshold) {
            HeapItem new_item;
            new_item.priority = -v;
            new_item.index = ii;
            new_item.out_shift = out_shift;
            new_item.inp_shift = inp_shift;
            new_item.neighbor_shift = neighbor_shift;
            if (!heap_push(self, new_item)) {
                PyErr_SetString(PyExc_RuntimeError, "Error: heap_push.\n");
                return 0;
            }
        }
        for (jj = (int)iter.nd_m1; jj >= 0; jj--) {
            if (*(iter.coordinates + jj) < *(iter.dims_m1 + jj)) {
                if ((*(iter.coordinates + jj) < *(iter.boundary_low + jj)) ||
                    (*(iter.coordinates + jj) >= *(iter.boundary_high + jj))) {
                    neighbor_shift += (int)*(iter.nn_strides + jj);
                }
                *(iter.coordinates + jj) += 1;
                pi += *(iter.strides + jj);
                inp_shift += (int)*(iter.strides + jj);
                break;
            } else {
                *(iter.coordinates + jj) = 0;
                neighbor_shift -= (int)*(iter.nn_strides_back + jj);
                pi -= *(iter.strides_back + jj);
                inp_shift -= (int)*(iter.strides_back + jj);
            }
        }
    }

    return 1;
}

// #####################################################################################################################

















