#include "c_pycv_pyr_utils.h"

// ######################################### GAUSSIAN ##################################################################

int gaussian_kernel(double sigma, int ndim, int radius, double **kernel)
{
    int *coordinates = NULL, ii, jj, size;
    double *h = NULL, h_max = 0, p, v, h_sum = 0, epsilon = GAUSSIAN_EPSILON;

    UTILS_MALLOC(ndim, sizeof(int), coordinates);
    if (!ndim)
        return 0;

    size = GAUSSIAN_KERNEL_SIZE(radius, ndim);

    UTILS_MALLOC(size, sizeof(double), *kernel);
    if (!size) {
        free(coordinates);
        return 0;
    }

    h = *kernel;
    if (size == 1) {
        *h = 0;
        return 1;
    }
    for (ii = 0; ii < ndim; ii++) {
        *(coordinates + ii) = -radius;
    }

    p = 1 / (2 * (sigma * sigma));

    for (ii = 0; ii < size; ii++) {
        v = 0;
        for (jj = 0; jj < ndim; jj++) {
            v += (double)(*(coordinates + jj) * *(coordinates + jj));
        }
        v = exp(-v * p);
        h_max = v > h_max ? v : h_max;
        *(h + ii) = v;

        for (jj = ndim - 1; jj >= 0; jj--) {
            if (*(coordinates + jj) < radius) {
                *(coordinates + jj) += 1;
                break;
            } else {
                *(coordinates + jj) = -radius;
            }
        }
    }
    h_max *= epsilon;

    for (ii = 0; ii < size; ii++) {
        if (*(h + ii) < h_max) {
            *(h + ii) = 0;
        }
        h_sum += *(h + ii);
    }

    if (h_sum != 0) {
        for (ii = 0; ii < size; ii++) {
            *(h + ii) /= h_sum;
        }
    }
    free(coordinates);
    return 1;
}

// ######################################### OFFSETS ###################################################################

int offsets_fit_coordinate(int coordinate, int dim, int flag, int mode)
{
    int out = coordinate, dim2;
    if (coordinate >= 0 && coordinate < dim)
        return out;
    switch ((PyrExtend)mode) {
        case EXTEND_CONSTANT:
            out = flag;
            break;
        case EXTEND_EDGE:
            out = coordinate < 0 ? 0 : dim - 1;
            break;
        case EXTEND_WRAP:
            out = coordinate;
            if (coordinate < 0) {
                out += dim * (int)(-out / dim);
                if (out < 0) {
                    out += dim;
                }
            } else {
                out -= dim * (int)(out / dim);
            }
            break;
        case EXTEND_SYMMETRIC:
            dim2 = 2 * dim;
            out = coordinate;
            if (out < 0) {
                if (out < -dim2) {
                    out += dim2 * (int)(-out / dim2);
                }
                if (out < -dim) {
                    out += dim2;
                } else {
                    out = -out - 1;
                }
            } else {
                out -= dim2 * (int)(out / dim2);
                if (out >= dim) {
                    out = dim2 - out - 1;
                }
            }
            break;
        case EXTEND_REFLECT:
            dim2 = 2 * dim - 2;
            out = coordinate;
            if (out < 0) {
                out += dim2 * (int)(-out / dim2);
                if (out <= 1 - dim) {
                    out += dim2;
                } else {
                    out = -out;
                }
            } else {
                out -= dim2 * (int)(out / dim2);
                if (out >= dim) {
                    out = dim2 - out;
                }
            }
            break;
        default:
            out = -1; // Invalid mode
    }
    return out;
}

void offsets_init(int input_dim, int input_stride, int offset_dim, int mode, int *flag, int *offsets)
{
    int center, ii, jj, pp, pos = 0, *ofs = NULL;
    ofs = offsets;

    center = offset_dim / 2;
    *flag = input_dim * input_stride;

    for (ii = 0; ii < offset_dim; ii++) {
        for (jj = 0; jj < offset_dim; jj++) {
            pp = jj - center + pos;
            pp = offsets_fit_coordinate(pp, input_dim, *flag, mode);
            *ofs++ = pp == *flag ? pp : (pp - pos) * input_stride;
        }
        if (pos == center) {
            pos += input_dim - offset_dim + 1;
            if (pos <= center) {
                pos = center + 1;
            }
        } else {
            pos += 1;
        }
    }
}

// *********************************************************************************************************************

void Offsets1D_adapt_dim(Offsets1D *self, int input_dim)
{
    int diff = (self->stride - self->dim) / 2;
    self->low = self->dim / 2;
    self->high = input_dim - self->dim + self->low;

    self->stride_back = (self->dim - 1) * self->stride;

    self->init_pos = diff;
    self->nn_stride = (diff + 1) * self->stride;
}

int Offsets1D_new(Offsets1D **self, int input_dim, int input_stride, int dim, int mode)
{
    *self = (Offsets1D *)malloc(sizeof(Offsets1D));
    if (!*self)
        return 0;
    int n_offsets = dim * dim;
    UTILS_MALLOC(n_offsets, sizeof(int), (*self)->offsets);
    if (!n_offsets) {
        free(*self);
        *self = NULL;
        return 0;
    }
    offsets_init(input_dim, input_stride, dim, mode, &((*self)->flag), (*self)->offsets);
    (*self)->dim = dim;
    (*self)->stride = dim;
    Offsets1D_adapt_dim(*self, input_dim);
    return 1;
}

void Offsets1D_free(Offsets1D **self)
{
    if (*self) {
        free((*self)->offsets);
        free(*self);
        *self = NULL;
    }
}

int Offsets1D_update_offsets(Offsets1D **self, int input_dim, int input_stride, int dim, int mode)
{
    if ((*self) == NULL) {
        return Offsets1D_new(self, input_dim, input_stride, dim, mode);
    } else if (dim > (*self)->stride) {
        int n_offsets = dim * dim;
        UTILS_REALLOC(n_offsets, sizeof(int), (*self)->offsets);
        if (!n_offsets) {
            free(*self);
            *self = NULL;
            return 0;
        }
        offsets_init(input_dim, input_stride, dim, mode, &((*self)->flag), (*self)->offsets);
        (*self)->stride = dim;
    }
    (*self)->dim = dim;
    Offsets1D_adapt_dim(*self, input_dim);
    return 1;
}

void Offsets1D_update_input_stride(Offsets1D *self, int input_stride, int prev_stride)
{
    for (int ii = 0; ii < self->stride * self->stride; ii++) {
        *(self->offsets + ii) /= prev_stride;
        *(self->offsets + ii) *= input_stride;
    }
    self->flag /= prev_stride;
    self->flag *= input_stride;
}

void Offsets1D_print(Offsets1D **self)
{
    if (*self != NULL) {
        printf("dim=%d, stride=%d, stride_back=%d\n", (*self)->dim, (*self)->stride, (*self)->stride_back);
        printf("low=%d, high=%d, init_pos=%d, flag=%d\n", (*self)->low, (*self)->high, (*self)->init_pos, (*self)->flag);
        int *ofs = (*self)->offsets + (*self)->init_pos;
        printf("[\n");
        for (int ii = 0; ii < (*self)->dim; ii++) {
            UTILS_PRINT_LIST(ofs, (*self)->dim, 0);
            if (ii == (*self)->low - 1 || ii == (*self)->low) {
                ofs += (*self)->nn_stride;
            } else {
                ofs += (*self)->stride;
            }
        }
        printf("]\n\n");
    }
}

// ######################################## Iterator ###################################################################

// **************************************** Iterator 1D ****************************************************************

int Iterator1D_new(Iterator1D **self, int axis, int dim, int stride)
{
    *self = (Iterator1D *)malloc(sizeof(Iterator1D));
    if (!*self)
        return 0;
    (*self)->axis = axis;
    (*self)->coordinate = 0;
    (*self)->dim = dim;
    (*self)->stride = stride;
    (*self)->stride_back = stride * (dim - 1);
    (*self)->offsets = NULL;
    (*self)->next = NULL;
    return 1;
}

void Iterator1D_free(Iterator1D **self)
{
    if (*self) {
        if ((*self)->next != NULL)
            Iterator1D_free(&(*self)->next);
        Offsets1D_free(&(*self)->offsets);
        free(*self);
        *self = NULL;
    }
}

void Iterator1D_update_stride(Iterator1D *self, int stride)
{
    if (stride != self->stride) {
        if (self->offsets != NULL) {
            Offsets1D_update_input_stride(self->offsets, stride, self->stride);
        }
        self->stride = stride;
        self->stride_back = stride * (self->dim - 1);
    }
}

void Iterator1D_update_dim(Iterator1D *self, int dim, int stride, int mode)
{
    if (self->dim != dim) {
        self->dim = dim;
        self->stride = stride;
        self->stride_back = stride * (self->dim - 1);
        if (self->offsets != NULL)
            offsets_init(self->dim, self->stride, self->offsets->stride, mode, &(self->offsets->flag), self->offsets->offsets);
    } else {
        Iterator1D_update_stride(self, stride);
    }
    if (self->offsets != NULL)
        Offsets1D_adapt_dim(self->offsets, dim);
}

int Iterator1D_update_offset_dim(Iterator1D *self, int dim, int mode)
{return Offsets1D_update_offsets(&(self->offsets), self->dim, self->stride, (self->dim > dim ? dim : self->dim), mode);}

void Iterator1D_print(Iterator1D **self)
{
    if (*self != NULL) {
        printf("axis=%d, dim=%d, stride=%d, stride_back=%d, coordinate=%d\n",
               (*self)->axis, (*self)->dim, (*self)->stride, (*self)->stride_back, (*self)->coordinate);
        Offsets1D_print(&((*self)->offsets));
        printf("--------------------------------------------\n");
        Iterator1D_print(&(*self)->next);
    }
}

// **************************************** Iterator ND ****************************************************************

int IteratorND_copy_to(Iterator1D **self, Iterator1D **to)
{
    if (*self != NULL) {
        Iterator1D *from = *self, *node = NULL;
        if (!Iterator1D_new(to, from->axis, from->dim, from->stride))
            return 0;
        from = from->next;
        node = *to;
        while (from != NULL) {
            if (!Iterator1D_new(&(node->next), from->axis, from->dim, from->stride)) {
                Iterator1D_free(to);
                return 0;
            }
            node = node->next;
            from = from->next;
        }
    }
    return 1;
}

void IteratorND_next_axis(Iterator1D **head)
{
    if (*head != NULL) {
        Iterator1D *prev = *head;
        UTILS_NODE_GOTO_LAST_M1(prev);
        if (prev->next != NULL) {
            Iterator1D *temp = prev->next;
            prev->next = NULL;
            temp->next = *head;
            *head = temp;
        }
    }
}

void IteratorND_reset_axis(Iterator1D **head)
{
    if (*head != NULL) {
        Iterator1D *tail = *head;
        while (tail->axis)
            tail = tail->next;
        if (tail->next != NULL) {
            Iterator1D *next = tail->next, *new_head = tail->next;
            tail->next = NULL;
            while (next->next != NULL)
                next = next->next;
            next->next = *head;
            *head = new_head;
        }
    }
}

// ####################################### Kernels Mem #################################################################

int GaussianMem_new(GaussianMem **self, double sigma)
{
    int radius;
    *self = (GaussianMem *)malloc(sizeof(GaussianMem));
    if (!*self)
        return 0;
    (*self)->next = NULL;

    if (sigma == 0)
        radius = 0;
    else
        radius = GAUSSIAN_RADIUS_FROM_SIGMA(sigma, GAUSSIAN_DEFAULT_TRUNCATE);

    (*self)->sigma = sigma;
    (*self)->len = GAUSSIAN_KERNEL_SIZE(radius, 1);

    if (!gaussian_kernel(sigma, 1, radius, &((*self)->h))){
        free(*self);
        return 0;
    }
    (*self)->entries = 1;
    return 1;
}

void GaussianMem_free(GaussianMem **self)
{
    if (*self) {
        if ((*self)->next != NULL)
            GaussianMem_free(&(*self)->next);
        if ((*self)->len)
            free((*self)->h);
        free(*self);
        *self = NULL;
    }
}

void GaussianMem_push(GaussianMem **head, double sigma, GaussianMem **out)
{
    GaussianMem *kernel = *head, *prev = NULL;

    *out = NULL;

    if ((*head) == NULL) {
        GaussianMem_new(head, sigma);
        *out = *head;
    } else {
        while (kernel != NULL && kernel->sigma < sigma) {
            prev = kernel;
            kernel = kernel->next;
        }

        if (kernel != NULL && kernel->sigma == sigma) {
            kernel->entries += 1;
            *out = kernel;
        } else if (prev == NULL) {
            GaussianMem *node = NULL;
            if (GaussianMem_new(&node, sigma)) {
                node->next = *head;
                *head = node;
                *out = *head;
            }
        } else {
            prev->next = NULL;
            if (GaussianMem_new(&(prev->next), sigma)) {
                prev = prev->next;
                prev->next = kernel;
                *out = prev;
            } else {
                prev->next = kernel;
            }
        }
    }
}

void GaussianMem_pop(GaussianMem **head, double sigma)
{
    if ((*head)!= NULL) {
        GaussianMem *kernel = *head, *prev = NULL;

        while (kernel != NULL && kernel->sigma < sigma) {
            prev = kernel;
            kernel = kernel->next;
        }

        if (kernel != NULL && kernel->sigma == sigma) {
            kernel->entries -= 1;
            if (!kernel->entries) {
                if (prev == NULL)
                    *head = kernel->next;
                else
                    prev->next = kernel->next;
                kernel->next = NULL;
                GaussianMem_free(&kernel);
            }
        }
    }
}

void GaussianMem_print(GaussianMem **head)
{
    if (*head != NULL) {
        printf("sigma=%.2f, len=%d, entries=%d\n", (*head)->sigma, (*head)->len, (*head)->entries);
        UTILS_PRINT_LIST((*head)->h, (*head)->len, 1);
        printf("\n");
        GaussianMem_print(&(*head)->next);
    }
}

// ####################################### Gaussian 1D #################################################################

int Gaussian1D_new(Gaussian1D **self, int axis, double sigma, GaussianMem **mem)
{
    *self = (Gaussian1D *)malloc(sizeof(Gaussian1D));
    if (!*self)
        return 0;
    (*self)->axis = axis;
    (*self)->next = NULL;
    GaussianMem_push(mem, sigma, &(*self)->kernel);
    if ((*self)->kernel == NULL) {
        free(*self);
        *self = NULL;
        return 0;
    }
    return 1;
}

void Gaussian1D_free(Gaussian1D **self, GaussianMem **mem)
{
    if (*self != NULL) {
        if ((*self)->next)
            Gaussian1D_free(&(*self)->next, mem);
        double sigma = (*self)->kernel->sigma;
        GaussianMem_pop(mem, sigma);
        (*self)->kernel = NULL;
        free(*self);
        *self = NULL;
    }
}

void Gaussian1D_print(Gaussian1D **head)
{
    if (*head != NULL) {
        GaussianMem *kernel = (*head)->kernel;
        printf("sigma=%.2f, len=%d, entries=%d\n", kernel->sigma, kernel->len, kernel->entries);
        UTILS_PRINT_LIST(kernel->h, kernel->len, 1);
        printf("\n");
        Gaussian1D_print(&(*head)->next);
    }
}

int Gaussian1D_build_scalespace(Gaussian1D **root, GaussianMem **mem, int ndim, int nscales, double *scalespace)
{
    int size = ndim * nscales;
    if (!Gaussian1D_new(root, 0, *scalespace, mem))
        return 0;
    Gaussian1D *node = *root;
    for (int ii = 1; ii < size; ii++) {
        int axis = node->axis == ndim - 1 ? 0 : node->axis + 1;
        if (!Gaussian1D_new(&(node->next), axis, *(scalespace + ii), mem)) {
            Gaussian1D_free(root, mem);
            return 0;
        }
        node = node->next;
    }
    return 1;
}

// ######################################## Rescale 1D #################################################################

int Rescale1D_new(Rescale1D **self, int axis, double factor, int order, GaussianMem **mem)
{
    *self = (Rescale1D *)malloc(sizeof(Rescale1D));
    if (!*self)
        return 0;
    (*self)->axis = axis;
    (*self)->factor = factor;
    (*self)->kernel = NULL;
    (*self)->next = NULL;
    if (order > 0 && factor < 1.0) {
        double sigma = ((1 / factor) - 1) / 2;
        if (sigma < 0)
            sigma = 0.0;
        GaussianMem_push(mem, sigma, &(*self)->kernel);
        if ((*self)->kernel == NULL) {
            free(*self);
            *self = NULL;
            return 0;
        }
    }
    return 1;
}

void Rescale1D_free(Rescale1D **self, GaussianMem **mem)
{
    if (*self != NULL) {
        Rescale1D_free(&(*self)->next, mem);
        if ((*self)->kernel != NULL) {
            double sigma = (*self)->kernel->sigma;
            GaussianMem_pop(mem, sigma);
            (*self)->kernel = NULL;
        }
        free(*self);
        *self = NULL;
    }
}

int Rescale1D_update_order(Rescale1D **self, int order, GaussianMem **mem)
{
    double sigma;
    if (!order && (*self)->kernel != NULL) {
        sigma = (*self)->kernel->sigma;
        GaussianMem_pop(mem, sigma);
        (*self)->kernel = NULL;
    } else if (order && (*self)->kernel == NULL) {
        sigma = ((1 / (*self)->factor) - 1) / 2;
        if (sigma < 0)
            sigma = 0.0;
        GaussianMem_push(mem, sigma, &(*self)->kernel);
        return (*self)->kernel != NULL;
    }
    return 1;
}

// *********************************************************************************************************************

int RescaleIterator_new(RescaleIterator **self, Iterator1D **base, int extend_mode)
{
    *self = (RescaleIterator *)malloc(sizeof(RescaleIterator));
    if (!*self)
        return 0;

    if (!IteratorND_copy_to(base, &((*self)->input)) || !IteratorND_copy_to(base, &((*self)->output))) {
        Iterator1D_free(&((*self)->input));
        Iterator1D_free(&((*self)->output));
        free(*self);
        *self = NULL;
        return 0;
    }
    (*self)->extend_mode = extend_mode;
    return 1;
}

void RescaleIterator_free(RescaleIterator **self)
{
    if (*self != NULL) {
        Iterator1D_free(&((*self)->input));
        Iterator1D_free(&((*self)->output));
        free(*self);
        *self = NULL;
    }
}

void RescaleIterator_update_output(RescaleIterator *self, int dim)
{
    Iterator1D *node = self->output;
    UTILS_NODE_GOTO_LAST(node);

    int stride = node->stride;
    int last = node->axis;
    Iterator1D_update_dim(node, dim, stride, self->extend_mode);

    stride *= node->dim;
    node = self->output;

    while (node->axis < last) {
        Iterator1D_update_stride(node, stride);
        stride *= node->dim;
        node = node->next;
    }
}

void RescaleIterator_update_input(RescaleIterator *self)
{
    Iterator1D *n1 = NULL, *n2 = NULL;

    IteratorND_next_axis(&(self->input));
    IteratorND_next_axis(&(self->output));

    n1 = self->input;
    n2 = self->output;

    while (n1 != NULL && n1->axis <= 0) {
        Iterator1D_update_dim(n1, n2->dim, n2->stride, self->extend_mode);
        n1 = n1->next;
        n2 = n2->next;
    }
}

void RescaleIterator_update_base(RescaleIterator *self, Iterator1D *base)
{
    Iterator1D *node = self->output, *nb = base;
    while (node != NULL) {
        Iterator1D_update_dim(nb, node->dim, node->stride, self->extend_mode);
        node = node->next;
        nb = nb->next;
    }
}

// ######################################## Layer ######################################################################

void Layer_free(Layer **self)
{
    if (*self != NULL) {
        Gaussian1D_free(&((*self)->scalespace), &((*self)->mem));
        Rescale1D_free(&((*self)->factors), &((*self)->mem));
        Iterator1D_free(&((*self)->iterator));
        free(*self);
        *self = NULL;
    }
}

int Layer_new(Layer **self, int ndim, int itemsize, int numtype)
{
    *self = (Layer *)malloc(sizeof(Layer));
    if (!*self)
        return 0;
    (*self)->ndim = ndim;
    (*self)->nscales = 0;
    (*self)->order = 0;
    (*self)->numtype = numtype;
    (*self)->itemsize = itemsize;
    (*self)->extend_mode = 6;
    (*self)->cval = 0.0;

    (*self)->mem = NULL;
    (*self)->scalespace = NULL;
    (*self)->factors = NULL;
    (*self)->iterator = NULL;
    return 1;
}

// ********************************************* setter ****************************************************************

void Layer_set_itemsize(Layer *self, int itemsize, int numtype)
{
    self->numtype = numtype;
    if (self->itemsize != itemsize) {
        self->itemsize = itemsize;
        Iterator1D *node = self->iterator;
        while (node != NULL) {
            Iterator1D_update_stride(node, itemsize);
            itemsize *= node->dim;
            node = node->next;
        }
    }
}

void Layer_set_extend_mode(Layer *self, int extend_mode)
{
    if (self->extend_mode != extend_mode) {
        self->extend_mode = extend_mode;
        Iterator1D *node = self->iterator;
        while (node != NULL) {
            if (node->offsets != NULL) {
                Offsets1D *offsets = node->offsets;
                offsets_init(node->dim, node->stride, offsets->dim, extend_mode, &(offsets->flag), offsets->offsets);
            }
            node = node->next;
        }
    }
}

int Layer_set_scalespace(Layer *self, int nscales, double *scalespace)
{
    if (!self->nscales) {
        if (!Gaussian1D_build_scalespace(&(self->scalespace), &(self->mem), self->ndim, nscales, scalespace))
            return 0;
        self->nscales = nscales;
    } else {
        Gaussian1D *tail = self->scalespace;
        while (tail->next != NULL)
            tail = tail->next;
        if (!Gaussian1D_build_scalespace(&(tail->next), &(self->mem), self->ndim, nscales, scalespace))
            return 0;
        while (self->nscales != 0) {
            Gaussian1D *temp = self->scalespace;
            self->scalespace = temp->next;
            temp->next = NULL;
            Gaussian1D_free(&temp, &(self->mem));
            self->nscales -= 1;
        }
        self->nscales = nscales;
    }
    return 1;
}

int Layer_set_factors(Layer *self, double *factors)
{
    int ndim = self->ndim;
    if (self->factors != NULL) {
        Rescale1D_free(&(self->factors), &(self->mem));
    }
    while (ndim--) {
        Rescale1D *node = NULL;
        if (!Rescale1D_new(&node, ndim, *(factors + ndim), self->order, &(self->mem))) {
            Rescale1D_free(&(self->factors), &(self->mem));
            return 0;
        }
        node->next = self->factors;
        self->factors = node;
    }
    return 1;
}

int Layer_set_order(Layer *self, int order)
{
    Rescale1D *factors = self->factors;

    while (factors != NULL) {
        if (!Rescale1D_update_order(&factors, order, &(self->mem)))
            return 0;
    }
    self->order = order;
    return 1;
}

int Layer_set_input_dim(Layer *self, int *dims)
{
    int stride = self->itemsize;
    Iterator1D *node = NULL;

    if (self->iterator == NULL) {
        if (!Iterator1D_new(&(self->iterator), self->ndim - 1, *(dims + self->ndim - 1), stride))
            return 0;
        node = self->iterator;
        stride *= node->dim;
        for (int ii = self->ndim - 2; ii >= 0; ii--) {
            if (!Iterator1D_new(&(node->next), ii, *(dims + ii), stride)) {
                Iterator1D_free(&(self->iterator));
                return 0;
            }
            stride *= node->dim;
            node = node->next;
        }
    } else {
        node = self->iterator;
        while (node != NULL) {
            Iterator1D_update_dim(node, *(dims + node->axis), stride, self->extend_mode);
            stride *= node->dim;
            node = node->next;
        }
    }

    return 1;
}

int Layer_update_offsets(Layer *self)
{
    if (self->iterator == NULL)
        return 0;
    int *dims = NULL, ndim = self->ndim;
    Iterator1D *node = self->iterator;
    Gaussian1D *g_node = self->scalespace;
    Rescale1D *r_node = self->factors;

    UTILS_MALLOC(ndim, sizeof(int), dims);
    if (!ndim)
        return 0;

    while (node != NULL) {
        if (node->offsets != NULL)
             *(dims + node->axis) = node->offsets->stride;
        else
            *(dims + node->axis) = 5;
        node = node->next;
    }

    while (g_node != NULL) {
        if (*(dims + g_node->axis) < g_node->kernel->len)
            *(dims + g_node->axis) = g_node->kernel->len;
        g_node = g_node->next;
    }

    while (r_node != NULL) {
        if (r_node->kernel != NULL && *(dims + r_node->axis) < r_node->kernel->len)
            *(dims + r_node->axis) = r_node->kernel->len;
        r_node = r_node->next;
    }

    node = self->iterator;
    while (node != NULL) {
        if (!Iterator1D_update_offset_dim(node, *(dims + node->axis), self->extend_mode)) {
            free(dims);
            return 0;
        }
        node = node->next;
    }

    free(dims);
    return 1;
}

void Layer_reduce(Layer *self)
{
    if (self->iterator != NULL && self->factors != NULL) {
        Iterator1D *iter_node = self->iterator;
        Rescale1D *factors = NULL;

        UTILS_NODE_REVERS(&(self->factors), Rescale1D);
        factors = self->factors;
        int stride = iter_node->stride;

        while (iter_node != NULL) {
            int dim = (int)((double)iter_node->dim * factors->factor + 0.5);
            Iterator1D_update_dim(iter_node, dim, stride, self->extend_mode);
            stride *= iter_node->dim;
            iter_node = iter_node->next;
            factors = factors->next;
        }
        UTILS_NODE_REVERS(&(self->factors), Rescale1D);
    }
}

// #####################################################################################################################
