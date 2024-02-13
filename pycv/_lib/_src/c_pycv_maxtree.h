#ifndef C_PYCV_MAXTREE_H
#define C_PYCV_MAXTREE_H

// #####################################################################################################################

int PYCV_build_max_tree(PyArrayObject *input,
                        PyArrayObject *traverser,
                        PyArrayObject *parent,
                        npy_intp connectivity);


int PYCV_max_tree_compute_area(PyArrayObject *input,
                               PyArrayObject *output,
                               npy_intp connectivity,
                               PyArrayObject *traverser,
                               PyArrayObject *parent);


int PYCV_max_tree_filter(PyArrayObject *input,
                         npy_double threshold,
                         PyArrayObject *values_map,
                         PyArrayObject *output,
                         npy_intp connectivity,
                         PyArrayObject *traverser,
                         PyArrayObject *parent);

// #####################################################################################################################

#endif