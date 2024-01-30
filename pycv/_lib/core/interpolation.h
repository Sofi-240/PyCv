#ifndef INTERPOLATION_TMP_H
#define INTERPOLATION_TMP_H

int ops_resize(PyArrayObject *input,
               PyArrayObject *output,
               npy_intp order,
               int grid_mode,
               BordersMode mode,
               double constant_value);

int ops_geometric_transform(PyArrayObject *matrix,
                            PyArrayObject *input,
                            PyArrayObject *output,
                            PyArrayObject *src,
                            PyArrayObject *dst,
                            int order,
                            BordersMode mode,
                            double constant_value);

#endif