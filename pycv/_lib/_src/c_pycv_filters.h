#ifndef C_PYCV_FILTERS_H
#define C_PYCV_FILTERS_H

// #####################################################################################################################

int PYCV_convolve(PyArrayObject *input,
                  PyArrayObject *kernel,
                  PyArrayObject *output,
                  npy_intp *center,
                  PYCV_ExtendBorder mode,
                  npy_double c_val);


int PYCV_rank_filter(PyArrayObject *input,
                     PyArrayObject *footprint,
                     PyArrayObject *output,
                     npy_intp rank,
                     npy_intp *center,
                     PYCV_ExtendBorder mode,
                     npy_double c_val);

// #####################################################################################################################


#endif