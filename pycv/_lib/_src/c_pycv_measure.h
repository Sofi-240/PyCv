#ifndef C_PYCV_MEASURE_H
#define C_PYCV_MEASURE_H

// #####################################################################################################################


int PYCV_find_object_peaks(PyArrayObject *input,
                           npy_intp *min_distance,
                           npy_double threshold,
                           PYCV_ExtendBorder mode,
                           npy_double c_val,
                           PyArrayObject *output);

// #####################################################################################################################


#endif