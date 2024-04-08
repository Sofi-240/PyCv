#ifndef C_PYCV_PEAKS_H
#define C_PYCV_PEAKS_H

// #####################################################################################################################

int PYCV_peaks_nonmaximum_suppression(PyArrayObject *input,
                                      npy_intp *min_distance,
                                      double threshold,
                                      PYCV_ExtendBorder mode,
                                      double c_val,
                                      PyArrayObject **output);


// #####################################################################################################################


#endif