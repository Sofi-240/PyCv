#ifndef QUERY_ARRAY_H
#define QUERY_ARRAY_H

typedef enum {
    LOCAL_MIN = 0,
    LOCAL_MAX = 1,
} LOCAL_Mode;

int is_local_q(PyArrayObject *input,
               PyArrayObject *strel,
               PyArrayObject *output,
               LOCAL_Mode mode,
               npy_intp *origins);


#endif