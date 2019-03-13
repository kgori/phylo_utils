# cython: language_level=3
cimport numpy as np
from cython cimport boundscheck
from libc.math cimport log
from libc.float cimport DBL_EPSILON
cdef double SCALE_THRESHOLD = DBL_EPSILON

@boundscheck(False)
cpdef inline int clv(double[:,:,:] probs_a, double[:,:,:] probs_b,
                     double[:,:] clv_a, double[:,:] clv_b,
                     double[:] scale_a, double[:] scale_b,
                     double[:] scale, double[:,:] out) nogil:
    cdef Py_ssize_t i, j, k, z
    cdef double s, max_s, scaleval
    for z in range(out.shape[1]):
        max_s = 0
        for i in range(probs_a.shape[0]):
            s = 0
            for j in range(clv_a.shape[0]):
                for k in range(probs_a.shape[1]):
                    s += probs_a[i, j, z] * probs_b[i, k, z] * clv_a[j, z] * clv_b[k, z]
            if s > max_s:
                max_s = s
            out[i, z] = s
        if max_s < SCALE_THRESHOLD:
            scale[k] = log(max_s) + scale_a[k] + scale_b[k]
            for i in range(probs_a.shape[0]):
                out[i, z] /= max_s
        else:
            scale[k] = scale_a[k] + scale_b[k]
    return 0


cdef _lnl_branch_derivs(probs, pi, partials_a, partials_b, scale_a, scale_b, out):
    pass

cdef _lnl_branch(probs, pi, partials_a, partials_b, scale_a, scale_b, out):
    pass

cdef _lnl_node(pi, partials, scale, out):
    pass

