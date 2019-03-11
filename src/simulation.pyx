# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.stdlib cimport rand, RAND_MAX, srand
cimport numpy as np
from random import randint
srand(randint(0, RAND_MAX))

ctypedef fused int_t:
    int
    long

#########################################################################
## SIMULATION FUNCTIONS - Random Sampling
#########################################################################

cpdef int set_seed(int seed):
    srand(seed)

cpdef int_t weighted_choice(int_t[::1] choices, double[::1] weights) nogil:
    """
    Return a random choice from int array 'choices', proportionally
    weighted by 'weights'
    """
    cdef size_t l = weights.shape[0]
    cdef double total = 0.0
    for i in xrange(l):
        total += weights[i]
    cdef double r = total * rand()/RAND_MAX
    cdef double upto = 0

    for i in xrange(l):
        upto += weights[i]
        if upto > r:
            return choices[i]
    return 0

cpdef int weighted_choices(int_t[:] choices, double[:] weights,
                           int_t[:] output) nogil:
    """
    Fill output array with weighted choices from 'choices' array, proportionally
    weighted by weights
    """
    cdef size_t nchoices = weights.shape[0]
    cdef size_t nsites = output.shape[0]
    cdef double total = 0.0
    cdef size_t i, j
    for i in xrange(nchoices):
        total += weights[i]
    cdef double r, upto

    for i in xrange(nsites):
        r = total * rand()/RAND_MAX
        upto = 0
        for j in xrange(nchoices):
            upto += weights[j]
            if upto > r:
                output[i] = choices[j]
                break
    return 0

cpdef int evolve_states(int_t[::1] all_states, int_t[::1] parent_states,
                       int_t[::1] rate_class, double[:,:,:] probs, double[::1] probs_at_index,
                       int_t[::1] child_states) nogil:
    """
    Evolve states from parent to child according to 'probs'
    @param all_states All possible states (i.e. the alphabet) to simulate from
    @param parent states Current sequence present at the ancestor
    @param rate_class Array giving the gamma rate class that each site belongs to
    @param probs [NxNxC] array of [NxN] substitution probabilities for each of C
           gamma rate classes
    @param probs_at_index [N] array used as temporary storage of probabilities
    @param child_states Simulated child states are written to this buffer
    """
    cdef size_t nsites = parent_states.shape[0]
    cdef size_t nstates = probs_at_index.shape[0]
    cdef size_t parent_state_at_index, rate_class_at_index, i, j

    for i in xrange(nsites):
        parent_state_at_index = parent_states[i]
        rate_class_at_index = rate_class[i]
        for j in xrange(nstates):
            probs_at_index[j] = probs[parent_state_at_index, j, rate_class_at_index]
        child_states[i] = weighted_choice(all_states, probs_at_index)

    return 0
