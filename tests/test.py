import numpy as np
import utils
import time

def prep(l, astype):
    return np.ascontiguousarray(np.array(l).flatten(), dtype=astype)

def numpy_likvec(probs, partials):
    """ Calculate the likelihood vector for a site """
    result = np.empty(partials.shape)
    for i in xrange(partials.shape[0]):
        result[i] = (probs*partials[i]).sum(1)
    return result

p = np.array([[ 0.825092,  0.084274,  0.045317,  0.045317],
              [ 0.084274,  0.825092,  0.045317,  0.045317],
              [ 0.045317,  0.045317,  0.825092,  0.084274],
              [ 0.045317,  0.045317,  0.084274,  0.825092]])
s = np.array([[ 0.,  0.,  0.,  1.], [ 0.,  1.,  0.,  0.], [ 0.,  1.,  0.,  0.]]*1000)
# r = np.empty(np.prod(s.shape))
# print r.flags
# print r

p2 = p.copy()
s2 = s.copy()

reps = 10000

start = time.time()
for _ in range(reps):
    c=utils.likvec(p, s)
end = time.time()
print 'Time taken for {} calls of likvec = {}'.format(reps, end-start)
c=c.reshape(3000,4)
print c

start = time.time()
for _ in range(reps):
    m=utils.likvec2(p, p2, s, s2)
end = time.time()
print 'Time taken for {} calls of likvec_mv = {}'.format(reps, end-start)
print m


reps/=200
start = time.time()
for _ in range(reps):
    n=numpy_likvec(p, s)
end = time.time()
print 'Time taken for {} calls of numpy_likvec = {}'.format(reps, end-start)
print n

print 'Results are the same?', np.allclose(m, c), np.allclose(m, n)
