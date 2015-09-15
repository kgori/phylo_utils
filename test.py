import numpy as np
import utils
import time

def prep(l, astype):
    return np.ascontiguousarray(np.array(l).flatten(), dtype=astype)

p = np.array([[ 0.825092,  0.084274,  0.045317,  0.045317],
              [ 0.084274,  0.825092,  0.045317,  0.045317],
              [ 0.045317,  0.045317,  0.825092,  0.084274],
              [ 0.045317,  0.045317,  0.084274,  0.825092]])
s = np.array([[ 0.,  0.,  0.,  1.], [ 0.,  1.,  0.,  0.], [ 0.,  1.,  0.,  0.]]*1000)
# r = np.empty(np.prod(s.shape))
# print r.flags
# print r

n = 500000

# start = time.time()
# for _ in range(n):
#     c=utils.likvec(p, s)
# end = time.time()
# print 'Time taken for {} calls of likvec = {}'.format(n, end-start)
# c=c.reshape(3000,4)
# print c

start = time.time()
for _ in range(n):
    m=utils.likvec_mv(p, s)
end = time.time()
print 'Time taken for {} calls of likvec_mv = {}'.format(n, end-start)
print m

# print 'Results are the same?', np.allclose(m, c)
