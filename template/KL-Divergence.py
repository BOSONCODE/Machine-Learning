'''
KL(p||q) = \sum{pk log (pk/qk)}
KL divergence是用来衡量两个分布之间的相似度的
KL(p||q) = \sum{pk log pk - pk log qk} = -H(p) + H(p, q)
Cross Entropy: H(p, q) = -\sum{pk log qk}
'''

import numpy as np
from math import log

def KL(p,  q):
    lenp = len(p)
    lenq = len(q)
    assert(lenp == lenq)
    p = np.array(p, dtype = np.float64)
    q = np.array(q, dtype = np.float64)
    sumk = 0.0
    for i in range(lenq):
        sumk = sumk + p[i]*log(p[i]/q[i])
    return sumk

def Cross_Entropy(p, q):
    lenp = len(p)
    lenq = len(q)
    assert(lenp == lenq)
    sumk = 0.0
    p = np.array(p, dtype = np.float64)
    q = np.array(q, dtype = np.float64)
    for i in range(lenq):
        sumk = sumk + p[i]*log(q[i])
    return -sumk

if __name__ ==  '__main__':
    A = [0.5, 0.5]
    B = [0.25, 0.75]
    C = [0.125, 0.875]
    print(KL(A, B), Cross_Entropy(A, C))
