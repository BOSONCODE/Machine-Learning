'''
for specific explanation:
    https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
    https://blog.csdn.net/acdreamers/article/details/44663421
NMF: Nonnegative Matrix Factor 非负矩阵分解

Objective Function: 
J(W, H) = min_{W,H}{\frac{1}{2}||V - WH||_F^2}, subject to W >= 0, H >= 0

给定 V 属于 (R+)(nxm) 寻找非负矩阵W属于(R+)(nxr)和非负矩阵H属于(R+)(rxm), 使得 WH逼近V
W is the basis matrix, H is the coefficient matrix as well as the construction of V

具体的推导过程如下:
\frac{grad(J(W, H))}{W_{ik}} = (VH')_{ik} - (WHH')_{ik}
\frac{grad(J(W, H))}{H_{kj}} = (W'V)_{kj} - (W'WH)_{kj}
然后使用梯度下降设 learningrate = a1, a2
W_{ik} = W_{ik} - a1[(VH')_{ik} - (WHH')_{ik}]
H_{ik} = W_{kj} - a2[(W'V)_{kj} - (W'WH)_{kj}]
为了保证NMF的非负的性质, 这里巧妙地采用自适应性变步长学习
也就是 a1 = \frac{W_{ik}}{(W'WH)_{ik}}
a1 = \frac{H_{kj}}{(WHH')_{kj}}
最终得到迭代的更新公式:
W_{ik} = W_{ik}* \frac{(VH')_{ik}}{(WHH')_{ik}}
H_{kj} = H_{kj}*\frac{(W'V)_{kj}}{(W'WH)_{kj}}
'''

import numpy as np

#input V(n, m), W(n, r), H(r, m)
#m denotes the sample number, n denotes the dimension 
def NMF(V, Winit, Hinit, maxIter, n, r, m):
    W = np.array(Winit, dtype = np.float).reshape(n, r)
    H = np.array(Hinit, dtype=np.float).reshape(r, m)
    for iter in range(maxIter):
        #update W
        VH = V @ H.transpose()
        WHH = W @ H @ H.transpose()
        W = W * (VH / WHH)
        #update H
        WV = W.transpose() @ V
        WWH = W.transpose() @ W @ H
        H = H * (WV / WWH)
    return W, H

if __name__ == "__main__":
    # Generate a target Matrix
    origin_W = np.random.randint(1, 9, size=(5, 2))
    origin_H = np.random.randint(1, 9, size=(2, 5))
    v = np.dot(origin_W, origin_H)
    print('the initial matrix:', v)

    # Generate initial Matrix W and H (Random generation)
    w = np.random.randint(1, 9,size=(5, 2))
    h = np.random.randint(1, 9, size=(2, 5))
    
    W, H = NMF(v, w, h, 100, 5, 2, 5)
    print('reconstruct matrix:', W@H)