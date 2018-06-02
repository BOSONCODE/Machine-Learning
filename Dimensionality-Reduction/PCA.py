'''
objective function: max_{P}\frac{1}{m-1}||P'(X - meanX)||^2
=> min_{tr(P'AP)} s.t. P'P = I A = cov(X, meanX)
'''
import numpy as np

#A: nxm n denotes the samplenum
#P: m*d

def PCA(X, d):
    X = np.array(X, dtype = np.float)
    meanX = np.mean(X, axis=0)
    X = X - meanX
    A = np.cov(X, rowvar = 0) #rowvar 表示每一行为一个样本
    eigVals, eigVector = np.linalg.eig(np.mat(A))
    eigValsIndice = np.argsort(eigVals)
    d_eigValIndice = eigValsIndice[-1: - (d + 1):-1]
    d_eigVector = eigVector[:, d_eigValIndice]
    return d_eigVector

if __name__ == '__main__':
    a = [[1, 2, 3], [2, 3, 4]]
    P = PCA(a, 3)
    print(a@P)
    