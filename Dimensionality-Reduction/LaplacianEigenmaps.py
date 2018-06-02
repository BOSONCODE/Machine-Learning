'''
Algorithm Name: Laplacian Eigenmaps
LE is aimed at maintaining the closest point both high dimension and low dimension
objective function: min_{Y} \sum\sum W_{i, j}||(Y_i - Y_j)||^2

Preprocessing:
W(i, j) = e^(-||x_i - x_j||^2/t) t is the constant
Here, we use the hot kernel function to measure the distance between sample x_i and x_j
'''
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def knn(inX, data, k):
    N = data.shape[0]
    diffMat = np.tile(inX, [N, 1]) - data
    sqdiffMat = diffMat ** 2
    sqDistances = np.sum(sqdiffMat, axis=1)
    sortedDistanceIndice = sqDistances.argsort()
    return sortedDistanceIndice[0:k]

#N: samplenum  M: dimension

def LE(X, k, t):
    N, M = X.shape
    D = np.mat(np.zeros([N, N]))
    W = np.mat(np.zeros([N, N]))
    for i in range(N):
        k_index = knn(X[i,:], X, k)
        for j in range(k):
            diffMat = X[i,:] - X[k_index[j],:]
            sqdiffMat = np.array(diffMat)** 2
            sqDistances = np.sum(sqdiffMat, axis=0)
            W[i,k_index[j]] = np.exp(0-sqDistances/t)  
            D[i, i] += W[i, k_index[j]]
    L = D - W
    invD = np.linalg.inv(D)
    tmp = np.dot(invD, L)
    eigVal, eigVector = np.linalg.eig(tmp)
    return eigVal, eigVector

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [2, 3, 4]])
    val, vec = LE(a, 1, 1)
    print(vec)

    
    