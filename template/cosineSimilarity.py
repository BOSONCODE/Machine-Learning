import numpy as np

def CosineSimilarity(x, y):
    x = np.array(x, dtype = np.int)
    y = np.array(y, dtype = np.int)
    return np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))

if __name__ == '__main__':
    x = np.array([1, 1, 2, 1, 1, 1, 0, 0, 0], dtype = np.int)   
    y = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1], dtype = np.int)
    print(CosineSimilarity(x, y))
