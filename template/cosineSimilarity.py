import numpy as np

x = np.array([1, 1, 2, 1, 1, 1, 0, 0, 0], dtype = np.int)
y = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1], dtype = np.int)

cos = np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))
print('余弦相似度: ', cos)
