'''
Fuction:
this module is used to select the trainnumber face-images
from one person in dataset A
'''
import numpy as np

#A:nxm, labels:1xn n denotes samplenum m denotes demension

def Image_Select(A, labels, trainnumber):
    N = len(A); assert(N > 0); M = len(A[0]); assert(M > 0); assert(trainnumber < N)
    ClassLabel = np.unique(labels)
    nClass = len(ClassLabel)
    for c in range(ClassLabel):
        c_label = labels[labels == c] #提取标签为c的标签，依次存入标签c_label=[c,c,c……]Nc个c
        c_Data = A[:, labels == c] #提取标签为c的样本
        

    return train_data, train_labels, test_data, test_labels