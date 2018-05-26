'''
Fuction:
this module is used to select the trainnumber face-images
from one person in dataset A
'''
import numpy as np

#A:mxn, labels:1xn n denotes samplenum m denotes demension

def Image_Select(A, labels, trainnumber):
    M = len(A); N = len(A[0]);  assert(trainnumber <= M);
    ClassLabel = np.unique(labels)
    nClass = len(ClassLabel)
    train_data = np.array([0, 0])
    train_labels = np.array([0, 0])
    test_data = np.array([0, 0])
    test_labels = np.array([0, 0])
    for c in range(1, nClass+1):
        c_label = np.argwhere(labels == c) #提取标签为c的标签，依次存入标签c_label=[c,c,c……]Nc个c
        c_Data = A[:,c_label] #提取标签为c的样本
        Nc = np.size(c_Data, 1)
        data = c_label.copy()
        np.random.shuffle(data)
        print(c_Data[:, data[0:trainnumber]])
        train_data = np.vstack((train_data, c_Data[:, data[0:trainnumber]]))
        train_labels = np.vstack((train_labels, c_label[:, data[0:trainnumber]]))
        test_data = np.vstack((test_data, c_Data[:, trainnumber : Nc]))
        test_labels = np.vstack((test_labels, c_label[:, trainnumber : Nc]))
    return train_data, train_labels, test_data, test_labels

if __name__ == '__main__':
    A = np.array([[1, 2, 3, 8], [4, 5, 6, 9]])
    labels = np.array([1, 2, 1, 2]);
    train_data, train_labels, test_data, test_labels = Image_Select(A, labels, 1)
    print(train_data, train_labels, test_data, test_labels)