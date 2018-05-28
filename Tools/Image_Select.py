'''
Fuction:
this module is used to select the trainnumber face-images
from one person in dataset A
'''
import numpy as np

#A:nxm, labels:1xn n denotes samplenum m denotes demension

def Image_Select(A, labels, trainnumber):
    N = len(A); M = len(A[0]);
    data = np.empty((0, A.shape[1]))
    label = np.empty((0))
    test_data = np.empty((0, A.shape[1]))
    test_label = np.empty((0))
    class_num = np.unique(labels)
    for c in class_num:
        c_index = np.where(labels == c)[0]
        np.random.shuffle(c_index)
        c_data = A[c_index[:trainnumber],:]
        c_labels = labels[c_index[:trainnumber]]
        data = np.concatenate((data, c_data), axis=0)
        label = np.concatenate((label, c_labels), axis=0)
        individual_test_data = A[c_index[trainnumber:],:]
        individual_test_labels = labels[c_index[trainnumber:]]
        test_data = np.concatenate((test_data, individual_test_data), axis = 0)
        test_label = np.concatenate((test_label, individual_test_labels), axis = 0)
    return data, label, test_data, test_label

if __name__ == '__main__':
    A = np.array([[1, 2, 3, 8], [4, 5, 6, 9]]).transpose()
    labels = np.array([1, 2, 1, 2]);
    train_data, train_labels, test_data, test_labels = Image_Select(A, labels, 1)
    print(train_data, train_labels, test_data, test_labels)