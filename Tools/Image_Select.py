'''
Fuction:
this module is used to select the trainnumber face-images
from one person in dataset A
'''
import numpy as np

#A:nxm, labels:1xn n denotes samplenum m denotes demension

#rd_data: 样本数x维度 label: 样本数x1
def Image_Select(rd_data, rd_labels, selected_images):
    data = np.empty((0, rd_data.shape[1]))
    label = np.empty((0))
    test_data = np.empty((0, rd_data.shape[1]))
    test_label = np.empty((0))

    unique_class = np.unique(rd_labels)
    for individual in unique_class:
        individual_idx = np.where(rd_labels == individual)[0]
        np.random.shuffle(individual_idx)

        individual_data = rd_data[individual_idx[:selected_images],:]
        individual_labels = rd_labels[individual_idx[:selected_images]]

        data = np.concatenate((data, individual_data), axis = 0)
        label = np.concatenate((label, individual_labels), axis = 0)

        individual_test_data = rd_data[individual_idx[selected_images:],:]
        individual_test_labels = rd_labels[individual_idx[selected_images:]]

        test_data = np.concatenate((test_data, individual_test_data), axis = 0)
        test_label = np.concatenate((test_label, individual_test_labels), axis = 0)

    # shuffle
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx, :]
    label = label[idx]

    test_idx = np.arange(test_data.shape[0])
    np.random.shuffle(test_idx)
    test_idx = test_idx[:np.size(test_idx)]

    test_data = test_data[test_idx, :]
    test_label = test_label[test_idx]

    #print("train size:", data.shape)
    #print("test size:", test_data.shape)

    return data, label, test_data, test_label

if __name__ == '__main__':
    A = np.array([[1, 2, 3, 8], [4, 5, 6, 9]]).transpose()
    labels = np.array([1, 2, 1, 2])
    train_data, train_labels, test_data, test_labels = Image_Select(A, labels, 1)
    print(train_data, train_labels, test_data, test_labels)