import numpy as np
import scipy.io as sio
from sklearn import neighbors
from sklearn import manifold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
from sklearn.decomposition import NMF

def Read_Data(): 
    InputData = sio.loadmat('C:/Users/Administrator/Desktop/Python/dataset/UMIST_face_575.mat')
    data = InputData['A'].T
    label = InputData['labels'].T
    label.shape = (np.size(label))
    train_data, train_label, test_data, test_label = Image_Select(data, label, 9)
    return train_data, train_label, test_data, test_label
    '''
    #skf = KFold(n_splits=10)
    #data = DR_Unsupervised(data, 200, method="LLE")
    #data = DR_supervised(data, label, 20, 'LDA')
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(data, label):
        yield data[train_index], label[train_index], data[test_index], label[test_index]
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=10)
    return X_train, y_train, X_test, y_test
    '''

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
        test_label = np.concatenate((test_label, individual_test_labels), axis=0)
    return data, label, test_data, test_label
'''
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
'''
    
    
def unsupervised_DR(X, d, name = "PCA"):
    if name == "PCA":
        import PCA
        P = PCA.PCA(X, d)
        '''from sklearn.decomposition import PCA
        pca = PCA(n_components=d)
        data = pca.fit_transform(X)
        return data'''
        return X @ P
    elif name == "LE":
        import LaplacianEigenmaps
        val, data = LaplacianEigenmaps.LE(X, 5, 1)
        return data[:, d]
    elif name == "none":
        return X

def Classify2(data, label, test_data, test_label, method="KNN"):
    if method == "KNN":
        newData = unsupervised_DR(data, 14, "PCA")
        knn = neighbors.KNeighborsClassifier(n_neighbors=1)
        knn.fit(newData, label)
        newTestData = unsupervised_DR(test_data, 14, "PCA")
        p_acc = knn.score(newTestData, test_label, sample_weight=None)
        print(p_acc)

def Classify(data, label, test_data, test_label, method = "KNN"):
    if method == "KNN":
        k = 10
        model = NMF(n_components=k, init='random', random_state=0)
        num = data.shape[0]
        dim = 32
        newData = np.array(np.zeros([num, dim*k]))
        for i in range(num):
            W = model.fit_transform(data[i].reshape(dim, dim))
            newData[i] = W.reshape(1, dim*k)
        knn = neighbors.KNeighborsClassifier(n_neighbors=1)
        knn.fit(newData, label)
        num = test_data.shape[0]
        newTestData = np.array(np.zeros([num, dim*k]))
        for i in range(num):
            W = model.fit_transform(test_data[i].reshape(dim, dim))
            newTestData[i] = W.reshape(1, dim*k)
        p_acc = knn.score(newTestData, test_label, sample_weight=None)
        print(p_acc)
        #y_pred = knn.predict(test_data)
        #print(metrics.accuracy_score(test_label, y_pred))

if __name__ == '__main__':
    data, label, test_data, test_label = Read_Data()
        #data = DR_supervised(data, label, 1024, method = "LDA")
        #test_data = DR_supervised(test_data, test_label, 1024, method = "LDA")
        #data = DR_Unsupervised(data, 20, method="PCA")
        #test_data = DR_Unsupervised(test_data, 20, method = "PCA")
    Classify2(data, label, test_data, test_label, method="KNN")