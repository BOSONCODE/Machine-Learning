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

def Read_Data(): 
    InputData = sio.loadmat('dataset/YaleB_32x32.mat')
    data = InputData['A'].T
    label = InputData['labels'].T
    #skf = KFold(n_splits=10)
    #data = DR_Unsupervised(data, 200, method="LLE")
    #data = DR_supervised(data, label, 20, 'LDA')
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(data, label):
        yield data[train_index], label[train_index], data[test_index], label[test_index]
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=10)
    return X_train, y_train, X_test, y_test
    '''
def DR_supervised(data, label, test_data, d, method = "LDA"):
    if method == "LDA":
        lda = LinearDiscriminantAnalysis(n_components = d)
        lda.fit(data, label)
        data = lda.transform(data)
        test_data = lda.transform(test_data)
        #data = lda(data, label)
        return data, test_data

def Classify(data, label, test_data, test_label, method = "KNN"):
    if method == "KNN":
        data, test_data = DR_supervised(data, label, test_data, 20, 'LDA')
        knn = neighbors.KNeighborsClassifier(n_neighbors=5)
        knn.fit(data, label)
        p_acc = knn.score(test_data, test_label, sample_weight=None)
        print(p_acc)
        y_pred = knn.predict(test_data)
        print(metrics.accuracy_score(test_label, y_pred))

if __name__ == '__main__':
    for data, label, test_data, test_label in Read_Data():
        #data = DR_supervised(data, label, 1024, method = "LDA")
        #test_data = DR_supervised(test_data, test_label, 1024, method = "LDA")
        #data = DR_Unsupervised(data, 20, method="PCA")
        #test_data = DR_Unsupervised(test_data, 20, method = "PCA")
        Classify(data, label, test_data, test_label, method = "KNN")