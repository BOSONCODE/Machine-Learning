import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import input_data

def plotDataSet(dataMat):
    '''
    输入：从txt文本文件里读取的输入数据
    功能：画出前两个特征的二维图
    输出：散点图
    '''
    x1 = np.mat(dataMat)[:,1]
    x2 = np.mat(dataMat)[:,2]
    line1,= plt.plot(x1[:50],x2[:50],'ro',label = 'class1')
    line2, = plt.plot(x1[50:],x2[50:],'b^',label ='class0')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(handles=[line1,line2],loc = 2)
    #plt.show() 

def plotBestFit(dataMat,weights):
    '''
    输入：输入数据,权值矩阵
    功能：画出前两个特征的二维图及分类曲线
    输出：散点图
    '''
    print(weights[1])
    plt.figure()
    plotDataSet(dataMat)
    x = np.mat(np.arange(-4.0,4.0,0.1))
    y = (-weights[0]-weights[1] * x)/weights[2]
    plt.plot(x.transpose(),y.transpose())
    plt.show()


def test():
    data_train = sio.loadmat('MNIST_total_70000_samples')
    data_train_image = data_train['A'].transpose()
    train_data = data_train['labels'].flatten()
    op = tf.one_hot(train_data, 10)
    data_train_label = tf.Session().run(op)
    
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    in_units = 784
    h1_units = 300
    w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))

    b1 = tf.Variable(tf.zeros([h1_units]))
    w2 = tf.Variable(tf.zeros([h1_units, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    x = tf.placeholder(tf.float32, [None, in_units])
    keep_prob = tf.placeholder(tf.float32)

    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
    train = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    init = tf.global_variables_initializer()


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(init)
        for i in range(3001):
            #batch_xs, batch_ys = mnist.train.next_batch(100)  
            sess.run(train, feed_dict={x:data_train_image, y_:data_train_label, keep_prob:0.75})
            #sess.run(train, feed_dict={x:batch_xs,y_:batch_ys, keep_prob:0.75})
            if i % 200 ==0:  
            #训练过程每200步在测试集上验证一下准确率，动态显示训练过程  
                #print(i, 'training_arruracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels,   keep_prob: 1.0}))
                print(i, 'training_arruracy:', accuracy.eval({x:data_train_image, y_:data_train_label, keep_prob: 1.0}))
        #print(sess.run(accuracy, feed_dict={x: data_train_image, y_: data_train_label, keep_prob: 1.0}))
    
    
if __name__ == "__main__":
    data_train = sio.loadmat('MNIST_total_70000_samples')
    data_train_image = data_train['A'].transpose()
    train_data = data_train['labels'].flatten()
