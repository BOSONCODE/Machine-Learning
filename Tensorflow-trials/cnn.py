import tensorflow as tf
import input_data
import scipy.io as sio
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

    
    #定义权重和偏差的初始化函数，这样省得后来一遍遍定义，直接调用初始化函数就可以了。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
    
    #定义卷积层和池化层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
    
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    #定义输入的placeholder，x是特征，y_是真实的label。
    #因为卷积神经网络是会用到2D的空间信息，所以要把784维的数据恢复成28*28的结构，
    #使用的函数就是tf.shape的函数。
if __name__ == "__main__":
    with tf.Session() as sess:
        '''config = tf.ConfigProto(allow_soft_placement=True)
    # 这一行设置 gpu 随使用增长，我一般都会加上
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)'''
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        
        #定义第一个卷积层
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2_2(h_conv1)
        
        #定义第二个卷积层
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2_2(h_conv2)
        
        #定义第一个全连接层
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        #根据之前讲过的，在训练数据比较小的情况下，为了防止过拟合，随机的将一些节点置0，增加网络的泛化能力
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        #最后一个输出层也要对权重和偏差进行初始化。
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
        #定义损失函数和训练的步骤，使用Adam优化器最小化损失函数。
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        #计算准确率
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.initialize_all_variables().run()
        for i in range(20):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        batch = mnist.train.next_batch(1)
        print(sess.run(y_conv, feed_dict = {x:batch[0], keep_prob:1.0}))