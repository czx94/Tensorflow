from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cifar10

# # #读入mnist数据集
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 载入CIFAR-10数据集
#label are one_hot encoded
cifar10_dir = './cifar-10-batches-py/'
X_train, y_train, X_test, y_test = cifar10.load_CIFAR10(cifar10_dir)
#create data generator
train_generator = cifar10.data_generator(X_train, y_train, 128)
test_generator = cifar10.data_generator(X_test, y_test, 128)

#Hyperparameters
learning_rate = 0.001
batch_size = 256
dropout = 0.5
training_iters = 100000
display_step = 50

#params
length = 32
width = 32
dimension = 3
n_classes = 10

#io
x = tf.placeholder('float', [None, length, width, dimension])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

#kernel size
weights = {
    'c1': tf.Variable(tf.truncated_normal([3,3,3,64], stddev=0.001, name='wc1')),
    'c2': tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.001, name='wc2')),
    'c3': tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.01, name='wc3')),
    'c4': tf.truncated_normal([3,3,128,64], stddev=0.01, name='wc4'),
    'c5': tf.truncated_normal([3,3,64,64], stddev=0.1, name='wc5'),
    'c6': tf.truncated_normal([3,3,64,32], stddev=0.1, name='wc6'),
    'f1': tf.Variable(tf.truncated_normal([4*4*32, 1024], stddev=0.1, name='fc1')),
    'f2': tf.Variable(tf.truncated_normal([1024, 128], stddev=0.1, name='fc2')),
    'f3': tf.Variable(tf.truncated_normal([128, 10], stddev=0.1, name='fc3'))
}

biases = {
    'c1': tf.Variable(tf.constant(0.1, shape=[64], name='bc1')),
    'c2': tf.Variable(tf.constant(0.1, shape=[128], name='bc2')),
    'c3': tf.Variable(tf.constant(0.1, shape=[128], name='bc3')),
    'c4': tf.constant(0.1, shape=[64], name='bc4'),
    'c5': tf.constant(0.1, shape=[64], name='bc5'),
    'c6': tf.constant(0.1, shape=[32], name='bc6'),
    'f1': tf.Variable(tf.constant(0.1, shape=[1024], name='fc1')),
    'f2': tf.Variable(tf.constant(0.1, shape=[128], name='fc1')),
    'f3': tf.Variable(tf.constant(0.1, shape=[10], name='fc1'))
}

#funcs
def conv2d(input, kernel, bias, name):
    wx_plus_b = tf.nn.bias_add(tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME'), bias)
    batch_norm = bn_layer(wx_plus_b, True, name)
    return tf.nn.relu(batch_norm, name=name)

def pooling(input, k, name):
    return tf.nn.max_pool(input, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name)

def lrn(input, name, lsize=5):
    return tf.nn.lrn(input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# 实现Batch Normalization
def bn_layer(x,is_training,name,moving_decay=0.9,eps=1e-5):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]

    param_shape = shape[-1]
    with tf.variable_scope(name):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
        beta  = tf.get_variable('beat', param_shape,initializer=tf.constant_initializer(0))

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
                lambda:(ema.average(batch_mean),ema.average(batch_var)))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)

def fc_layer(input, weight, bias, name, batch_norm = True):
    wx_plus_b = tf.matmul(input, weight) + bias
    if batch_norm:
        wx_plus_b = bn_layer(wx_plus_b, True, name)
    return tf.nn.relu(wx_plus_b, name=name)

#model
def alexnet(X, weights, biases, dropout):
    # X = tf.reshape(X, [-1, 28, 28, 1])

    #conv1
    conv1 = conv2d(X, weights['c1'], biases['c1'], 'conv1')
    pool1 = pooling(conv1, 2, 'pool1')
    # norm1 = bn(pool1, 'norm1')
    norm1 = tf.nn.dropout(pool1, dropout)

    # conv2
    conv2 = conv2d(norm1, weights['c2'], biases['c2'], 'conv2')
    pool2 = pooling(conv2, 2, 'pool2')
    # norm2 = bn(pool2, 'norm2')
    norm2 = tf.nn.dropout(pool2, dropout)

    # conv3
    conv3 = conv2d(norm2, weights['c3'], biases['c3'], 'conv3')
    pool3 = pooling(conv3, 2, 'pool3')
    # norm3 = bn(pool3, 'norm3')
    norm3 = tf.nn.dropout(pool3, dropout)

    # conv1
    conv1 = conv2d(X, weights['c1'], biases['c1'], 'conv1')
    pool1 = pooling(conv1, 2, 'pool1')
    # norm1 = bn(pool1, 'norm1')
    norm1 = tf.nn.dropout(pool1, dropout)

    # conv2
    conv2 = conv2d(norm1, weights['c2'], biases['c2'], 'conv2')
    pool2 = pooling(conv2, 2, 'pool2')
    # norm2 = bn(pool2, 'norm2')
    norm2 = tf.nn.dropout(pool2, dropout)

    # conv3
    conv3 = conv2d(norm2, weights['c3'], biases['c3'], 'conv3')
    pool3 = pooling(conv3, 2, 'pool3')
    # norm3 = bn(pool3, 'norm3')
    norm3 = tf.nn.dropout(pool3, dropout)

    #fc
    fc1 = tf.reshape(norm3, [-1, weights['f1'].get_shape().as_list()[0]])
    fc1 = fc_layer(fc1, weights['f1'], biases['f1'], 'fc1')
    fc2 = fc_layer(fc1, weights['f2'], biases['f2'], 'fc2')
    fc3 = fc_layer(fc2, weights['f3'], biases['f3'], 'fc3', batch_norm=False)

    return fc3

#construct model
model = alexnet(x, weights, biases, keep_prob)

#learn
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#init
init = tf.initialize_all_variables()

#run!
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step < training_iters:
        batch_xs, batch_ys = next(train_generator)
        # get batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Train acc
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # train loss
            loss_val = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss_val) + ", Training Accuracy= " + "{:.5f}".format(acc))
        #Test acc
        if step % 1000 == 0:
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.}))
        step += 1

    print ("Optimization Finished!")