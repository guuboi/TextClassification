# encoding: utf-8
import os
import time
import numpy as np
import tensorflow as tf
from utils import batch_index, time_diff, load_word2id, load_corpus_word2vec, load_corpus


class TextCNN(object):
    def __init__(self, config, embeddings):
        self.update_w2v = config.update_w2v
        self.n_class = config.n_class
        self.max_sen_len= config.max_sen_len
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.filters = config.filters
        self.output_channels = config.output_channels
        self.n_hidden = config.n_hidden
        self.n_epoch = config.n_epoch
        self.learning_rate = config.learning_rate
        self.drop_keep_prob = config.drop_keep_prob

        self.x = tf.placeholder(tf.int32, [None, self.max_sen_len], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')
        self.word_embeddings = tf.constant(embeddings, tf.float32)
        self.build()


    def cnn(self, mode='train'):
        """
        :param mode:默认为None，主要调节dropout操作对训练和预测带来的差异。
        :return: 未经softmax变换的fully-connected输出结果
        """
        inputs = self.add_embeddings()
        inputs = tf.expand_dims(inputs, -1)
        conv_outputs = []
        for i, filter_size in enumerate(self.filters):
            with tf.variable_scope('conv_maxpool-%s' % filter_size):
                # 卷积核
                kernel = tf.get_variable(name='conv_kernel%d' % i,
                                         shape=[filter_size, self.embedding_dim, 1, self.output_channels],
                                         initializer=tf.random_uniform_initializer(-0.01, 0.01))
                # 偏倚
                bias = tf.get_variable(name="conv_bias%d" % i,
                                       shape=[self.output_channels],
                                       initializer=tf.zeros_initializer())
                # 卷积层
                conv = tf.nn.conv2d(input=inputs,
                                    filter=kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID") + bias
                # 非线性变换
                h = tf.nn.relu(conv, name='relu')
                # 最大池化
                pooled = tf.reduce_max(h, axis=[1, 2])
                conv_outputs.append(pooled)

        # 拼接所有filter的卷积输出结果
        conv_outputs = tf.concat(conv_outputs, axis=1)
        total_channels = conv_outputs.shape[-1]  # num_of_filters * num_of_output_channel
        # 模型训练过程中执行对conv_outpus执行dropout操作
        if mode == 'train':
            conv_outputs = tf.nn.dropout(conv_outputs, keep_prob=self.drop_keep_prob)

        # 添加fully-connected层
        with tf.variable_scope('fully_connected'):
            limit = np.sqrt(6.0 / (total_channels.value + self.n_class))
            self.W = tf.get_variable(name='fc_weight',
                                     shape=[total_channels, self.n_class],
                                     initializer=tf.random_uniform_initializer(-limit, limit))
            self.b = tf.get_variable(name='fc_bias',
                                     shape=[self.n_class],
                                     initializer=tf.random_uniform_initializer(-0.003, 0.003))
        if mode == 'train':
            pred = tf.matmul(conv_outputs, self.W) + self.b
        else:
            pred = self.drop_keep_prob * tf.matmul(conv_outputs, self.W) + self.b

        return pred

    def add_embeddings(self):
        inputs = tf.nn.embedding_lookup(self.word_embeddings, self.x)
        return inputs

    def add_loss(self, pred):
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
        cost = tf.reduce_mean(cost)
        return cost

    def add_optimizer(self, loss):
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-6)
        opt = optimizer.minimize(loss)
        return opt

    def add_accuracy(self, pred):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def get_batches(self, x, y=None, batch_size=100, is_shuffle=True):
        for index in batch_index(len(x), batch_size, is_shuffle=is_shuffle):
            n = len(index)
            feed_dict = {
                self.x: x[index]
            }
            if y is not None:
                feed_dict[self.y] = y[index]
            yield feed_dict, n

    def build(self):
        self.pred = self.cnn(mode='train')
        self.loss = self.add_loss(self.pred)
        self.optimizer = self.add_optimizer(self.loss)

    def train_on_batch(self, sess, feed):
        accuracy = self.add_accuracy(self.pred)
        _, _loss, _acc = sess.run([self.optimizer, self.loss, accuracy], feed_dict=feed)
        return _loss, _acc

    def test_on_batch(self, sess, feed):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            pred = self.cnn(mode='test')
        loss = self.add_loss(pred)
        accuracy = self.add_accuracy(pred)
        _loss, _acc = sess.run([loss, accuracy], feed_dict=feed)
        return _loss, _acc

    def predict_on_batch(self, sess, feed, prob=True):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            pred = self.cnn(mode='predict')
        result = tf.argmax(pred, 1)
        if prob:
            result = tf.nn.softmax(logits=pred, dim=1)

        res = sess.run(result, feed_dict=feed)
        return res

    def evaluate(self, sess, x, y):
        """评估在某一数据集上的准确率和损失"""
        num = len(x)
        total_loss, total_acc = 0., 0.
        for _feed, _n in self.get_batches(x, y, batch_size=self.batch_size):
            loss, acc = self.test_on_batch(sess, _feed)
            total_loss += loss * _n
            total_acc += acc * _n

        return total_loss / num, total_acc / num

    def fit(self, sess, x_train, y_train, x_dev, y_dev, save_dir=None, print_per_batch=100):
        # saver = tf.train.Saver()
        # if save_dir:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        # sess.run(tf.global_variables_initializer())

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0 # 总批次
        best_acc_dev = 0.0  # 最佳验证集准确率
        last_improved = 0   # 记录上次提升批次
        require_improvement = 1000  # 如果超过1000轮模型效果未提升，提前结束训练
        flags = False
        for epoch in range(self.n_epoch):
            print('Epoch:', epoch + 1)
            for train_feed, train_n in self.get_batches(x_train, y_train, batch_size=self.batch_size):
                loss_train, acc_train = self.train_on_batch(sess, train_feed)
                loss_dev, acc_dev = self.evaluate(sess, x_dev, y_dev)

                # if total_batch % print_per_batch == 0:
                if acc_dev > best_acc_dev:
                    # 保存在验证集上性能最好的模型
                    best_acc_dev = acc_dev
                    last_improved = total_batch
                    # if save_dir:
                    #     saver.save(sess=sess, save_path=save_dir)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = time_diff(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' + \
                      ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    print('No optimization for a long time, auto-stopping...')
                    flags = True
                    break
            if flags:
                break


class CONFIG():
    update_w2v = True
    n_class = 8
    max_sen_len = 50
    embedding_dim = 50
    batch_size = 200
    filters = [3, 4, 5]
    output_channels = 20
    n_hidden = 50
    n_epoch = 5
    learning_rate = 0.01
    drop_keep_prob = 0.5


config = CONFIG()
word2id = load_word2id('./data/word_to_id.txt')
print('加载word2vec==========================')
word2vec = load_corpus_word2vec('./data/corpus_word2vec.txt')
print('加载train语料库========================')
train = load_corpus('./data/train/', word2id, max_sen_len=config.max_sen_len)
print('加载dev语料库==========================')
dev = load_corpus('./data/dev/', word2id, max_sen_len=config.max_sen_len)
print('加载test语料库=========================')
test = load_corpus('./data/test/', word2id, max_sen_len=config.max_sen_len)


x_tr, y_tr = train
x_val, y_val = dev

config = CONFIG()
tc = TextCNN(config=config, embeddings=word2vec)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    tc.fit(sess, x_tr, y_tr, x_val, y_val)
