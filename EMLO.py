# coding: utf-8
"""使用emlo训练词向量,使用热唤醒和衰减学习率"""

from datetime import timedelta
import numpy as np
import tensorflow as tf
import time
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def get_data_count(path):
    c = 0
    for record in tqdm(tf.python_io.tf_record_iterator(path)):
        c += 1
    return c


class emlo(object):

    def __init__(self, config):
        self.config = config
        self.__createModel()
        self.train_data = {}
        self.test_data = {}

    def initialize(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver(name='save_saver')
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, save_path)

    def __get_data(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        if is_train:
            dataset = dataset.shuffle(self.config.batch_size * 10)
            dataset = dataset.prefetch(self.config.batch_size)
        dataset = dataset.batch(self.config.batch_size)
        iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        seq, tag, mask = iter.get_next()

        seq = tf.cast(seq, tf.int32)
        tag = tf.cast(tag, tf.int32)
        mask = tf.cast(mask, tf.float32)

        seq = tf.reshape(seq, [-1, self.config.n_ctx])
        tag = tf.reshape(tag, [-1, self.config.n_ctx])
        mask = tf.reshape(mask, [-1])
        mask = tf.sequence_mask(mask, self.config.n_ctx - 1)

        # 创建tag

        return seq, tag, mask, iter.make_initializer(dataset)

    def __train(self, seq, cells, tag, mask, is_training=False):
        """
        计算损失函数
        :param seq: 序列
        :param tag: 预测序列
        :param mask: 掩码，去除无效部分
        :return:
        """
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            # 创建双向的输入
            embedding = tf.get_variable("embedding", [self.config.n_vocab, self.config.n_embd], dtype=tf.float32)

            seq_em = [tf.nn.embedding_lookup(embedding, seq, name='input_em_f'),
                      tf.nn.embedding_lookup(embedding, tf.reverse(seq, [1]),
                                             name='input_em_b')
                      ]
            lstm_outputs = self.__bilstm(seq_em, cells, self.config, 'Bi-LSTM', is_training)

            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
                # Glorit init (std=(1.0 / sqrt(fan_in))
                softmax_init = tf.random_normal_initializer(0.0,
                                                            1.0 / np.sqrt(self.config.projection_dim))
                softmax_W = tf.get_variable(
                    'softmax_W', [self.config.n_vocab, self.config.projection_dim],
                    dtype=tf.float32,
                    initializer=softmax_init
                )
                softmax_b = tf.get_variable(
                    'softmax_b', [self.config.n_vocab],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

            loss = self.__lm_loss(lstm_outputs, tag, mask, softmax_W, softmax_b)

            perplexity = self.__perplexity(lstm_outputs, tag, mask, softmax_W, softmax_b)

        return loss, perplexity

    def __lm_loss(self, lstm_outputs, tag, mask, softmax_W, softmax_b):

        lstm_outputs = [lstm_outputs[0][-1], lstm_outputs[1][-1]]

        lstm_outputs = [tf.reshape(x, [-1, self.config.n_ctx, self.config.n_embd]) for x in lstm_outputs]

        logits1 = tf.matmul(tf.reshape(lstm_outputs[0][:, :-1, :], [-1, self.config.n_embd]),
                            softmax_W, transpose_b=True) + softmax_b

        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1,
                                                           labels=tf.one_hot(tf.reshape(tag[:, 1:], [-1, 1]),
                                                                             self.config.n_vocab))

        loss1 = tf.reshape(loss1, [-1, self.config.n_ctx - 1])
        loss1 = tf.boolean_mask(loss1, mask)

        logits2 = tf.matmul(tf.reshape(lstm_outputs[1][:, :-1, :], [-1, self.config.n_embd]),
                            softmax_W, transpose_b=True) + softmax_b
        loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2,
                                                           labels=tf.one_hot(
                                                               tf.reshape(tf.reverse(tag[:, :-1], [1]), [-1, 1]),
                                                               self.config.n_vocab))
        loss2 = tf.reshape(loss2, [-1, self.config.n_ctx - 1])
        loss2 = tf.boolean_mask(loss2, tf.reverse(mask, [1]))

        return (tf.reduce_mean(loss1) + tf.reduce_mean(loss2)) / 2

    def __perplexity(self, lstm_outputs, tag, mask, softmax_W, softmax_b):
        """
        计算困惑度
        :param lstm_outputs:
        :param tag:
        :param mask:
        :return:
        """
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            lstm_outputs = [lstm_outputs[0][-1], lstm_outputs[1][-1]]
            lstm_outputs = [tf.reshape(x, [-1, self.config.n_ctx, self.config.n_embd]) for x in lstm_outputs]

            wte = softmax_W
            softmax_b = tf.reshape(softmax_b, [-1, 1])
            all_perplexity = []
            for index, h_flat in enumerate(lstm_outputs):
                h_flat = h_flat[:, :-1, :]
                h_flat = tf.reshape(h_flat, [-1, self.config.n_embd])
                # 标签的wte
                if index == 0:
                    tag_c = tag[:, 1:]
                    mask_c = mask
                else:
                    tag_c = tf.reverse(tag[:, :-1], [1])
                    mask_c = tf.reverse(mask, [1])
                tag_w = tf.reshape(tf.gather(wte, tag_c), [-1, self.config.n_embd])
                tag_b = tf.reshape(tf.gather(softmax_b, tag_c), [-1, 1])
                logits_tag = tf.exp(tf.reduce_sum(h_flat * tag_w, axis=-1, keepdims=True) + tag_b)
                # 循环计算logits
                logits_all = tf.reduce_sum(
                    tf.exp(tf.matmul(h_flat, wte, transpose_b=True) + tf.transpose(softmax_b, [1, 0])), axis=-1, keepdims=True)
                logits = logits_tag / logits_all
                logits = tf.reshape(logits, [-1, self.config.n_ctx - 1])
                p = logits
                p = tf.boolean_mask(tf.log(p), mask_c)
                p = -tf.reduce_mean(p, axis=-1)
                perplexity = tf.exp(p)
                all_perplexity.append(perplexity)

        return tf.reduce_mean((all_perplexity[0] + all_perplexity[1]) / 2)

    def __bilstm(self, inputs, cells, config, name, is_training):
        with tf.name_scope("bilstm-%s" % (name)):
            with tf.variable_scope("var-bilstm-%s" % (name), reuse=tf.AUTO_REUSE):
                rt = []
                for index, x in enumerate(inputs):
                    x = tf.layers.dropout(x, config.keep_prob, training=is_training)
                    input1 = cells[index][0](x)
                    input1 = tf.layers.dense(input1, config.projection_dim, reuse=tf.AUTO_REUSE,
                                             name='dense_1_%d' % (index))
                    input1 = tf.layers.dropout(input1, config.keep_prob, training=is_training)

                    # 第二层加残差和dropout
                    input2 = cells[index][1](input1)
                    input2 = tf.layers.dense(input2, config.projection_dim, reuse=tf.AUTO_REUSE,
                                             name='dense_2_%d' % (index))
                    input2 = input2 + input1
                    input2 = tf.layers.dropout(input2, config.keep_prob, training=is_training)
                    rt.append([input1, input2])
            return rt

    def __get_learning_rate(self, num_warmup_steps):
        # 使用衰减学习率
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=self.config.learning_rate, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            self.config.max_step,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = self.config.learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        return learning_rate, global_step

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            def parser(record):
                features = tf.parse_single_example(record,
                                                   features={
                                                       'seq': tf.FixedLenFeature([self.config.n_ctx], tf.int64),
                                                       'tag': tf.FixedLenFeature([self.config.n_ctx], tf.int64),
                                                       'mask': tf.FixedLenFeature([1], tf.int64)
                                                   }
                                                   )
                return features['seq'], features['tag'], features['mask']

            seq, tag, self.mask, self.train_data_op = self.__get_data(
                [self.config.train_data_path], parser,
                is_train=True)

            # Implements linear decay of the learning rate.
            self.learning_rate, self.global_step = self.__get_learning_rate(self.config.warm_up)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            cells = []
            for x in range(self.config.layers):
                cells.append([tf.keras.layers.CuDNNLSTM(self.config.lstm_dim, return_sequences=True),
                              tf.keras.layers.CuDNNLSTM(self.config.lstm_dim, return_sequences=True)])

            loss, perplexity = self.__train(seq, cells, tag, self.mask, True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim = optimizer.minimize(loss, global_step=self.global_step)

            self.loss = loss
            self.perplexity = perplexity

            self.summary_loss = tf.summary.scalar('Loss', self.loss)
            self.summary_perplexity = tf.summary.scalar('Perplexity', self.perplexity)

            self.saver = tf.train.Saver()
            self.saver_v = tf.train.Saver(tf.trainable_variables())

            self.merged = tf.summary.merge_all()

            for index, x in enumerate(tf.trainable_variables()):
                logging.info('%d:%s' % (index, x))

    def train(self, load_path, save_path, log_path):
        log_writer = tf.summary.FileWriter(log_path, self.graph)
        start_time = time.time()
        total_batch = 0  # 总批次
        all_loss = 0
        all_perplexity = 0
        flag = True

        print_per_batch = self.config.print_per_batch // 10
        save_per_batch = self.config.print_per_batch

        # 获取数据量
        # data_count = get_data_count(self.config.train_data_path)
        data_count = 80500000
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            sess.run(tf.global_variables_initializer())
            if load_path:
                self.saver.restore(sess, load_path)

            for epoch in range(self.config.num_epochs):
                logging.info('epoch:%d'%(epoch))
                if not flag:
                    return
                sess.run(self.train_data_op)
                for step in range(data_count // self.config.batch_size):
                    # 运行后话，loss，perplexity分别是这批次数据的损失函数值和混合度
                    loss, perplexity, summary_loss, summary_perplexity, global_step, learning_rate, _ = sess.run(
                        [self.loss, self.perplexity, self.summary_loss, self.summary_perplexity, self.global_step,
                         self.learning_rate,
                         self.optim])  # 运行优化
                    log_writer.add_summary(summary_loss, global_step)
                    log_writer.add_summary(summary_perplexity, global_step)
                    all_loss += loss
                    all_perplexity += perplexity
                    total_batch += 1

                    if total_batch % (print_per_batch) == 0:
                        time_dif = self.get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Loss: {1:>6.2}, perplexity: {2:>6.5}, lr: {3}, Time: {4}'
                        logging.info(msg.format(total_batch, all_loss / print_per_batch,
                                                all_perplexity / print_per_batch, learning_rate,
                                                time_dif))
                        all_loss = 0
                        all_perplexity = 0

                    if total_batch % self.config.print_per_batch == 0:
                        self.saver.save(sess=sess, save_path=save_path)

                    if global_step > self.config.max_step:
                        flag = False
                        return

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))


class Config(object):
    """CNN配置参数"""
    n_vocab = 17247  # 词库大小
    n_ctx = 202  # 序列最大长度
    n_embd = 300  # 词向量维度
    layers = 2  # 网络层数
    proj_clip = 3
    projection_dim = n_embd
    lstm_dim = 600
    cell_clip = 3
    keep_prob = 0.1
    use_skip_connections = True

    num_sampled = 8192
    num_gpu = 1

    batch_size = 64
    learning_rate = 0.001

    print_per_batch = 1000  # 每多少轮输出一次结果

    train_data_path = '../data/train/lm/train_test.record'
    num_epochs = 200
    max_step = 80000
    warm_up = 3000


if __name__ == '__main__':
    config = Config()
    oj = emlo(config)
    with oj.graph.as_default():
        for index, x in enumerate(tf.trainable_variables()):
            print('%d:%s' % (index, x))
    load_path = '../model/w2v_300_25_sg1_lm/model.ckpt'
    save_path = '../model/EMLO_new/model.ckpt'
    log_path = '../model/log/EMLO_new'
    oj.train(load_path, save_path, log_path)
