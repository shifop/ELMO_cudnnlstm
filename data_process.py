import json
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
import tensorflow.contrib.keras as kr
import os
from collections import Counter
import re
from absl import flags,app

"""生成语言模型训练数据"""


def save_as_record(path, data, vocab_map, max_length):
    train_writer = tf.python_io.TFRecordWriter(path)
    padding_value = len(vocab_map)
    delete_count = 0
    for x in tqdm(data):
        # 转化为id
        cache = []

        for word in x:
            if word == ' ' or word == '':
                continue
            if word in vocab_map.keys():
                cache.append(vocab_map[word])
            else:
                cache.append(vocab_map['<NONE>'])

        seq = [vocab_map['<START>']]
        seq.extend(cache[:max_length-2])
        seq.append(vocab_map['<END>'])
        mask = [len(seq)-1]
        seq = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=padding_value, padding='post',
                                                      truncating='post')
        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(mask, np.int64)))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())
    train_writer.close()


def read_txt(path, filterfn=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if filterfn!=None:
                line = filterfn(line)
            data.append(line.strip())
    return data


def get_count(path, limit, filterfn=None):
    data = read_txt(path, filterfn)
    length_map=[]
    vocab = []
    rt = []
    for index, x in enumerate(data):
        if index==200000:
            break
        cache = [w for w in x]
        if len(cache) >limit or len(cache)<10:
            continue
        length = str(len(cache))
        length_map.append(length)
        vocab.extend(cache)
        rt.append(x)

    length_map = Counter(length_map)
    vocab = Counter(vocab)
    return rt, length_map, vocab

def rm(data):
    return re.sub("/. *", "_",data)[:-1]

def main(argv):
    w2i = {}
    w2i['<START>'] = 0
    w2i['<END>'] = 1
    w2i['<NONE>'] = 2

    data, length_map, vocab = get_count(FLAGS.data_path, limit=FLAGS.max_length)

    for index, x in enumerate(set([x for x in vocab.keys() if vocab[x] > 1])):
        w2i[x] = index + 3

    print('文本数量：%d' % (len(data)))

    # 分割训练集和验证集
    length = len(data)
    random.shuffle(data)
    train_data = data

    # 保存数据
    save_as_record(os.path.join(FLAGS.save_path,"train_test.record"), train_data, w2i, FLAGS.max_length+2)

    # 保存w2i
    with open(os.path.join(FLAGS.save_path,"w2i.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(w2i, ensure_ascii=False))


if __name__=="__main__":
    import logging
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)
    flags.DEFINE_string("data_path", default="./data/raw/电信工单-语言模型训练语料.txt",
                        help="训练数据的地址，每行一篇文本,编码是utf-8")
    flags.DEFINE_integer("max_length", default=200,
                         help="文本最大长度")
    flags.DEFINE_string("save_path", default="./data/process/default",
                        help="处理好的数据保存地址")

    FLAGS = flags.FLAGS
    app.run(main)