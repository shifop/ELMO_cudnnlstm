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

def read_txt(path, filterfn=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if filterfn!=None:
                line = filterfn(line)
            line = line.strip()
            if len(line)<3:
                continue
            data.append(line.strip())
    return data


def get_count(path, filterfn=None):
    length_map = []
    vocab = Counter('1')
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        index = 0
        for line in tqdm(f):
            if filterfn!=None:
                line = filterfn(line)
            line = line.strip()
            if len(line)<3:
                continue
            index+=1
            cache = [w for w in line]
            length = str(len(cache))
            length_map.append(length)
            vocab.update(cache)
            count+=1

    length_map = Counter(length_map)
    return length_map, vocab, count

def rm(data):
    return re.sub("/. *", "_",data)[:-1]


def write_to_doc(doc, train_writer, vocab_map, max_length, padding_value):
    for x in doc:
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
        seq.extend(cache[:max_length - 2])
        seq.append(vocab_map['<END>'])
        mask = [len(seq) - 1]
        seq = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=padding_value, padding='post',
                                                      truncating='post')
        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(mask, np.int64)))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())

def main(argv):
    w2i = {}
    w2i['<START>'] = 0
    w2i['<END>'] = 1
    w2i['<NONE>'] = 2

    # 生产词典
    length_map, vocab, count = get_count(FLAGS.data_path)

    for index, x in enumerate(set([x for x in vocab.keys() if vocab[x] > 1])):
        w2i[x] = index + 3

    print('文本数量：%d' % (count))

    # 分割训练集和验证集

    vocab_map = w2i
    max_length = FLAGS.max_length
    path = os.path.join(FLAGS.save_path,"train_test.record")

    train_writer = tf.python_io.TFRecordWriter(path)
    padding_value = len(vocab_map)
    doc = []
    tqdm_ = tqdm(total=count)
    with open(FLAGS.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) < 3:
                continue
            doc.append(line)
            tqdm_.update(1)

            if len(doc)==500000:
                random.shuffle(doc)
                logging.info('\n读取50w数据， 开始写入文件')
                write_to_doc(doc, train_writer, vocab_map, max_length, padding_value)
                doc = []
    logging.info('处理剩余数据')
    write_to_doc(doc, train_writer, vocab_map, max_length, padding_value)
    train_writer.close()

    # 保存w2i
    with open(os.path.join(FLAGS.save_path,"w2i.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(w2i, ensure_ascii=False))


if __name__=="__main__":
    import logging

    random.seed(0)
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)
    flags.DEFINE_string("data_path", default="./data/raw/corpus2.txt",
                        help="训练数据的地址，每行一篇文本,编码是utf-8")
    flags.DEFINE_integer("max_length", default=128,
                         help="文本最大长度")
    flags.DEFINE_string("save_path", default="./data/process/",
                        help="处理好的数据保存地址")

    FLAGS = flags.FLAGS
    app.run(main)