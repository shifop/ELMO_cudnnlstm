from EMLO import emlo
from absl import flags, app

class Config(object):

    def __init__(self, FLAGS):
        self.n_embd = FLAGS.embedding_size  # 词向量维度
        self.projection_dim = self.n_embd
        self.n_ctx = FLAGS.seq_length  # 序列长度
        self.n_vocab = FLAGS.vocab_size

        self.layers = FLAGS.layers
        self.proj_clip = FLAGS.proj_clip
        self.lstm_dim = FLAGS.lstm_dim
        self.cell_clip = FLAGS.cell_clip
        self.keep_prob = FLAGS.keep_prob
        self.use_skip_connections = FLAGS.use_skip_connections

        self.learning_rate = FLAGS.learning_rate  # 学习率
        self.batch_size = FLAGS.batch_size  # 每批训练大小
        self.num_epochs = FLAGS.num_epochs  # 总迭代轮次

        self.print_per_batch = FLAGS.print_per_batch  # 每多少轮输出一次结果

        self.train_data_path = FLAGS.train_data_path

        self.num_sampled = FLAGS.num_sampled
        self.max_step = FLAGS.max_step
        self.warm_up = FLAGS.warm_up



def main(argv):
    config = Config(FLAGS)
    oj = emlo(config)
    load_path = FLAGS.load_path
    save_path = FLAGS.save_path
    log_path = FLAGS.log_path
    oj.train(load_path, save_path, log_path)


if __name__=='__main__':
    flags.DEFINE_integer("embedding_size", default=300,
                         help="词向量维度")
    flags.DEFINE_integer("seq_length", default=202,
                         help="文本长度")
    flags.DEFINE_integer("vocab_size", default=3374,
                         help="词库大小")
    flags.DEFINE_integer("layers", default=2,
                         help="lstm层数")
    flags.DEFINE_integer("proj_clip", default=3,
                         help="正则系数")
    flags.DEFINE_integer("cell_clip", default=3,
                         help="正则系数")
    flags.DEFINE_integer("lstm_dim", default=600,
                         help="lstm内部维度")
    flags.DEFINE_float("keep_prob", default=0.1,
                         help="dropout比例")
    flags.DEFINE_boolean("use_skip_connections", default=True,
                       help="是否使用穿孔连接")
    flags.DEFINE_float("learning_rate", default=1e-3,
                       help="学习率")
    flags.DEFINE_integer("batch_size", default=64,
                         help="每批次包含的文本数")
    flags.DEFINE_integer("num_epochs", default=100,
                         help="训练多少代")
    flags.DEFINE_integer("print_per_batch", default=1000,
                         help="每多少步输出一次信息")
    flags.DEFINE_string("train_data_path", default="./data/process/default/train_data.record",
                        help="训练数据地址")
    flags.DEFINE_string("load_path", default="./model/default_1/model.ckpt",
                        help="加载模型地址")
    flags.DEFINE_string("save_path", default="./model/default/model.ckpt",
                        help="模型保存地址")
    flags.DEFINE_string("log_path", default="./model/log_default/",
                        help="日志保存地址")
    flags.DEFINE_integer("num_sampled", default=4096,
                         help="采样数量")
    flags.DEFINE_integer("max_step", default=240000,
                         help="最大训练次数")
    flags.DEFINE_integer("warm_up", default=10000,
                         help="")

    FLAGS = flags.FLAGS
    app.run(main)