import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import os
import numpy as np
import regex as re
import pickle as pkl
from utils import read_class_file
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# config
class Config(object):
    """配置参数"""
    def __init__(self, data_path):
        self.model_name = 'TextCNN'
        self.vocab_path = data_path + '/vocab.pkl'                                # 词表
        self.save_path = './models/TextCNNv1w_add.ckpt'        # 模型训练结果
        self.log_path = data_path + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(data_path + '/embedding_SougouNews.npz')["embeddings"].astype('float32'))
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 4762                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 70                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4, 5)                                   # 卷积核尺寸
        self.num_filters = 256
        # 卷积核数量(channels数)
        class_file = data_path + '/class.txt'
        self.label_2_id, self.id_2_label = read_class_file(class_file)
        print()
        self.num_classes = len(self.id_2_label)
        self.label_lis = list(set([label for label, id in self.label_2_id.items()]))


# Text CNN
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)  ## out.size(): [128,32,300]  cnn:[in, out, filter] * text:[batch, 1, 32, 300] = [batch, out, (L-filter)/stride+1, 1] [128, 256, 31, 1] [128, 256, 1, 1] [128, 256]
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class PredictModel(object):
    def __init__(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config = Config(data_path='./data/processed')
        self.tokenizer = lambda x: [y for y in x]
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.UNK, self.PAD, self.label = '<UNK>', '<PAD>', '-1'
        self.model = Model(self.config).to(self.config.device)
        self.model.load_state_dict(torch.load(self.config.save_path, map_location='cpu'))
        self.model.eval()


    def build_data(self, text):
        content = text.strip()
        words_line = []
        token = self.tokenizer(content)
        seq_len = len(token)
        if self.config.pad_size:
            if len(token) < self.config.pad_size:
                token.extend([self.PAD] * (self.config.pad_size - len(token)))
            else:
                token = token[:self.config.pad_size]
        # word to id
        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(self.UNK)))

        return torch.LongTensor([words_line]).to(self.config.device)

    def prediction_model(self, text):
        data = self.build_data(text)
        with torch.no_grad():
            outputs = self.model(data)
            """出向量、预测出标签"""
            num = torch.argmax(outputs)
            outputs = torch.softmax(outputs, dim=-1)
            pro = round(float(torch.max(outputs) / torch.sum(outputs)), 3)
        label = self.config.id_2_label[int(num)]
        return {'label': label, 'score': pro}

predictModel = PredictModel()


if __name__ == "__main__":
    config = Config(data_path='./data/processed')
    preModel = PredictModel()
    preModel.prediction_model("的营养作用")