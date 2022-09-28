import os
import torch
import pandas as pd
from fastNLP import DataSet, Vocabulary
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.pipe import IMDBPipe
from torch import nn
from fastNLP.modules import LSTM
from fastNLP.embeddings import BertEmbedding
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric, ClassifyFPreRecMetric, ConfusionMatrixMetric
from fastNLP import Tester
from fastNLP.io.model_io import ModelSaver
import pickle as pkl
from models import GRU

"""
模型训练
参考：https://rwqzcq.github.io/2022/01/04/nlp/fastnlp/
"""
df = pd.read_csv("dataset/processed.csv")
# 设置最大字符数
# max_len = 410
# # 处理content字段
# def process_content(sentence):
#     sentence = str(sentence).strip()
#     if len(sentence) >= max_len:
#         sentence = sentence[0: max_len]
#     return sentence
# df['content'] = df['content'].apply(process_content)
# 再次过滤内容为空的数据
df = df[df['content'] != '']
# 构造fastnlp数据格式
df['sentence'] = df['content']
df['words'] = df['content'].apply(lambda x: x.split())
df['seq_len'] = df['words'].apply(lambda x: len(x))
df['target'] = df['label']

def gen_dataset(_type):
    """
    生成dataset
    """
    _df = df[df['type'] == _type]
    dataset = DataSet({
        'raw_words': _df['content'].values,
        'words': _df['words'].values,
        'seq_len': _df['seq_len'].values,
        'target': _df['target'].values
    })
    return dataset

train_dataset = gen_dataset('train')
test_dataset = gen_dataset('test')
valid_dataset = gen_dataset('valid')

# 构建vocab
vocab = Vocabulary()
#  将验证集或者测试集在建立词表是放入no_create_entry_dataset这个参数中。
vocab.from_dataset(train_dataset, field_name='words', no_create_entry_dataset=[valid_dataset, test_dataset])

# 创建databundle
data_bundle = DataBundle({'words': vocab}, {'train': train_dataset, 'test': test_dataset, 'valid': valid_dataset})

# 处理databunble
data_bundle = IMDBPipe().process(data_bundle)
# 定义模型
class BiGRUMaxPoolCls(nn.Module):
    def __init__(self, embed, num_classes, hidden_size=400, num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = embed
        self.rnn = GRU(self.embed.embedding_dim, hidden_size=hidden_size//2, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, words, seq_len):  # 这里的名称必须和DataSet中相应的field对应，比如之前我们DataSet中有chars，这里就必须为chars
        # chars:[batch_size, max_len]
        # seq_len: [batch_size, ]
        words = self.embed(words)
        outputs, _ = self.rnn(words, seq_len)
        outputs = self.dropout_layer(outputs)
        outputs, _ = torch.max(outputs, dim=1)
        outputs = self.fc(outputs)

        return {'pred':outputs}  # [batch_size,], 返回值必须是dict类型，且预测值的key建议设为pred

char_vocab = data_bundle.get_vocab('words')
# 加载Bert
bert_embed = BertEmbedding(char_vocab, model_dir_or_name='en', auto_truncate=True, requires_grad=True)
# 定义模型
model = BiGRUMaxPoolCls(bert_embed, len(data_bundle.get_vocab('target')))
# 定义损失函数
loss = CrossEntropyLoss()
# 定义优化器
optimizer = Adam(model.parameters(), lr=2e-5)
# 定义评价指标
acc_metric = AccuracyMetric()
cls_report_metric = ClassifyFPreRecMetric(only_gross=False, f_type="macro")
cls_confusion_metric = ConfusionMatrixMetric()
# 定义设备
device = 0 if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快

# 定义trainer
batch_size = 8
n_epochs = 20
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                  optimizer=optimizer, batch_size=batch_size, dev_data=data_bundle.get_dataset('valid'),
                  metrics=[cls_report_metric], device=device, n_epochs=n_epochs)
# 开始训练，训练完成之后默认会加载在dev上表现最好的模型
trainer.train()

# 测试模型
print("Performance on test is:")

tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=[acc_metric, cls_report_metric, cls_confusion_metric], batch_size=batch_size, device=device)
tester.test()

# 保存模型
model_path = './save_models/bert_senti.pt'
saver = ModelSaver(model_path)
saver.save_pytorch(model)

with open('./save_models/data_bundle.pkl', 'wb') as wp:
    pkl.dump(data_bundle, wp)