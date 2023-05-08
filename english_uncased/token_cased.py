#-- coding: utf-8 --
#@Date : 05/01/2023 15:17
#@Author : zxu
#@File : token_cased.py
#@Software: PyCharm

import torch
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification

from torch.optim.sgd import SGD
from tqdm import tqdm
import numpy as np
import os


df = pd.read_csv('../data/ner/ner.csv')
df.head()

# 根据空格拆分标签，并将它们转换为列表
labels = [i.split() for i in df['labels'].values.tolist()]
# 检查数据集中有多少标签
unique_labels = set()
for lb in labels:
  [unique_labels.add(i) for i in lb if i not in unique_labels]

labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
# text_tokenized = tokenizer(example, padding='max_length',
#                            max_length=512, truncation=True,
#                            return_tensors="pt")


def align_label(texts, labels):
    # 首先tokenizer输入文本
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
  # 获取word_ids
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []
    # 采用上述的第一中方法来调整标签，使得标签与输入数据对其。
    for word_idx in word_ids:
        # 如果token不在word_ids内，则用 “-100” 填充
        if word_idx is None:
            label_ids.append(-100)
        # 如果token在word_ids内，且word_idx不为None，则从labels_to_ids获取label id
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        # 如果token在word_ids内，且word_idx为None
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if False else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids
# 构建自己的数据集类
class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df):
        # 根据空格拆分labels
        lb = [i.split() for i in df['labels'].values.tolist()]
        # tokenizer 向量化文本
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512,
                                truncation=True, return_tensors="pt") for i in txt]
        # 对齐标签
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels

df = df[0:10000]
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])

class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(
                       'bert-base-cased',
                                     num_labels=len(unique_labels),cache_dir='../models/bert-base-cased/tokenCls')

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        return output

def train_loop(model, df_train, df_val):
    # 定义训练和验证集数据
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1)
    # 判断是否使用GPU，如果有，尽量使用，可以加快训练速度
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    # 定义优化器
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    # if use_cuda:
    #     model = model.cuda()
    # 开始训练循环
    best_acc = 0
    best_loss = 1000
    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        # 训练模型
        model.train()
        # 按批量循环训练模型
        for train_data, train_label in tqdm(train_dataloader):
      # 从train_data中获取mask和input_id
            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)
            # 梯度清零！！
            optimizer.zero_grad()
            # 输入模型训练结果：损失及分类概率
            loss, logits = model(input_id, mask, train_label)
            # 过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            # 获取最大概率值
            predictions = logits_clean.argmax(dim=1)
      # 计算准确率
            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()
      # 反向传递
            loss.backward()
            # 参数更新
            optimizer.step()
        # 模型评估
        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        for val_data, val_label in val_dataloader:
      # 批量获取验证数据
            val_label = val_label[0].to(device)
            mask = val_data['attention_mask'][0].to(device)
            input_id = val_data['input_ids'][0].to(device)
      # 输出模型预测结果
            loss, logits = model(input_id, mask, val_label)
      # 清楚无效token对应的结果
            logits_clean = logits[0][val_label != -100]
            label_clean = val_label[val_label != -100]
            # 获取概率值最大的预测
            predictions = logits_clean.argmax(dim=1)
            # 计算精度
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
            total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'''Epochs: {epoch_num + 1} |
                Loss: {total_loss_train / len(df_train): .3f} |
                Accuracy: {total_acc_train / len(df_train): .3f} |
                Val_Loss: {total_loss_val / len(df_val): .3f} |
                Accuracy: {total_acc_val / len(df_val): .3f}''')

LEARNING_RATE = 1e-2
EPOCHS = 5
model = BertModel()
train_loop(model, df_train, df_val)
# os.makedirs('../mode', exist_ok=True)
torch.save(model, '../models/savedModel')