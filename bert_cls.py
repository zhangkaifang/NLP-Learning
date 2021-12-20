# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""=====================================
@author : kaifang zhang
@time   : 2021/12/17 10:28 AM
@contact: kaifang.zkf@dtwave-inc.com
====================================="""
import codecs  # 文件读取
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 用于训练集和测试集划分
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

random.seed(1001)

#################### 1. 读取文本
news_label = [int(x.split('_!_')[1]) - 100 for x in codecs.open('toutiao_cat_data.txt')]  # 逐行,获取标签
# print(news_label[:5])
news_text = [x.strip().split('_!_')[-1] if x.strip()[-3:] != '_!_' else x.strip().split('_!_')[-2]
             for x in codecs.open('toutiao_cat_data.txt')]  # 文本
# print(news_text[:5])

#################### 2. 划分为训练集和验证集
x_train, x_test, train_label, test_label = train_test_split(news_text[:50000],
                                                            news_label[:50000],
                                                            test_size=0.2,
                                                            stratify=news_label[:50000])  # stratify按照标签进行采样，训练集和验证部分同分布

#################### 3. tokenizer分词器，本质就是词典,对字进行编码
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
max_length = 64
train_encoding = tokenizer.batch_encode_plus(x_train, truncation=True, padding=True, max_length=max_length)
test_encoding = tokenizer.batch_encode_plus(x_test, truncation=True, padding=True, max_length=max_length)


#################### 4. 数据集读取, 把数据封装成Dataset对象
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):  # idx表示索引
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)
# print(train_dataset[1])
#################### 4. 单个数据读取到批量batch_size读取
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


#################### 4. 精度计算
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#################### 5. 定义分类模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=17)  # num_labels类别数量
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

#################### 6. 优化方法
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


#################### 7. 训练函数
def train(epoch):
    model.train()
    total_train_loss = 0
    iter_num = 0
    # total_iter = len(train_loader)
    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
    for batch in epoch_iterator:
        # 正向传播
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        # 这里添加进度条的描述信息
        epoch_iterator.set_description(
            f"epoch:{epoch} " +
            f"loss: {loss.item():.4f} " +
            f"lr: {scheduler.get_last_lr()[0]:.1e}")

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

        # 参数更新
        optimizer.step()
        scheduler.step()

        iter_num += 1
        # if (iter_num % 100 == 0):
        #     print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
        #         epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


#################### 8. 验证函数
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))


if __name__ == '__main__':
    for epoch in range(4):
        train(epoch)
        validation()
