# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""=====================================
@author : kaifang zhang
@time   : 2021/12/19 1:33 PM
@contact: kaifang.zkf@dtwave-inc.com
====================================="""
import codecs
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
# B-ORG I-ORG 机构的开始位置和中间位置
# B-PER I-PER 人物名字的开始位置和中间位置
# B-LOC I-LOC 位置的开始位置和中间位置

################## 1. 读取数据
# 训练数据和标签
train_lines = codecs.open('msra/train/sentences.txt').readlines()
train_lines = [x.replace(' ', '').strip() for x in train_lines]  # 用于移除字符串开头和结尾指定的字符（默认为空格或换行符）或字符序列。
train_tags = codecs.open('msra/train/tags.txt').readlines()
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[tag_type.index(x) for x in tag] for tag in train_tags]
train_lines, train_tags = train_lines[:20000], train_tags[:20000]  # 只取两万数据
print(f"样例数据：{train_lines[0]} \n样例标签：{train_tags[0]}")

# 验证数据和标签
val_lines = codecs.open('msra/val/sentences.txt').readlines()
val_lines = [x.replace(' ', '').strip() for x in val_lines]
val_tags = codecs.open('msra/val/tags.txt').readlines()
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[tag_type.index(x) for x in tag] for tag in val_tags]  # 标签转换为数值

################## 2. 对数据进行分词
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 中文注意加list(train_lines),因为不加因为单词作为整体了。
maxlen = 64
train_encoding = tokenizer.batch_encode_plus(list(train_lines), truncation=True, padding=True, max_length=maxlen)
val_encoding = tokenizer.batch_encode_plus(list(val_lines), truncation=True, padding=True, max_length=maxlen)


################## 3. 定义Dataset类对象
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx][:maxlen]) for key, value in self.encodings.items()}
        # 字级别的标注，注意填充cls，这里[0]代表cls。后面不够长的这里也是补充0，样本tokenizer的时候已经填充了
        # item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (63-len(self.labels[idx])))[:64]
        item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (maxlen - 1 - len(self.labels[idx])))[:maxlen]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encoding, train_tags)
test_dataset = TextDataset(val_encoding, val_tags)
batchsz = 32
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)

# print(train_dataset[0])

# 测试样本是否满足最大长度
for idx in range(len(train_dataset)):
    item = train_dataset[idx]
    for key in item:
        if item[key].shape[0] != 64:
            print(key, item[key].shape)
for idx in range(len(test_dataset)):
    item = test_dataset[idx]
    for key in item:
        if item[key].shape[0] != 64:
            print(key, item[key].shape)

################## 4. 定义模型
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=7)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器和学习率
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)  # Default value in run_glue.py


################## 4. 训练测试以及字符的分类准确率
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # shape: [32, 64]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        # loss = outputs.loss
        logits1 = outputs[1]  # shape: [32, 64, 7]
        out = logits1.argmax(dim=2)
        out1 = out.data
        # logits2 = outputs.logits

        if idx % 20 == 0:  # 看模型的准确率
            with torch.no_grad():
                # 假如输入的是64个字符，64 * 7
                print((outputs[1].argmax(2).data == labels.data).float().mean().item(), loss.item())

        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optimizer.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += (outputs[1].argmax(2).data == labels.data).float().mean().item()

    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))
    print("-------------------------------")


# tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
def predcit(s):
    item = tokenizer([s], truncation=True, padding='longest', max_length=64)  # 加一个list
    with torch.no_grad():
        input_ids = torch.tensor(item['input_ids']).to(device).reshape(1, -1)
        attention_mask = torch.tensor(item['attention_mask']).to(device).reshape(1, -1)
        labels = torch.tensor([0] * attention_mask.shape[1]).to(device).reshape(1, -1)

        outputs = model(input_ids, attention_mask, labels)
        outputs = outputs[0].data.cpu().numpy()

    outputs = outputs[0].argmax(1)[1:-1]
    ner_result = ''
    ner_flag = ''

    for o, c in zip(outputs, s):
        # 0 就是 O，没有含义
        if o == 0 and ner_result == '':
            continue
        #
        elif o == 0 and ner_result != '':
            if ner_flag == 'O':
                print('机构：', ner_result)
            if ner_flag == 'P':
                print('人名：', ner_result)
            if ner_flag == 'L':
                print('位置：', ner_result)

            ner_result = ''

        elif o != 0:
            ner_flag = tag_type[o][2]
            ner_result += c
    return outputs


# for epoch in range(4):
#     print("------------Epoch: %d ----------------" % epoch)
#     train()
#     validation()
# torch.save(model, 'bert-ner.pt')
model = torch.load('/data/aibox/kaifang/NLP学习资料/bert-ner.pt')
s = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
# 识别出句子里面的实体识别（NER）
data = predcit(s)
s = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
# 识别出句子里面的实体识别（NER）
data = predcit(s)
s = '人工智能是未来的希望，也是中国和美国的冲突点。'
data = predcit(s)
s = '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。'
data = predcit(s)
s = '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难'
data = predcit(s)
s = '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报'
data = predcit(s)
s = '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
data = predcit(s)
