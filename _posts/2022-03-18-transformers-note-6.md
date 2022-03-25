---
layout: article
title: Hugging Face 的 Transformers 库快速入门（六）：序列标注任务
tags:
    - Transformers
    - NLP
    - Pytorch
mathjax: true
sidebar:
  nav: transformers-note
---

我们的第一个实战任务是序列标注 (Sequence Labeling)，其目标是为文本（词语序列）中的每一个词 (token) 分配对应的标签，因此 Transformers 库将其称为 token 分类任务。

常见的序列标注任务有：

- **命名实体识别 (Named Entity Recognition, NER)：**识别出文本中诸如人物、地点、组织等实体，即为所有的 token 都打上实体标签，将其划分到某类实体，或者判定为“非实体”；
- **词性标注 (Part-Of-Speech tagging, POS)：**为文本中的每一个词语标注上对应的词性，例如名词、动词、形容词等。

下面我们以 NER 为例，运用 Transformers 库微调一个基于 BERT 的模型来完成任务。

## 准备数据

我们选择 [1998 年人民日报语料库](https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/SEYRX5)作为数据集，在其之上训练一个 NER 模型：每次输入一个句子，识别出其中的人物 (PER)、地点 (LOC) 和组织 (ORG)。人民日报语料库标注了大量的语言学信息，可以使用[处理工具](https://github.com/howl-anderson/tools_for_corpus_of_people_daily)从中抽取出数据用于分词、NER 等任务，这里我们直接使用处理好的 NER 语料：

[http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)

该语料已经划分好了训练集、验证集和测试集，分别对应 example.train、example.dev 和 example.test 文件，其中训练集 / 验证集 / 测试集分别包含 20864 / 2318 / 4636 个句子。语料采用我们在上一篇[《快速分词器》](/2022/03/08/transformers-note-5.html)中介绍的 IOB2 格式进行标注，一行对应一个字：

```
海 O
钓 O
比 O
赛 O
地 O
点 O
在 O
厦 B-LOC
门 I-LOC
与 O
金 B-LOC
门 I-LOC
之 O
间 O
的 O
海 O
域 O
。 O
```

回顾一下，在 IOB2 格式中，”B-XXX”表示某一类标签的开始，”I-XXX”表示某一类标签的中间，”O”表示非标签。NER 语料中包含有人物 (PER)、地点 (LOC) 和组织 (ORG) 三种实体类型，因此共有 7 种标签：

- **O：**该词不属于任何实体；
- **B-PER/I-PER：**该词是一个人物实体的起始/中间；
- **B-LOC/I-LOC：**该词是一个地点实体的起始/中间；
- **B-ORG/I-ORG：**该词是一个组织实体的起始/中间。

### 构建数据集

与之前一样，我们首先编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签：

```python
from torch.utils.data import Dataset

categories = set()

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, tags = '', []
                for i, c in enumerate(line.split('\n')):
                    word, tag = c.split(' ')
                    sentence += word
                    if tag[0] == 'B':
                        tags.append([i, i, word, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag[0] == 'I':
                        tags[-1][1] = i
                        tags[-1][2] += word
                Data[idx] = {
                    'sentence': sentence, 
                    'tags': tags
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

数据集中句子之间采用空行分隔，因此我们首先通过 `'\n\n'` 切分出句子，然后按行读取句子中每一个字和对应的标签，如果标签以 `B` 或者 `I` 开头，就表示出现了实体。下面我们通过读取文件构造数据集，并打印出一个训练样本：

```python
train_data = PeopleDaily('china-people-daily-ner-corpus/example.train')
valid_data = PeopleDaily('china-people-daily-ner-corpus/example.dev')
test_data = PeopleDaily('china-people-daily-ner-corpus/example.test')

print(train_data[0])
```

```
{'sentence': '海钓比赛地点在厦门与金门之间的海域。', 'tags': [[7, 8, '厦门', 'LOC'], [10, 11, '金门', 'LOC']]}
```

可以看到我们成功地抽取出了句子中的实体标签，以及实体在原文中对应的位置。

### 数据预处理

接着，我们就需要通过 `DataLoader` 库来按批 (batch) 从数据集中加载数据，将文本转换为模型可以接受的 token IDs 以及将实体类别转换为对应的数值。前面我们已经通过 `categories` 搜集了数据集中的所有实体标签，因此很容易建立标签映射字典：

```python
id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

print(id2label)
print(label2id)
```

```
{0: 'O', 1: 'B-LOC', 2: 'I-LOC', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-PER', 6: 'I-PER'}
{'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-PER': 5, 'I-PER': 6}
```

与[《快速分词器》](/2022/03/08/transformers-note-5.html)中的操作类似，我们需要通过快速分词器提供的映射函数，将实体标签从原文映射到切分出的token 上。下面我们以处理第一个样本为例，生成 token 对应的标签序列：

```python
from transformers import AutoTokenizer
import numpy as np

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence = '海钓比赛地点在厦门与金门之间的海域。'
tags = [[7, 8, '厦门', 'LOC'], [10, 11, '金门', 'LOC']]

encoding = tokenizer(sentence, truncation=True)
tokens = encoding.tokens()
label = np.zeros(len(tokens), dtype=int)
for char_start, char_end, word, tag in tags:
    token_start = encoding.char_to_token(char_start)
    token_end = encoding.char_to_token(char_end)
    label[token_start] = label2id[f"B-{tag}"]
    label[token_start+1:token_end+1] = label2id[f"I-{tag}"]

print(tokens)
print(label)
print([id2label[id] for id in label])
```

```
['[CLS]', '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。', '[SEP]']
[0 0 0 0 0 0 0 0 1 2 0 1 2 0 0 0 0 0 0 0]
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
```

可以看到，通过 `char_to_token()` 函数，我们成功地将实体标签从原文位置映射到切分后的 token 索引，并且将对应的标签设置为实体编号。

实际编写 DataLoader 的批处理函数 `collate_fn()` 时，我们处理的是一批中的多个样本，因此需要对上面的操作进行扩展。而且由于最终会通过交叉熵损失来优化模型参数，我们还需要将 `[CLS]`、`[SEP]`、`[PAD]` 等特殊 token 对应的标签设为 -100，以便在计算损失时忽略它们：

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_sentence, batch_tags  = [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_tags.append(sample['tags'])
    batch_inputs = tokenizer(
        batch_sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[s_idx][0] = -100
        batch_label[s_idx][len(encoding.tokens())-1:] = -100
        for char_start, char_end, _, tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
```

```
batch_X shape: {
    'input_ids': torch.Size([4, 65]), 
    'token_type_ids': torch.Size([4, 65]), 
    'attention_mask': torch.Size([4, 65])
}
batch_y shape: torch.Size([4, 65])

{'input_ids': tensor([
        [ 101, 7716, 6645, 1298, 6432, 8024, 1762,  125, 3299, 4638, 3189, 3315,
         6913, 6435, 6612,  677, 8024,  704, 1744, 7339, 6820, 3295, 7566, 1044,
         6814, 5401, 1744, 7339, 8124, 1146,  722, 1914, 8024,  852, 3297, 5303,
         4638, 5310, 2229,  793, 3221, 1927, 1164,  511,  102,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0],
        [ 101, 3341, 5632, 1744, 2157, 4906, 2825,  914, 6822, 1355, 2245,  704,
         2552,  510,  704, 1744, 1093,  689, 1920, 2110,  510,  704, 1744, 2456,
         3332, 4777, 4955, 7368,  510, 1266, 3175,  769, 1920, 5023, 1296,  855,
         4638,  683, 2157, 5440, 2175,  749, 7987, 1366, 4638,  821,  689, 8024,
         2900, 1139,  749, 7987, 1366, 1355, 2245, 2773, 4526,  704, 4638,  679,
         6639,  722, 1905,  511,  102],
        [ 101, 3173, 1814, 2773, 3159,  510,  673, 3983, 2275, 2773, 3159,  510,
         7942, 3817, 4518,  924, 1310, 2773,  510, 4721, 3333, 2773, 3159,  100,
          100, 2218, 3221,  711,  749,  924, 1310, 1157, 1157, 6414, 4495, 1762,
         3031, 5074, 7027, 4638, 7484, 1462, 2048, 1036,  511,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0],
        [ 101, 5401, 2102, 4767, 3779, 1062, 1385, 5307, 6814, 1939, 1213, 2894,
         3011, 8024, 6821, 3613,  793,  855, 2233, 5018, 1061,  511,  102,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0]]), 
 'token_type_ids': tensor(...), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

tensor([[-100,    5,    6,    6,    0,    0,    0,    0,    0,    0,    1,    2,
            0,    0,    0,    0,    0,    3,    4,    4,    0,    0,    0,    0,
            0,    3,    4,    4,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100],
        [-100,    0,    0,    3,    4,    4,    4,    4,    4,    4,    4,    4,
            4,    0,    3,    4,    4,    4,    4,    4,    0,    3,    4,    4,
            4,    4,    4,    4,    0,    3,    4,    4,    4,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    1,    2,    0,    0,    0,    0,
            0,    0,    0,    1,    2,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0, -100],
        [-100,    1,    2,    0,    0,    0,    1,    2,    2,    0,    0,    0,
            1,    2,    2,    0,    0,    0,    0,    1,    2,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100],
        [-100,    3,    4,    4,    4,    4,    4,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100]])
```

可以看到，DataLoader 按照我们设置的 batch size，每次对 4 个样本进行编码，并且将 token 序列 pad 到了相同的长度。标签序列中实体 token 对应的索引都转换为了实体编号，特殊 token 对应的索引都被设置为 -100。

> **注意：**由于我们在 DataLoader 中通过设置参数 `shuffle=True` 打乱了训练集，因此每一次遍历样本的顺序都是随机的，你运行的结果也会与上面不同。随机遍历训练集会使得每次训练得到的模型参数都不同，这会导致实验结果难以复现，因此大部分研究者会采用伪随机序列来进行实验。
>
> 即通过设置种子来生成随机序列，只要种子相同，生成的随机序列就是相同的：
>
> ```python
> import torch
> import random
> import numpy as np
> import os
> 
> seed = 7
> torch.manual_seed(seed)
> torch.cuda.manual_seed(seed)
> torch.cuda.manual_seed_all(seed)
> random.seed(seed)
> np.random.seed(seed)
> os.environ['PYTHONHASHSEED'] = str(seed)
> ```
>
> 只要你与本文一样将种子设置为 7，就可以得到与上面完全相同的结果。

## 训练模型

### 构建模型

对于序列标注任务，Transformers 库提供了封装好的 `AutoModelForTokenClassification` 类可以直接使用，只需传入分类标签数量即可快速实现一个 token 分类器。我们可以直接通过 `num_labels` 传入标签数量，或者是传入标签到编号的映射（更推荐），例如：

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
```

考虑到实际构建模型时还会包含很多自定义模块，因此本文采用手工编写 Pytorch 模型的方式来完成：首先利用 Transformers 库加载 BERT 模型，然后接一个全连接层完成分类：

```python
from torch import nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, len(id2label))

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        logits = self.classifier(bert_output.last_hidden_state)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

```
Using cpu device
NeuralNetwork(
  (bert_encoder): BertModel(
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(...)
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (classifier): Linear(in_features=768, out_features=7, bias=True)
)
```

可以看到，我们构建的模型首先运用 BERT 模型将每一个 token 都编码为维度为 768 的张量，然后将输出序列送入到一个包含 7 个神经元的线性全连接层中对每一个 token 进行分类。

为了测试模型的操作是否符合预期，我们尝试将一个 batch 的数据送入模型：

```python
outputs = model(batch_X)
print(outputs.shape)
```

```
torch.Size([4, 65, 7])
```

模型输出尺寸为 $4\times 65 \times 7$ 的张量，完全符合预期，对于 batch 内 4 个都被 pad 到 65 个 token 的样本，模型对每个 token 都输出了 7 维的 logits 值，分别对应 7 种实体标签的预测结果。

### 优化模型参数

与之前一样，我们将每一轮 (Epoch) 分为"训练循环"和"验证/测试循环"，在训练循环中计算损失、优化模型的参数，在验证/测试循环中评估模型的性能。下面我们首先实现训练循环：

```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss
```

注意，与文本分类任务对于每个样本只输出一个预测张量不同，token 分类任务会输出一个预测张量的序列（可以看成是对每个 token 都进行了一次分类），因此使用交叉熵计算模型损失时，不能像之前一样直接将模型的预测结果与标签送入到 `CrossEntropyLoss` 中进行计算。

对于高维度的输出（例如 2D 图像需要按像素计算交叉熵），`CrossEntropyLoss` 需要将输入维度调整为 $(batch, C, d_1, d_2,...,d_K)$，其中 $C$ 是类别个数，$K$ 是输入的维度。对于 token 分类任务，就是在 token 序列维度（Keras 中称时间步）上计算交叉熵，因此我们通过 `pred.permute(0, 2, 1)` 交换后两维，将模型预测结果从$(batch, seq, 7)$ 调整为 $(batch, 7, seq)$。

验证/测试循环负责评估模型的性能。这里我们借助 [seqeval](https://github.com/chakki-works/seqeval) 库进行评估，seqeval 是一个专门用于序列标注评估的 Python 库，支持 IOB、IOB、IOBES 等多种标注格式以及多种评估策略，例如：

```python
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

y_true = [['O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O']]

print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
```

```
              precision    recall  f1-score   support

         LOC       0.50      0.50      0.50         2
         PER       1.00      1.00      1.00         1

   micro avg       0.67      0.67      0.67         3
   macro avg       0.75      0.75      0.75         3
weighted avg       0.67      0.67      0.67         3
```

可以看到，对于第一个地点实体，模型虽然预测正确了其中 2 个 token 的标签，但是仍然判为识别错误，只有当预测的起始和结束位置都正确时才算识别正确。

在验证/测试循环中，我们首先将预测结果和正确标签都先转换为 seqeval 库接受的字符标签的形式，并且过滤掉标签值为 -100 的特殊 token，然后送入到 seqeval 提供的 `classification_report` 函数中计算 P / R / F1 等各项指标：

```python
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1)
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in y]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, y)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
```

最后，将"训练循环"和"验证/测试循环"组合成 Epoch，就可以进行模型的训练和验证了。这里我们使用 AdamW 优化器，并且通过 `get_scheduler()` 函数定义学习率调度器：

```python
from transformers import AdamW, get_scheduler

learning_rate = 1e-5
epoch_num = 3

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    test_loop(valid_dataloader, model)
print("Done!")
```

```
Using cuda device

Epoch 1/3
-------------------------------
loss: 0.051204: 100%|███████| 5216/5216 [06:11<00:00, 14.03it/s]
100%|█████████████████████████| 580/580 [00:24<00:00, 24.05it/s]
              precision    recall  f1-score   support

         LOC       0.94      0.96      0.95      1951
         ORG       0.89      0.88      0.88       984
         PER       0.97      0.98      0.98       884

   micro avg       0.94      0.94      0.94      3819
   macro avg       0.94      0.94      0.94      3819
weighted avg       0.94      0.94      0.94      3819

Epoch 2/3
-------------------------------
loss: 0.033449: 100%|███████| 5216/5216 [06:10<00:00, 14.08it/s]
100%|█████████████████████████| 580/580 [00:23<00:00, 24.27it/s]
              precision    recall  f1-score   support

         LOC       0.96      0.96      0.96      1951
         ORG       0.91      0.92      0.91       984
         PER       0.99      0.98      0.99       884

   micro avg       0.95      0.96      0.95      3819
   macro avg       0.95      0.96      0.95      3819
weighted avg       0.95      0.96      0.96      3819

Epoch 3/3
-------------------------------
loss: 0.024659: 100%|███████| 5216/5216 [06:10<00:00, 14.09it/s]
100%|█████████████████████████| 580/580 [00:23<00:00, 24.25it/s]
              precision    recall  f1-score   support

         LOC       0.97      0.97      0.97      1951
         ORG       0.93      0.92      0.92       984
         PER       0.99      0.98      0.98       884

   micro avg       0.96      0.96      0.96      3819
   macro avg       0.96      0.95      0.96      3819
weighted avg       0.96      0.96      0.96      3819

Done!
```

### 保存模型

实际应用中，我们会根据模型在验证集上的性能来调整超参数以及选出最好的模型，最后将选出的模型应用于测试集以评估最终的性能。因此，我们首先在上面的验证/测试循环中返回 seqeval 库计算出的指标，然后在每一个 Epoch 中根据指标中的宏/微 F1 值 (macro-F1/micro-F1) 保存在验证集上最好的模型：

```python
def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1)
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in y]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, y)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    return classification_report(
      true_labels, 
      true_predictions, 
      mode='strict', 
      scheme=IOB2, 
      output_dict=True
    )

total_loss = 0.
best_macro_f1 = 0.
best_micro_f1 = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    metrics = test_loop(valid_dataloader, model)
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    save_weights = False
    if valid_macro_f1 > best_macro_f1:
        best_macro_f1 = valid_macro_f1
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
        )
        save_weights = True
    if valid_micro_f1 > best_micro_f1:
        best_micro_f1 = valid_micro_f1
        if not save_weights: 
            print('saving new weights...\n')
            torch.save(
                model.state_dict(), 
                f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
            )
print("Done!")
```

```
Using cuda device

Epoch 1/3
-------------------------------
loss: 0.051204: 100%|███████| 5216/5216 [05:59<00:00, 14.52it/s]
100%|█████████████████████████| 580/580 [00:23<00:00, 24.63it/s]
              precision    recall  f1-score   support

         LOC       0.94      0.96      0.95      1951
         ORG       0.89      0.88      0.88       984
         PER       0.97      0.98      0.98       884

   micro avg       0.94      0.94      0.94      3819
   macro avg       0.94      0.94      0.94      3819
weighted avg       0.94      0.94      0.94      3819

saving new weights...

Epoch 2/3
-------------------------------
loss: 0.033449: 100%|███████| 5216/5216 [06:06<00:00, 14.24it/s]
100%|█████████████████████████| 580/580 [00:23<00:00, 24.76it/s]
              precision    recall  f1-score   support

         LOC       0.96      0.96      0.96      1951
         ORG       0.91      0.92      0.91       984
         PER       0.99      0.98      0.99       884

   micro avg       0.95      0.96      0.95      3819
   macro avg       0.95      0.96      0.95      3819
weighted avg       0.95      0.96      0.96      3819

saving new weights...

Epoch 3/3
-------------------------------
loss: 0.024659: 100%|███████| 5216/5216 [06:06<00:00, 14.24it/s]
100%|█████████████████████████| 580/580 [00:23<00:00, 24.69it/s]
              precision    recall  f1-score   support

         LOC       0.97      0.97      0.97      1951
         ORG       0.93      0.92      0.92       984
         PER       0.99      0.98      0.98       884

   micro avg       0.96      0.96      0.96      3819
   macro avg       0.96      0.95      0.96      3819
weighted avg       0.96      0.96      0.96      3819

saving new weights...
```

可以看到，随着训练的进行，验证集上无论是宏 F1 值还是微 F1值都在不断提升。因此，3 轮 Epoch 训练结束后，会在目录下保存 3 个模型权重：

```
epoch_1_valid_macrof1_93.656_microf1_93.904_weights.bin
epoch_2_valid_macrof1_95.403_microf1_95.493_weights.bin
epoch_3_valid_macrof1_95.710_microf1_95.877_weights.bin
```

至此，我们手工构建的 NER 模型的训练过程就完成了，完整的训练代码如下：

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import random
import os
import numpy as np

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

categories = set()

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, tags = '', []
                for i, c in enumerate(line.split('\n')):
                    word, tag = c.split(' ')
                    sentence += word
                    if tag[0] == 'B':
                        tags.append([i, i, word, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag[0] == 'I':
                        tags[-1][1] = i
                        tags[-1][2] += word
                Data[idx] = {
                    'sentence': sentence, 
                    'tags': tags
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = PeopleDaily('china-people-daily-ner-corpus/example.train')
valid_data = PeopleDaily('china-people-daily-ner-corpus/example.dev')
test_data = PeopleDaily('china-people-daily-ner-corpus/example.test')

id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

def collote_fn(batch_samples):
    batch_sentence, batch_tags  = [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_tags.append(sample['tags'])
    batch_inputs = tokenizer(
        batch_sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[s_idx][0] = -100
        batch_label[s_idx][len(encoding.tokens())-1:] = -100
        for char_start, char_end, _, tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, len(id2label))

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        logits = self.classifier(bert_output.last_hidden_state)
        return logits

model = NeuralNetwork().to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1)
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in y]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, y)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    return classification_report(
      true_labels, 
      true_predictions, 
      mode='strict', 
      scheme=IOB2, 
      output_dict=True
    )

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_macro_f1 = 0.
best_micro_f1 = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    metrics = test_loop(valid_dataloader, model)
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    save_weights = False
    if valid_macro_f1 > best_macro_f1:
        best_macro_f1 = valid_macro_f1
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
        )
        save_weights = True
    if valid_micro_f1 > best_micro_f1:
        best_micro_f1 = valid_micro_f1
        if not save_weights: 
            print('saving new weights...\n')
            torch.save(
                model.state_dict(), 
                f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
            )
print("Done!")
```

## 测试模型

在本小节中，我们加载验证集上性能最优的模型权重，汇报其在测试集上的性能，并且将模型的预测结果保存到文件中。

### 处理模型的输出

正如我们前面所说，模型的输出是一个由预测张量组成的列表，每个张量对应一个 token 的实体类别预测结果，只需要在输出的 logits 值上运用 softmax 函数就可以获得实体类别的预测概率。

与[《快速分词器》](/2022/03/08/transformers-note-5.html)中介绍的处理方式类似，我们首先从输出中取出"B-"或"I-"开头的 token，然后将这些 token 组合成实体，最后将实体对应的所有 token 的平均概率值作为实体的概率。下面我们以处理单个句子为例，加载训练好的 NER 模型来识别句子中的实体：

```python
sentence = '日本外务省3月18日发布消息称，日本首相岸田文雄将于19至21日访问印度和柬埔寨。'

model.load_state_dict(torch.load('epoch_3_valid_macrof1_96.169_microf1_96.258_weights.bin', map_location=torch.device(device)))
model.eval()
results = []
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    pred = model(inputs)
    probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].tolist()
    predictions = pred.argmax(dim=-1)[0].tolist()

    pred_label = []
    inputs_with_offsets = tokenizer(sentence, return_offsets_mapping=True)
    tokens = inputs_with_offsets.tokens()
    offsets = inputs_with_offsets["offset_mapping"]

    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) and 
                id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(pred_label)
```

```
[
 {'entity_group': 'ORG', 'score': 0.999188220500946, 'word': '日本外务省', 'start': 0, 'end': 5}, 
 {'entity_group': 'LOC', 'score': 0.9991478621959686, 'word': '日本', 'start': 16, 'end': 18}, 
 {'entity_group': 'PER', 'score': 0.9995187073945999, 'word': '岸田文雄', 'start': 20, 'end': 24}, 
 {'entity_group': 'LOC', 'score': 0.999652773141861, 'word': '印度', 'start': 34, 'end': 36}, 
 {'entity_group': 'LOC', 'score': 0.9995155533154806, 'word': '柬埔寨', 'start': 37, 'end': 40}
]
```

可以看到模型成功地将"日本外务省"识别为组织 (ORG)，将"岸田文雄"识别为人物 (PER)，将"日本"、"印度"、"柬埔寨"识别为地点 (LOC)。

### 保存预测结果

最后，我们对上面的代码进行扩展以处理整个测试集，不仅像之前验证/测试循环中那样评估模型在测试集上的性能，并且将模型的预测结果以 json 的形式存储到文件中：

```python
import json

model.load_state_dict(torch.load('epoch_3_valid_macrof1_96.169_microf1_96.258_weights.bin', map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    true_labels, true_predictions = [], []
    for X, y in tqdm(test_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        predictions = pred.argmax(dim=-1)
        true_labels += [[id2label[int(l)] for l in label if l != -100] for label in y]
        true_predictions += [
            [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, y)
        ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    results = []
    print('predicting labels...')
    for idx in tqdm(range(len(test_data))):
        example = test_data[idx]
        inputs = tokenizer(example['sentence'], truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        pred = model(inputs)
        probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].tolist()
        predictions = pred.argmax(dim=-1)[0].tolist()

        pred_label = []
        inputs_with_offsets = tokenizer(example['sentence'], return_offsets_mapping=True)
        tokens = inputs_with_offsets.tokens()
        offsets = inputs_with_offsets["offset_mapping"]

        idx = 0
        while idx < len(predictions):
            pred = predictions[idx]
            label = id2label[pred]
            if label != "O":
                label = label[2:] # Remove the B- or I-
                start, end = offsets[idx]
                all_scores = [probabilities[idx][pred]]
                # Grab all the tokens labeled with I-label
                while (
                    idx + 1 < len(predictions) and 
                    id2label[predictions[idx + 1]] == f"I-{label}"
                ):
                    all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                    _, end = offsets[idx + 1]
                    idx += 1

                score = np.mean(all_scores).item()
                word = example['sentence'][start:end]
                pred_label.append(
                    {
                        "entity_group": label,
                        "score": score,
                        "word": word,
                        "start": start,
                        "end": end,
                    }
                )
            idx += 1
        results.append(
            {
                "sentence": example['sentence'], 
                "pred_label": pred_label, 
                "true_label": example['tags']
            }
        )
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for exapmle_result in results:
            f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
```

```
Using cuda device

evaluating on test set...
100%|█████████████████████████| 1159/1159 [00:46<00:00, 24.90it/s]
              precision    recall  f1-score   support

         LOC       0.95      0.95      0.95      3658
         ORG       0.91      0.92      0.92      2185
         PER       0.98      0.97      0.98      1864

   micro avg       0.95      0.95      0.95      7707
   macro avg       0.95      0.95      0.95      7707
weighted avg       0.95      0.95      0.95      7707

predicting labels...
100%|█████████████████████████| 4636/4636 [01:03<00:00, 73.23it/s]
```

可以看到，模型最终在测试集上的宏/微 F1 值都达到 95% 左右。考虑到我们只使用了基础版本的 BERT 模型，并且只训练了 3 轮，这已经是一个不错的结果了。

我们打开保存预测结果的 test_data_pred.json 文件，每一行对应一个样本：

```
{
 "sentence": "我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。", 
 "pred_label": [
     {"entity_group": "LOC", "score": 0.9954637885093689, "word": "欧", "start": 15, "end": 16}, 
     {"entity_group": "LOC", "score": 0.9948422312736511, "word": "美", "start": 16, "end": 17}, 
     {"entity_group": "LOC", "score": 0.9960285425186157, "word": "港", "start": 18, "end": 19}, 
     {"entity_group": "LOC", "score": 0.9940919280052185, "word": "台", "start": 19, "end": 20}
 ], 
 "true_label": [
     [15, 15, "欧", "LOC"], 
     [16, 16, "美", "LOC"], 
     [18, 18, "港", "LOC"], 
     [19, 19, "台", "LOC"]
 ]
}
...
```

可以看到，其中 `sentence` 字段对应于原文，`pred_label` 字段对应于预测出的实体（包括实体类型、预测概率、实体文本、原文位置），`true_label` 对应于标注的实体信息。

至此，我们使用 Transformers 库进行 NER 任务就全部完成了。

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[3]](https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py) Bert4Keras 库中文 NER 实现