---
layout: article
title: Hugging Face 的 Transformers 库快速入门（四）：微调预训练模型
tags:
    - Transformers
    - NLP
    - Pytorch
    - 机器学习
mathjax: true
sidebar:
  nav: transformers-note
---

在上一篇[《Hugging Face 的 Transformers 库快速入门（三）：必要的 Pytorch 知识》](/2021/12/14/transformers-note-3.html)中，我们介绍了训练模型必要的 Pytorch 知识， 本文我们将正式上手训练自己的模型。

本文以基础的句子对分类任务为例，展示如何微调一个预训练模型，并且保存验证集上最好的模型权重。

## 加载数据集

我们选择蚂蚁金融语义相似度数据集 [AFQMC](https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip) 作为语料，在其之上训练一个同义句判断模型：每次输入两个句子，判断它们是否为同义句。AFQMC 提供了官方的数据划分，训练集/验证集/测试集分别包含 34334 / 4316 / 3861 个句子对，对应的标签 0 表示是非同义句，1 表示是同义句：

```
{"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}
```

### Dataset

就像我们在上一篇中介绍的那样，Pytorch 通过 `Dataset` 类和 `DataLoader` 类处理数据集和加载数据构建 batch。因此我们首先需要编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签：

```python
from torch.utils.data import Dataset
import json

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('afqmc_public/train.json')
valid_data = AFQMC('afqmc_public/dev.json')

print(train_data[0])
```

```
{'sentence1': '蚂蚁借呗等额还款可以换成先息后本吗', 'sentence2': '借呗有先息到期还本吗', 'label': '0'}
```

由于 AFQMC 数据库中的样本是以 json 格式存储的，因此我们在 `__init__()` 中使用 `json` 库按行读取样本，并且以行号作为索引构建数据集。每一个样本都以字典形式保存，分别以 `sentence1`、`sentence2` 和 `label` 为键存储句子对和对应的标签。

当然了，如果数据集非常巨大，难以一次全部加载到内存中，我们也可以继承 `IterableDataset` 类构建 Iterable-style 数据集：

```python
from torch.utils.data import IterableDataset
import json

class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

train_data = IterableAFQMC('afqmc_public/train.json')
print(next(iter(train_data)))
```

```
{'sentence1': '蚂蚁借呗等额还款可以换成先息后本吗', 'sentence2': '借呗有先息到期还本吗', 'label': '0'}
```

### DataLoader

创建好数据集之后，就需要通过 `DataLoader` 库来按批 (batch) 加载数据，将样本组织成模型可以接受的输入格式。对于 NLP 任务来说，这个环节就是对一个 batch 中的句子（这里是“句子对”）按照预训练模型的要求进行编码（包括 Padding、截断等操作），正如我们在上一篇文章中说的那样，通过在 DataLoader 中设置批处理函数 `collate_fn` 来实现。

> 这种不是在整个数据集上，而是只在一个 batch 内进行的 Padding 操作被称为动态 Padding (Dynamic padding)，Hugging Face 也提供了 `DataCollatorWithPadding` 类来进行，如果感兴趣可以自行[了解](https://huggingface.co/course/chapter3/2?fw=pt#dynamic-padding)。

这里我们以中文 BERT 模型为例，通过加载对应的分词器，对样本中的句子对进行编码：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('afqmc_public/train.json')
valid_data = AFQMC('afqmc_public/dev.json')

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
```

```
batch_X shape: {
    'input_ids': torch.Size([4, 39]), 
    'token_type_ids': torch.Size([4, 39]), 
    'attention_mask': torch.Size([4, 39])
}
batch_y shape: torch.Size([4])

{'input_ids': tensor([
        [ 101, 5709, 1446, 5543, 3118,  802,  736, 3952, 3952, 2767, 1408,  102,
         3952, 2767, 1041,  966, 5543, 4500, 5709, 1446, 1408,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0],
        [ 101,  872, 8024, 2769, 6821, 5709, 1446, 4638, 7178, 6820, 2130,  749,
         8024, 6929, 2582,  720, 1357, 3867,  749,  102, 1963, 3362, 1357, 3867,
          749, 5709, 1446,  722, 1400, 8024, 1355, 4495, 4638, 6842, 3621, 2582,
          720, 1215,  102],
        [ 101, 1963,  862, 2990, 7770,  955, 1446,  677, 7361,  102, 6010, 6009,
          955, 1446, 1963,  862,  712, 1220, 2990, 7583, 2428,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0],
        [ 101, 2582, 3416, 2990, 7770,  955, 1446, 7583, 2428,  102,  955, 1446,
         2990, 4157, 7583, 2428, 1416,  102,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0]]), 
 'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
tensor([1, 0, 1, 1])
```

可以看到，DataLoader 按照我们设置的 batch size，每次对 4 个样本进行编码，将句子对处理成  **[CLS] sentence1 [SEP] sentence2 [SEP]**  的格式，并且将 token 序列 pad 到了相同的长度。同时，将标签也处理成了 Pytorch 的张量格式。

## 训练模型

### 构建模型

对于本文的句子对分类任务，可以使用我们前面介绍过的 `AutoModelForSequenceClassification` 类来完成，不过考虑到在大部分情况下，预训练模型仅仅被用作编码器，模型中还会包含很多自定义的模块，因此本文采用自己编写 Pytorch 模型的方式来完成：首先利用 Transformers 库加载 BERT 模型，然后接一个全连接层完成分类：

```python
import torch
from torch import nn
from transformers import AutoModel

checkpoint = "bert-base-chinese"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_vectors)
        return logits

model = NeuralNetwork()
print(model)
```

```python
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
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

可以看到，我们构建的模型首先将输入送入到一个 BERT 模型中进行编码，将每一个 token 都编码为维度为 768 的张量，然后我们从输出序列中取出第一个 `[CLS]` token 的编码结果作为句子对的语义表示，送入到一个包含两个神经元的线性全连接层中完成分类。

为了确保模型的输出符合我们的预期，我们尝试将一个 Batch 的数据送入模型：

```python
outputs = model(batch_X)
print(outputs.shape)
```

```
torch.Size([4, 2])
```

可以看到模型输出了一个 $4 \times 2$ 的张量，符合我们的预期（每个样本输出 2 维的 logits 值，batch 内共 4 个样本）。

### 优化模型参数

正如我们在上一篇文章中介绍的那样，我们将每一轮 Epoch 分为训练循环和验证/测试循环。在训练循环中计算损失，优化模型的参数，在验证/测试循环中评估模型的性能。其实 Transformers 库同样实现了很多的优化器，相比 Pytorch 固定学习率的优化器，Transformers 库实现的优化器会随着训练过程逐步减小学习率（这通常会产生更好的效果）。例如我们前面使用过的 AdamW 优化器：

```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

默认情况下，优化器会线性衰减学习率，例如对于上面的例子，学习率会线性地从 $\text{5e-5}$ 降到 $0$。为了正确地定义学习率调度器，我们需要知道总的训练步数 (step)，它等于训练轮数 (Epoch number) 乘以 batch 的数量（也就是训练 dataloader 的大小）：

```python
from transformers import get_scheduler

epochs = 3
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```

```
25752
```

完整的训练过程如下所示：

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('afqmc_public/train.json')
valid_data = AFQMC('afqmc_public/dev.json')

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_vectors)
        return logits

model = NeuralNetwork().to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1)*len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")

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
    test_loop(valid_dataloader, model, mode='Valid')
print("Done!")
```

```
Using cuda device

Epoch 1/3
-------------------------------
loss: 0.553325: 100%|███████| 8584/8584 [09:27<00:00, 15.13it/s]
Valid Accuracy: 73.0%

Epoch 2/3
-------------------------------
loss: 0.504794: 100%|███████| 8584/8584 [09:17<00:00, 15.39it/s]
Valid Accuracy: 73.8%

Epoch 3/3
-------------------------------
loss: 0.454273: 100%|███████| 8584/8584 [09:27<00:00, 15.12it/s]
Valid Accuracy: 74.1%
```

在 Pytorch 框架下，如果使用 GPU/TPU 训练模型， 就必须通过 `to(device)` 操作将模型和所有的数据都送入到对应的设备中，如果使用 CPU 训练，则可以不进行该操作。上面的代码中，我们通过 `Model.to(device)` 将模型送入设备，在训练循环和测试循环中通过 `X, y = X.to(device), y.to(device)` 将数据张量送入设备。

幸运地是，Hugging Face 提供了 [Accelerator 库](https://github.com/huggingface/accelerate)可以自动地完成这一操作，它会检测代码的运行环境，并且初始化正确的分布式设置（方便在多 GPU 或 TPU 上运行），这样我们就不需要手动地将模型和输入送入设备了。这里我们只需对上面的代码进行小幅的修改：

- 将 dataloaders、模型和优化器送入到 `accelerator.prepare()` 中；
- 移除所有的 `to(device)` 操作；
- 将 `loss.backward()` 替换为 `accelerator.backward(loss)`。

```
+ from accelerate import Accelerator

+ accelerator = Accelerator()
- device = 'cuda' if torch.cuda.is_available() else 'cpu'

- model.to(device)
- X, y = X.to(device), y.to(device)

+ train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, valid_dataloader, model, optimizer
+ )

- loss.backward()
+ accelerator.backward(loss)
```

使用 Accelerator 库的完整代码如下：

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import json
from accelerate import Accelerator

accelerator = Accelerator()

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('afqmc_public/train.json')
valid_data = AFQMC('afqmc_public/dev.json')

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_vectors)
        return logits

model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1)*len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, valid_dataloader, model, optimizer
)

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
    test_loop(valid_dataloader, model, mode='Valid')
print("Done!")
```

> 当然，你也可以继续手工地  `to(device)`，不过需要将 device 修改为 `accelerator.device`。

### 保存和加载模型

在大多数情况下，我们还需要根据验证集上的表现来调整超参数以及选出最好的模型，最后再将选出的模型应用于测试集以评估性能。例如我们在测试循环时返回计算出的准确率，然后对上面的 Epoch 训练代码进行小幅的调整，以保存验证集上准确率最高的模型：

```python
def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")
```

```
Using cuda device

Epoch 1/3
-------------------------------
loss: 0.564382: 100%|███████| 8584/8584 [09:16<00:00, 15.42it/s]
Valid Accuracy: 70.8%

saving new weights...

Epoch 2/3
-------------------------------
loss: 0.515803: 100%|███████| 8584/8584 [09:18<00:00, 15.36it/s]
Valid Accuracy: 73.7%

saving new weights...

Epoch 3/3
-------------------------------
loss: 0.466284: 100%|███████| 8584/8584 [09:17<00:00, 15.39it/s]
Valid Accuracy: 73.6%

Done!
```

可以看到，随着训练的进行，在验证集上的准确率先升后降（70.8% -> 73.7% -> 73.6%）。因此，3 轮 Epoch 训练结束后，会在目录下保存前两轮的模型权重：

```
epoch_1_valid_acc_70.8_model_weights.bin
epoch_2_valid_acc_73.7_model_weights.bin
```

接下来，我们加载验证集上最优的模型权重，汇报其在测试集上的性能。由于 AFQMC 公布的测试集上并没有标签，无法评估性能，这里我们暂且用验证集代替进行演示：

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
from accelerate import Accelerator

accelerator = Accelerator()

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

test_data = AFQMC('afqmc_public/dev.json')

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_vectors)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load('epoch_2_valid_acc_73.7_model_weights.bin'))

test_dataloader, model= accelerator.prepare(
    test_dataloader, model
)

correct = 0
model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
correct /= len(test_dataloader.dataset)
print(f"Test Accuracy: {(100*correct):>0.1f}%\n")
```

```
Test Accuracy: 73.7%
```

这里我们只保存了模型的权重（并没有同时保存模型结构），因此首先需要实例化一个结构完全一样的模型，再通过 `model.load_state_dict()` 函数加载权重。最终在测试集（这里用了验证集）上的准确率为 73.7%，与前面汇报的一致，也验证了加载过程是正确的。

## 参考

[1] [HuggingFace 在线教程](https://huggingface.co/course/chapter1/1)  
[2] [Pytorch 官方文档](https://pytorch.org/docs/stable/)  
[3] [Pytorch 在线教程](https://pytorch.org/tutorials/beginner/basics/intro.html)



