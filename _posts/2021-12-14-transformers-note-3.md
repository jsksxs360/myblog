---
layout: article
title: Hugging Face 的 Transformers 库快速入门（三）：必要的 Pytorch 知识
tags:
    - Pytorch
    - 机器学习
mathjax: false
sidebar:
  nav: transformers-note
---

在上一篇[《Hugging Face 的 Transformers 库快速入门（二）：模型与分词器》](/2021/12/11/transformers-note-2.html)中，我们介绍了 Transformers 库中的 `Model` 类和 `Tokenizers` 类，尤其是如何运用分词器对文本进行编码。

我们都知道，Transformers 库是建立在 Pytorch 框架之上（Tensorflow 的版本功能并不完善），虽然官方宣称学习 Transformers 库并不需要  Pytorch 的知识，但是在之后的模型训练（微调）等环节，实际上我们还是需要通过 Pytorch 提供的 `DataLoader` 类来加载数据、使用 Pytorch 的优化器对模型参数进行调整等等。

因此，文本将介绍一些我们后续会使用到的 Pytorch 类，尽可能让大家不需要去系统地学习 Pytorch 就可以上手使用 Transformers 库建立模型。

## 加载数据

Pytorch 提供了 `DataLoader` 和 `Dataset` 类（或者 `IterableDataset` 类）专门用于处理数据，它们既可以加载 Pytorch 自带的数据集，也可以加载我们自己的数据。其中：

- `Dataset` 类（或者 `IterableDataset` 类）负责存储样本以及它们对应的标签；
- `DataLoader` 类负责迭代地访问 `Dataset` （或者 `IterableDataset` ）中的样本。

### Dataset

数据集负责存储数据样本以及对应的标签，所有的数据集，无论是 Pytorch 自带的，或者是我们自定义的，都必须继承自 `Dataset` 或 `IterableDataset` 类。具体地，Pytorch 支持两种形式的数据集：

- Map-style 数据集

  Map-style 数据集继承自 `Dataset` 类，表示一个从样本索引 (indices/key) 到样本的 Map，而且索引还可以不是整数格式，这样我们就可以方便地通过 `dataset[idx]` 来访问指定索引的样本和对应的标签。Map-style 数据集必须实现 `__getitem__()` 函数，从而根据指定的 key 值返回对应的数据样本。一般还会实现 `__len__()` 函数，用于返回数据集的大小。

  > 注意：`DataLoader` 在默认情况下会创建一个生成整数索引的索引采样器 (sampler)，用于遍历数据集。因此，如果我们加载的是一个非整数索引的 Map-style 数据集，还需要手工地定义对应的采样器。

- Iterable-style 数据集

  Iterable-style 数据集继承自 `IterableDataset`，表示可迭代的数据集，它可以通过 `iter(dataset)` 以数据流 (steam) 的形式访问，尤其适用于访问非常巨大的数据集或者远程服务器产生的数据。 Iterable-style 数据集必须实现 `__iter__()` 函数，用于返回一个样本迭代器 (iterator)。

  > 注意：如果在 `DataLoader`  中开启多进程，即 num_workers > 0，那么在加载 Iterable-style 数据集时必须进行专门的设置，否则会重复访问样本。例如：
  >
  > ```python
  > from torch.utils.data import IterableDataset, DataLoader
  > 
  > class MyIterableDataset(IterableDataset):
  >     def __init__(self, start, end):
  >         super(MyIterableDataset).__init__()
  >         assert end > start
  >         self.start = start
  >         self.end = end
  > 
  >     def __iter__(self):
  >         return iter(range(self.start, self.end))
  > 
  > ds = MyIterableDataset(start=3, end=7) # [3, 4, 5, 6]
  > # Single-process loading
  > print(list(DataLoader(ds, num_workers=0)))
  > # Directly doing multi-process loading
  > print(list(DataLoader(ds, num_workers=2)))
  > ```
  >
  > ```
  > [tensor([3]), tensor([4]), tensor([5]), tensor([6])]
  > 
  > [tensor([3]), tensor([3]), tensor([4]), tensor([4]), tensor([5]), tensor([5]), tensor([6]), tensor([6])]
  > ```
  >
  > 可以看到，当在 DataLoader 中采用 2 个进程时，由于每个进程都获取到了独立的数据集的拷贝，因此会重复访问每一个样本。要避免这种情况，就需要在 DataLoader 中设置 `worker_init_fn` 来自定义每一个进程的数据集拷贝：
  >
  > ```python
  > from torch.utils.data import get_worker_info
  > 
  > def worker_init_fn(worker_id):
  >     worker_info = get_worker_info()
  >     dataset = worker_info.dataset  # the dataset copy in this worker process
  >     overall_start = dataset.start
  >     overall_end = dataset.end
  >     # configure the dataset to only process the split workload
  >     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
  >     worker_id = worker_info.id
  >     dataset.start = overall_start + worker_id * per_worker
  >     dataset.end = min(dataset.start + per_worker, overall_end)
  >     
  > # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
  > print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
  > # With even more workers
  > print(list(DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
  > ```
  >
  > ```
  > [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
  > 
  > [tensor([3]), tensor([4]), tensor([5]), tensor([6])]
  > ```

下面我们以加载一个图像分类数据集为例，看看如何具体地创建一个自定义的 Map-style 数据集：

```python
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

可以看到，我们具体实现了 `__init__()`、`__len__()` 和 `__getitem__()` 三个函数：`__init__()` 函数负责初始化数据集参数，这里设置了图像的存储目录、图像的标签（通过读取标签 csv 文件）以及样本和标签的数据转换函数；`__len__()` 函数返回数据集中样本的个数；`__getitem__()` 函数是 Map-style 数据集的核心，负责根据给定的索引 `idx` 返回数据集中的对应样本，这里会根据索引分别从目录下通过 `read_image` 读取图片和从 csv 文件中读取图片标签，并且返回处理后的图像和标签。

### DataLoaders

前面介绍的 `Dataset` 库（或者 `IterableDataset` 库）将数据集中所有的样本和对应的标签整理了起来，并且提供了一种按照索引访问某个样本的方式。不过在实际训练（微调）模型时，我们通常是先将数据集切分为很多的 minibatches，然后按批 (batch) 将样本送入模型，并且循环这一过程，每一个完整遍历所有样本的循环称为一个 epoch，并且我们通常会在每次 epoch 循环时，打乱样本顺序以避免过拟合。

Pytorch 提供了 `DataLoader` 类，专门用于处理这些操作，除了最基础的 `dataset` 和 `batch_size` 参数分别传入数据集和 batch 大小以外，还有以下这些常用的参数：

- `shuffle`：是否打乱数据集；
- `sampler`：采样器，也就是一个索引上的迭代器；
- `collate_fn`：批处理函数，用于对采样出的 batch 中的样本进行处理（例如我们前面提过的 Padding 操作）。

例如，我们按照 batch=64 遍历 Pytorch 自带的用于图像分类的 FashionMNIST 数据集（每个样本是一张 28x28 的灰度图，以及一个 10 分类的标签），并且打乱数据集：

```python
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
print(img.shape)
print(f"Label: {label}")
```

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
torch.Size([28, 28])
Label: 8
```

**数据加载顺序和 `Sampler` 类**

对于 Iterable-style 数据集来说，数据的加载顺序直接由用户控制，用户可以精确地控制每一个 batch 中返回的样本，因此不需要使用 `Sampler` 类。

对于 Map-style 数据集来说，由于索引可能不是整数，因此我们可以通过 `Sampler` 对象来设置加载时的索引序列，也就是设置一个索引上的迭代器。基于 `shuffle` 参数的值，DataLoader 会自动创建一个顺序或乱序的 sampler，我们也可以通过 `sampler` 参数传入一个自定义的 Sampler 对象，负责产生下一个要获取的索引。

常见的 Sampler 对象包括序列采样器 `SequentialSampler` 和随机采样器 `RandomSampler`，它们都通过传入待采样的数据集来创建：

```python
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_sampler = RandomSampler(training_data)
test_sampler = SequentialSampler(test_data)

train_dataloader = DataLoader(training_data, batch_size=64, sampler=train_sampler)
test_dataloader = DataLoader(test_data, batch_size=64, sampler=test_sampler)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")
```

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
```

**使用批处理函数 `collate_fn`**

批处理函数 `collate_fn` 每次会对一个 batch 对应的样本列表进行处理。默认的 `collate_fn` 会进行如下操作：

- 添加一个新维度作为 batch 维；
- 自动地将 NumPy 数组和 Python 数值转换为 PyTorch 张量；
- 保留原始的数据结构，例如每一个是字典的话，它会输出一个包含同样键 (key) 的字典，但是将值 (value) 替换为 batched 张量（如何可以转换的话）。

例如，如果每一个样本是一个 3 通道的图像和一个整数型的类别标签，即 `(image, class_index)`，那么默认的 `collate_fn` 会将这样的一个元组列表转换为一个包含 batched 图像张量和 batched 类别标签张量的元组。

我们可以使用自定义的 `collate_fn` 来获取特定的 batching，例如前面我们介绍过的 padding 序列，因为我们只需对每一个 batch 中的样本进行 padding 操作。

## 训练模型

Pytorch 所有的模块（层）都是 `nn.Module` 的子类，神经网络模型本身就是一个模块，它还包含了很多其他的模块。

### 构建模型

还是以前面加载的 FashionMNIST 数据库为例，我们首先构建一个神经网络来完成该图像分类任务。它同样继承自 `nn.Module` 类，通过 `__init__()` 初始化模型中使用到的层，在 `forward()` 中定义模型的具体操作，例如：

```python
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

```
Using cpu device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

可以看到，模型首先将二维图像通过 `Flatten` 层压成一维向量，然后经过两个带有 `ReLU` 激活函数的全连接隐藏层，最后送入到一个包含 10 个神经元的分类器以完成 10 分类任务。

> 注意：虽然 `forward()` 定义了模型的操作，但是如果要使用模型对数据进行处理，应该直接将数据送入模型，而不是调用 `forward()`。

这里，如果将图像送入我们上面构建的模型会输出一个 10 维的张量，每一维对应一个分类的预测值，与使用 Transformers 库的 pipelines 时一样，这里输出的是 logits 值，我们需要在模型的输出上再接一个 Softmax 层来输出最终的概率值。下面我们随机创建包含三个二维图像的 minibatch 来进行预测：

```python
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

X = torch.rand(3, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

```
Using cpu device
Predicted class: tensor([1, 1, 1])
```

### 优化模型参数

准备好数据，搭建好模型之后，我们就可以开始训练、验证和测试模型了。正如前面说的那样，模型训练是一个迭代的过程，每一轮迭代（称为 epoch）模型都会对数据进行预测，然后基于预测的结果计算损失 (loss)，并求 loss 对每一个模型参数的偏导，最后使用优化器通过梯度下降优化所有的参数。

> - **损失函数 (Loss function)** 用于度量预测值与标准答案之间差异，模型的训练过程就是最小化损失函数。Pytorch 实现了很多常见的[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)，可以直接调用，例如用于回归任务的均方误差 (Mean Square Error) [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)、用于分类任务的负对数似然 (Negative Log Likelihood)  [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)、同时结合了 `nn.LogSoftmax` 和 `nn.NLLLoss` 的交叉熵损失 (Cross Entropy)  [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) 等。
> - **优化器 (Optimization)** 使用特定的优化算法（例如经典的随机梯度下降算法）通过在每一个训练阶段减少模型的损失来调整模型的参数。Pytorch 实现了很多不同的[优化器](https://pytorch.org/docs/stable/optim.html)，例如 SGD、ADAM、RMSProp 等。

在每一轮迭代 (Epoch) 中，实际上包含了两个步骤：

- **训练循环 (The Train Loop)**：在训练集上进行迭代，尝试收敛到最佳的参数；
- **验证/测试循环 (The Validation/Test Loop)**：在测试/验证集上进行迭代以检查模型性能有没有提升。

具体地，在训练循环中，优化器通过以下三个步骤进行优化：

- 调用 `optimizer.zero_grad()` 重设模型的参数的梯度。梯度在默认情况下会进行累加，为了防止重复计算，在每一轮 Epoch 迭代前都需要进行清零；
- 通过 `loss.backwards()` 反向传播预测的损失，即计算损失对每一个参数的偏导；
- 调用 `optimizer.step()` 根据梯度调整模型的参数。

在这里，我们选择交叉熵作为损失函数，选择 AdamW 作为优化器。完整的训练循环和测试循环实现如下：

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

learning_rate = 1e-3
batch_size = 64
epochs = 3

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

```
Epoch 1
-------------------------------
loss: 0.607171  [ 6400/60000]
loss: 0.568096  [12800/60000]
loss: 0.349548  [19200/60000]
loss: 0.664445  [25600/60000]
loss: 0.231511  [32000/60000]
loss: 0.349752  [38400/60000]
loss: 0.631870  [44800/60000]
loss: 0.413865  [51200/60000]
loss: 0.303703  [57600/60000]
Test Error: 
 Accuracy: 84.8%, Avg loss: 0.418481 

Epoch 2
-------------------------------
loss: 0.329605  [ 6400/60000]
loss: 0.429682  [12800/60000]
loss: 0.292489  [19200/60000]
loss: 0.597905  [25600/60000]
loss: 0.173213  [32000/60000]
loss: 0.272843  [38400/60000]
loss: 0.564452  [44800/60000]
loss: 0.330194  [51200/60000]
loss: 0.262742  [57600/60000]
Test Error: 
 Accuracy: 84.9%, Avg loss: 0.405600 

Epoch 3
-------------------------------
loss: 0.290210  [ 6400/60000]
loss: 0.342479  [12800/60000]
loss: 0.235519  [19200/60000]
loss: 0.433092  [25600/60000]
loss: 0.159719  [32000/60000]
loss: 0.245311  [38400/60000]
loss: 0.478349  [44800/60000]
loss: 0.248536  [51200/60000]
loss: 0.213696  [57600/60000]
Test Error: 
 Accuracy: 85.6%, Avg loss: 0.386434 

Done!
```

可以看到，通过 3 轮迭代 (Epoch)，模型在训练集上的损失逐步下降，证明优化器成功地对模型参数进行了调整，并且在测试集上的准确率也随着迭代出现了上升，说明训练过程是有效的（并没有出现过拟合）。

> **注意：**我们一定要在预测之前调用 `model.eval()` 方法，以将 dropout 层和 batch normalization 层设置为评估模式，否则会产生不一致的预测结果。

## 保存及加载模型

在之前的文章中，我们介绍了 Transformers 库提供的模型 `Model` 类的保存以及加载方法，但是如果我们仅仅将预训练模型作为编码器，即只作为我们完整模型中的一个模块，那么完整模型就是一个自定义的 Pytorch 模型，它的保存和加载就必须使用 Pytorch 预设的接口。

### 保存和加载模型权重

Pytorch 模型会将所有的模型参数存储在一个状态字典 (state dictionary) 中，可以通过 `Model.state_dict()` 加载。Pytorch 通过 `torch.save()` 进行存储：

```python
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

为了加载保存的权重，我们首先需要创建一个结构完全相同的模型实例，然后通过 `Model.load_state_dict()` 函数进行加载：

```python
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### 保存和加载完整模型

采用上面的方式存储模型权重后，我们首先需要构建一个完全相同结构的模型实例来承接这些权重，因为保存的文件中并没有存储模型的结构。因此，如果我们希望在存储权重的同时，也一起保存模型的结构，就直接将整个模型传给 `torch.save()` ：

```python
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model, 'model.pth')
```

这样的话，就可以直接从保存的文件中加载整个模型（包括权重和结构）：

```python
model = torch.load('model.pth')
```

## 参考

[[1]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[2]](https://pytorch.org/docs/stable/) Pytorch 在线教程

