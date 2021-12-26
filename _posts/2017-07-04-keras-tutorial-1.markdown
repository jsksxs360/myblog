---
layout: article
title: Keras 快速上手指南（上）：Keras 入门
tags:
    - Keras
mathjax: true
sidebar:
  nav: keras-note
---

## 什么是 Keras

Keras 是基于 Theano 或 TensorFlow 的一个深度学习框架，它的设计参考了 Torch，用 Python 语言编写，是一个高度模块化的神经网络库，支持 GPU 和 CPU。

### 安装 Keras

使用 Keras 前还需要安装 Numpy、Scipy 等 Python 包，建议直接安装 Python 科学计算环境 [Anaconda](https://www.continuum.io/downloads)，一步到位。然后直接通过 `pip install keras` 安装 Keras 就可以了，非常的方便。

### 在 Theano 和 TensorFlow 间切换

Keras 的底层库使用 Theano 或 TensorFlow，这两个库也称为 Keras 的后端。无论是 Theano 还是 TensorFlow，都是一个“符号式”的库，这也使得 Keras 的编程与传统的 Python 代码有所差别。笼统的说，符号主义的计算首先定义各种变量，然后建立一个“计算图”，计算图规定了各个变量之间的计算关系。建立好的计算图需要编译以确定其内部细节，但是此时的计算图只是一个“空壳子”，里面没有任何实际的数据，只有当你把需要运算的输入放进去后，才能在整个模型中形成数据流，从而形成输出值。Keras 的模型搭建形式就是这种方法，搭建好的 Keras 模型只是一个空壳子，只有实际生成可调用的函数后（K.function），输入数据，才会形成真正的数据流。

> 使用计算图的语言，如 Theano，以难以调试而闻名，当 Keras 的 Debug 进入 Theano 这个层次时，往往也令人头痛。但大多数的深度学习框架使用的都是符号计算这一套方法，因为符号计算能够提供关键的计算优化、自动求导等功能。

Keras 会根据环境自动设置后端为 Theano 或 TensorFlow，我们也可以通过修改 Keras 配置文件来设置。Keras 配置文件位于用户目录下的 .keras 目录中，名称为 **keras.json**：

```json
{
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "floatx": "float32",
    "image_data_format": "channels_last"
}
```

其中 `backend` 字段设定 Keras 的后端，这里选择的是 `tensorflow`，也可以设定为 `theano`。

### data_format

在如何表示一组彩色图片的问题上，Theano 和 TensorFlow 发生了分歧：Theano 会把 100 张 RGB 三通道的 16×32 彩色图表示为 (100,3,16,32)，第 0 维是样本维，代表样本的数目，第 1 维是通道维，代表颜色通道数，后面两个就是高和宽了，这种数据组织方法，称为 **channels_first**，即通道维靠前；而 TensorFlow 的表达形式是 (100,16,32,3)，即把通道维放在了最后，因而称为 **channels_last**。

Keras 默认的数据组织形式也在配置文件 keras.json 中规定，由 `image_data_format` 一项设定，也可在代码中通过 `K.image_data_format()` 函数返回，请在网络的训练和测试中保持维度顺序一致。对 2D 数据来说，channels_last 设定维度顺序为 `(rows,cols,channels)` 而 channels_first 设定维度顺序为 `(channels, rows, cols)`。对 3D 数据而言，channels_last 设定为 `(conv_dim1, conv_dim2, conv_dim3, channels)`，而 channels_first 则是 `(channels, conv_dim1, conv_dim2, conv_dim3)`。

## 一些基本概念

下面介绍几个使用 Keras 过程中经常会遇到的词汇：

### 张量

张量(tensor)可以看作是向量、矩阵的自然推广，用来表示广泛的数据类型。0 阶张量即标量，也就是一个数；1 阶张量就是一个向量；2 阶张量就是一个矩阵；3 阶张量可以称为一个立方体，具有 3 个颜色通道的彩色图片就是一个这样的立方体；把立方体摞起来就是 4 阶张量了，不同去想像 4 阶张量是什么样子，它就是个数学上的概念。

张量的阶数有时候也称为维度，或者轴，轴这个词翻译自英文 axis。譬如一个矩阵 [[1,2],[3,4]]，是一个 2 阶张量，有两个维度或轴，沿着第 0 个轴（为了与 python 的计数方式一致，维度和轴从 0 算起）你看到的是 [1,2]，[3,4] 两个向量，沿着第 1 个轴你看到的是 [1,3]，[2,4] 两个向量。要理解“沿着某个轴”是什么意思，不妨试着运行一下下面的代码：

```python
import numpy as np
a = np.array([[1,2],[3,4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)
print sum0
print sum1
```
### 函数式模型

在 Keras 0.x 中有两种模型，一种是 Sequential 模型，又称为序贯模型，也就是单输入单输出，一条路通到底，层与层之间只有相邻关系，没有跨层连接。这种模型编译速度快，操作上也比较简单。第二种模型称为 Graph，即图模型，这个模型支持多输入多输出，层与层之间想怎么连怎么连，但是编译速度慢。可以看到，Sequential 其实是 Graph 的一个特殊情况。

在 Keras 1 和 Keras 2 中，图模型被移除，从而增加了“functional model API”这个东西，更加强调了 Sequential 模型是特殊的一种。一般的模型就称为 Model。

由于 functional model API 在使用时利用的是“函数式编程”的风格，这里将其称为函数式模型。总而言之，只要这个东西接收一个或一些张量作为输入，然后输出的也是一个或一些张量，但不管它是什么，统统都叫做“模型”。

### batch

深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为**批梯度下降(Batch gradient descent)**。

另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为**随机梯度下降(stochastic gradient descent)**。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，达不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

为了克服两种方法的缺点，现在一般采用的是一种折中手段，**小批的梯度下降(mini-batch gradient decent)**，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，所以计算量也不是很大。基本上现在的梯度下降都是基于 mini-batch 的，所以 Keras 的模块中经常会出现 `batch_size`，就是指这个。

> Keras 中用的优化器 SGD 是 stochastic gradient descent 的缩写，但不代表是一个样本就更新一回，而是基于 mini-batch 的。

### epochs

简单说，epochs 指的就是训练过程中数据将被“轮询”多少次，就这样。

## 快速上手 Keras

Keras 的核心数据结构是“模型”，模型是一种组织网络层的方式。Keras 中主要的模型是 Sequential 模型，Sequential 是一系列网络层按顺序构成的栈。你也可以查看[函数式模型]()来学习建立更复杂的模型。

Sequential 模型如下：

```python
from keras.models import Sequential
model = Sequential()
```

将一些网络层通过 `.add()` 堆叠起来，就构成了一个模型：

```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```

完成模型的搭建后，我们需要使用 `.compile()` 方法来编译模型：

```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。Keras 的一个核心理念就是简明易用同时，保证用户对 Keras 的绝对控制力度，用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码。例如，我们使用自定义的 SGD 优化器：

```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

完成模型编译后，我们在训练数据上按 batch 进行一定次数的迭代来训练网络：

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

当然，我们也可以手动将一个个 batch 的数据送入网络中训练，这时候需要使用：

```python
model.train_on_batch(x_batch, y_batch)
```

随后，我们可以使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求：

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

或者，我们可以使用我们的模型，对新的数据进行预测：

```python
classes = model.predict(x_test, batch_size=128)
```

搭建一个问答系统、图像分类模型，或神经图灵机、word2vec 词嵌入器就是这么快。支撑深度学习的基本想法本就是简单的，现在让我们把它的实现也变的简单起来！

为了更深入的了解 Keras，接下来我们介绍一下 Sequntial 模型和函数式模型的使用方法。

## Sequntial 模型 

### 构建序贯(Sequential)模型

序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”。可以通过向 Sequential 模型传递一个 layer 的 list 来构造该模型：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
Dense(32, units=784),
Activation('relu'),
Dense(10),
Activation('softmax'),
])
```

也可以通过 `.add()` 方法一个个的将 layer 加入模型中：

```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
```

### 指定输入数据的 shape

模型需要知道输入数据的 shape，因此，Sequential 模型的第一层需要接受一个关于输入数据 shape 的参数，后面的各个层则可以自动的推导出中间数据的 shape，因此不需要为每个层都指定这个参数。有几种方法来为第一层指定输入数据的 shape：

- 传递一个 `input_shape` 的关键字参数给第一层，`input_shape` 是一个 tuple 类型的数据，其中也可以填入 `None`，如果填入 `None` 则表示此位置可能是任何正整数。数据的 batch 大小不应包含在其中。
- 有些 2D 层，如 `Dense`，支持通过指定其输入维度 `input_dim` 来隐含的指定输入数据 shape。一些 3D 的时域层支持通过参数 `input_dim` 和 `input_length` 来指定输入shape。
- 如果你需要为输入指定一个固定大小的 batch_size（常用于 stateful RNN 网络），可以传递 `batch_size` 参数到一个层中，例如你想指定输入张量的 batch 大小是 32，数据 shape 是 (6，8)，则你需要传递 `batch_size=32` 和 `input_shape=(6,8)`。

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```

```python
model = Sequential()
model.add(Dense(32, input_shape=784))
```

### 编译

在训练模型之前，我们需要通过 `compile` 来对学习过程进行配置。`compile` 接收三个参数：

- 优化器 optimizer：该参数可指定为已预定义的优化器名，如 `rmsprop`、`adagrad`，或一个 `Optimizer` 类的对象。
- 损失函数 loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如`categorical_crossentropy`、`mse`，也可以为一个损失函数。
- 指标列表 metrics：对分类问题，我们一般将该列表设置为 `metrics=['accuracy']`。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数。指标函数应该返回单个张量，或一个完成 `metric_name - > metric_value` 映射的字典。

```python
# 对于一个多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 对于一个二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 对于一个均方误差回归问题
model.compile(optimizer='rmsprop',
              loss='mse')

# 用户自定义指标列表
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

### 训练

Keras以Numpy数组作为输入数据和标签的数据类型。训练模型一般使用 `fit` 函数。下面是一些例子：

```python
# 对于一个单输入模型的二分类问题
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建虚假数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 训练模型, 以每批次32样本迭代数据
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# 对于一个单输入模型的10分类问题
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 创建虚假数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# 将标签转换为one-hot表示
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# 训练模型, 以每批次32样本迭代数据
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

## Sequntial 模型示例

### 基于多层感知器的 softmax 多分类：

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 创建虚假数据
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个含有64个隐单元的全连接层
# 在第一层，你必须指定预期的输入数据shape:
# 这里是一个20维的向量.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

### MLP的二分类：

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 创建虚假数据
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

### 类似VGG的卷积神经网络：

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 创建虚假数据
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 输入: 100x100的3通道图像 -> 张量 (100, 100, 3).
# 在每块3x3的区域应用32个卷积过滤器.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```

### 使用LSTM的序列分类

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 使用1D卷积的序列分类

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 用于序列分类的栈式LSTM

在该模型中，我们将三个 LSTM 堆叠在一起，是该模型能够学习更高层次的时域特征表示。开始的两层 LSTM 返回其全部输出序列，而第三层 LSTM 只返回其输出序列的最后一步结果，从而其时域维度降低（即将输入序列转换为单个向量）:

![1](/img/article/keras-tutorial-1/multi_layer_lstm.png)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 预期输入数据shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # 返回一个32维向量的序列
model.add(LSTM(32, return_sequences=True))  # 返回一个32维向量的序列
model.add(LSTM(32))  # 返回一个独立的32维的向量
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 创建虚假的训练数据
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# 创建虚假的验证数据
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```

### 采用stateful LSTM的相同模型

stateful LSTM 的特点是，在处理过一个 batch 的训练数据后，其内部状态（记忆）会被作为下一个 batch 的训练数据的初始状态。状态 LSTM 使得我们可以在合理的计算复杂度内处理较长序列：

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# 预期输入批shape: (batch_size, timesteps, data_dim)
# 注意我们需要提供完整的 batch_input_shape 因为网络是有状态的.
# 第k批中的样本i跟踪第k-1批中的样本i.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 创建虚假的训练数据
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# 创建虚假的验证数据
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```

## Functional 模型

在 Keras 2 里我们将 Functional 译为“函数式”，对函数式编程有所了解的同学应能够快速 get 到该类模型想要表达的含义。函数式模型称作 Functional，但它的类名是 Model，因此我们有时候也用 Model 来代表函数式模型。

Keras 函数式模型接口是用户定义多输出模型、非循环有向模型或具有共享层的模型等复杂模型的途径。一句话，只要你的模型不是类似 VGG 一样一条路走到黑的模型，或者你的模型需要多于一个的输出，那么你总应该选择函数式模型。函数式模型是最广泛的一类模型，序贯模型(Sequential)只是它的一种特殊情况。

让我们从简单一点的模型开始：

### 第一个模型：全连接网络

Sequential 模型当然是实现全连接网络的最好方式，但我们从简单的全连接网络开始，有助于我们学习这部分的内容。在开始前，有几个概念需要澄清：

- 层对象接受张量为参数，返回一个张量。
- 输入是张量，输出也是张量的一个框架就是一个模型，通过 `Model` 定义。
- 这样的模型可以被像 Keras 的 `Sequential` 一样被训练。

```python
from keras.layers import Input, Dense
from keras.models import Model

# 这会返回一个张量
inputs = Input(shape=(784,))

# 层对象接受张量为参数，返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这会创建一个模型，包含输入层和三个Dense层
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # 开始训练
```

### 所有的模型都是可调用的，就像层一样

利用函数式模型的接口，我们可以很容易的重用已经训练好的模型：你可以把模型当作一个层一样，通过提供一个 tensor 来调用它。注意当你调用一个模型时，你不仅仅重用了它的结构，也重用了它的权重。

```python
x = Input(shape=(784,))
# 这会返回我们之前定义的 10-way softmax.
y = model(x)
```

这种方式可以允许你快速的创建能处理序列信号的模型，你可以很快将一个图像分类的模型变为一个对视频分类的模型，只需要一行代码：

```python
from keras.layers import TimeDistributed

# 20个时间步序列的输入张量，每个包含一个784维的向量
input_sequences = Input(shape=(20, 784))

# 这会应用我们之前定义的模型到输入序列的每一个时间步
# 之前模型的输出是一个 10-way softmax,
# 所以下面这个层的输出会是一个含有20个10维向量的序列
processed_sequences = TimeDistributed(model)(input_sequences)
```

### 多输入和多输出模型

使用函数式模型的一个典型场景是搭建多输入、多输出的模型。

考虑这样一个模型。我们希望预测 Twitter 上一条新闻会被转发和点赞多少次。模型的主要输入是新闻本身，也就是一个词语的序列。但我们还可以拥有额外的输入，如新闻发布的日期等。这个模型的损失函数将由两部分组成，辅助的损失函数评估仅仅基于新闻本身做出预测的情况，主损失函数评估基于新闻和额外信息的预测的情况，即使来自主损失函数的梯度发生弥散，来自辅助损失函数的信息也能够训练 Embeddding 和 LSTM 层。在模型中早点使用主要的损失函数是对于深度网络的一个良好的正则方法。总而言之，该模型框图如下：

![2](/img/article/keras-tutorial-1/multi_output_model.png)

让我们用函数式模型来实现这个框图。

主要的输入接收新闻本身，即一个整数的序列（每个整数编码了一个词）。这些整数位于 1 到 10，000 之间（即我们的字典有 10，000 个词）。这个序列有 100 个单词。

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 标题输入: 接收一个100个整数的序列，每个整数处于1到10000之间.
# 注意我们可以通过"name"参数命名任何层.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# 这个embedding层会编码输入序列到一个512维的向量序列
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# 一个LSTM会转化向量序列到一个单独的保存整个序列信息的向量
lstm_out = LSTM(32)(x)
```

然后，我们插入一个额外的损失，使得即使在主损失很高的情况下，LSTM 和 Embedding 层也可以平滑的训练。

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

再然后，我们将 LSTM 与额外的输入数据串联起来组成输入，送入模型中：

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 我们堆叠一个深度的全连接网络
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最后我们加上一个主要的logistic回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```

最后，我们定义整个 2 输入，2 输出的模型：

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

模型定义完毕，下一步编译模型。我们给额外的损失赋 0.2 的权重。我们可以通过关键字参数 `loss_weights` 或 `loss` 来为不同的输出设置不同的损失函数或权值。这两个参数均可为 Python 的列表或字典。这里我们给 `loss` 传递单个损失函数，这个损失函数会被应用于所有输出上。

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```

编译完成后，我们通过传递训练数据和目标值训练该模型：

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```

因为我们输入和输出是被命名过的（在定义时传递了“name”参数），我们也可以用下面的方式编译和训练模型：

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

### 共享层

另一个使用函数式模型的场合是使用共享层的时候。

考虑微博数据，我们希望建立模型来判别两条微博是否是来自同一个用户，这个需求同样可以用来判断一个用户的两条微博的相似性。

一种实现方式是，我们建立一个模型，它分别将两条微博的数据映射到两个特征向量上，然后将特征向量串联并加一个 logistic 回归层，输出它们来自同一个用户的概率。这种模型的训练数据是一对对的微博。

因为这个问题是对称的，所以处理第一条微博的模型当然也能重用于处理第二条微博。所以这里我们使用一个共享的 LSTM 层来进行映射。

首先，我们将微博的数据转为 (140，256) 的矩阵，即每条微博有 140 个字符，每个单词的特征由一个 256 维的词向量表示，向量的每个元素为 1 表示某个字符出现，为 0 表示不出现，这是一个 one-hot 编码。

之所以是 (140，256) 是因为一条微博最多有 140 个字符，而扩展的 ASCII 码表编码了常见的 256 个字符。

> 注：原文中此处为 Tweet，所以对外国人而言这是合理的。如果考虑中文字符，那一个单词的词向量就不止 256 了。

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```

若要对不同的输入共享同一层，就初始化该层一次，然后多次调用它：

```python
# 该层会输入一个矩阵然后返回一个大小为64的向量
shared_lstm = LSTM(64)

# 当我们重用相同层的实例很多次，并且层的权值也是重用的
# （这实际上完全是相同的层）
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 我们可以连接这两个向量:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 然后在最上面添加一个logistic回归层
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 我们定义了一个可训练的模型，将tweet的输入连接起来，输出预测
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

### 层“节点”的概念

无论何时，当你在某个输入上调用层时，你就创建了一个新的张量（即该层的输出），同时你也在为这个层增加一个“计算节点”。这个节点将输入张量映射为输出张量。当你多次调用该层时，这个层就有了多个节点，其下标分别为 0，1，2...

在上一版本的 Keras 中，你可以通过 `layer.get_output()` 方法来获得层的输出张量，或者通过 `layer.output_shape` 获得其输出张量的 shape。这个版本的 Keras 你仍然可以这么做（除了 `layer.get_output()` 被 `output` 替换）。但如果一个层与多个输入相连，会出现什么情况呢？

如果层只与一个输入相连，那没有任何困惑的地方。`.output` 将会返回该层唯一的输出：

```python
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

但当层与多个输入相连时，会出现问题：

```python
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```

上面这段代码会报错：

```shell
>> AssertionError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

通过下面这种调用方式即可解决：

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

对于 `input_shape` 和 `output_shape` 也是一样，如果一个层只有一个节点，或所有的节点都有相同的输入或输出 shape，那么 `input_shape` 和 `output_shape` 都是没有歧义的，并也只返回一个值。但是，例如你把一个相同的 `Conv2D` 应用于一个大小为 (3,32,32) 的数据，然后又将其应用于一个 (3,64,64) 的数据，那么此时该层就具有了多个输入和输出的 shape，你就需要显式的指定节点的下标，来表明你想取的是哪个了。

```python
a = Input(shape=(3, 32, 32))
b = Input(shape=(3, 64, 64))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)
assert conv.input_shape == (None, 3, 32, 32)

conved_b = conv(b)
assert conv.get_input_shape_at(0) == (None, 3, 32, 32)
assert conv.get_input_shape_at(1) == (None, 3, 64, 64)s
```

## Functional 模型示例

### inception模型

inception的详细结构参见Google的这篇论文：[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(3, 256, 256))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### 卷积层的残差连接

残差网络（Residual Network）的详细信息请参考这篇文章：[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

```python
from keras.layers import Conv2D, Input

# 输入张量是一个3通道的256x256图像
x = Input(shape=(3, 256, 256))
# 3x3卷积，3输出通道 (与输入通道一样)
y = Conv2D(3, (3, 3), padding='same')(x)
# 返回 x + y.
z = keras.layers.add([x, y])
```

### 共享视觉模型

该模型在两个输入上重用了图像处理的模型，用来判别两个 MNIST 数字是否是相同的数字

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# 首先定义视觉模型
digit_input = Input(shape=(1, 27, 27))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# 然后定义tell-digits-apart模型
digit_a = Input(shape=(1, 27, 27))
digit_b = Input(shape=(1, 27, 27))

# 共享视觉模型
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### 视觉问答模型

在针对一幅图片使用自然语言进行提问时，该模型能够提供关于该图片的一个单词的答案。

这个模型将自然语言的问题和图片分别映射为特征向量，将二者合并后训练一个 logistic 回归层，从一系列可能的回答中挑选一个。

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# 首先我们使用Sequential模型定义一个视觉模型
# 该模型会将一个图像编码成一个向量
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(3, 224, 224)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# 现在我们获得视觉模型的输出张量
image_input = Input(shape=(3, 224, 224))
encoded_image = vision_model(image_input)

# 接下来，我们定义一个语言模型把问题编码成一个向量
# 每个问题最多包含100个词，并且每个词用从1到9999的整数表示
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# 我们连接问题向量和图像向量
merged = keras.layers.concatenate([encoded_question, encoded_image])

# 然后我们在1000个可能的回答词语上训练一个logistic回归
output = Dense(1000, activation='softmax')(merged)

# 这是我们最终的模型:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 下一阶段在真实的数据上训练这个模型
```

### 视频问答模型

在做完图片问答模型后，我们可以快速将其转为视频问答的模型。在适当的训练下，你可以为模型提供一个短视频（如 100 帧）然后向模型提问一个关于该视频的问题，如“what sport is the boy playing？”->“football”

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 3, 224, 224))
# 通过之前已经训练好的 vision_model (权重重用)为视频编码
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 输出是一个向量的序列
encoded_video = LSTM(256)(encoded_frame_sequence)  # 输出是一个向量

# 这是一个模型层面的对问题的编码器表示，使用与之前相同的权值
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 使用编码器对问题进行编码
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 这是我们的视频问题回答模型
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```

## 常见问题

### 保存 Keras 模型

一般我们不推荐使用 pickle 或 cPickle 来保存 Keras 模型。Keras 自己就提供了 `model.save(filepath)` 函数将模型和权重保存在一个 HDF5 文件中，该文件将包含：

- 模型的结构，以便重构该模型
- 模型的权重
- 训练配置（损失函数，优化器等）
- 优化器的状态，以便于从上次训练中断的地方开始

使用 `keras.models.load_model(filepath)` 来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译。

```python
from keras.models import load_model

model.save('my_model.h5')  # 创建一个HDF5文件'my_model.h5'
del model  # 删除已经存在的model

# 返回一个编译好的模型，与之前的完全相同
model = load_model('my_model.h5')
```

> 注意，在使用前需要确保你已安装了 HDF5 和其 Python 库 h5py

如果你只是希望保存模型的结构，而不包含其权重或配置信息，可以使用：

```python
# 保存为JSON
json_string = model.to_json()

# 保存为YAML
yaml_string = model.to_yaml()
```

这项操作将把模型序列化为 json 或 yaml 文件，这些文件对人而言也是友好的，如果需要的话你甚至可以手动打开这些文件并进行编辑。当然，你也可以从保存好的 json 文件或 yaml 文件中载入模型：

```python
# 通过JSON重建模型:
from keras.models import model_from_json
model = model_from_json(json_string)

# 通过YAML重建模型
model = model_from_yaml(yaml_string)
```

如果需要保存模型的权重，可通过下面的代码利用 HDF5 进行保存。

```python
model.save_weights('my_model_weights.h5')
```

如果你需要在代码中初始化一个完全相同的模型，请使用：

```python
model.load_weights('my_model_weights.h5')
```

如果你需要加载权重到不同的网络结构（有些层一样）中，例如 fine-tune 或 transfer-learning，你可以通过层名字来加载模型：

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

```python
"""
假如原模型为：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# 新模型
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
model.add(Dense(10, name="new_dense"))  # will not be loaded

# 从第一个模型加载权值; 只会影响第一层 dense_1.
model.load_weights(fname, by_name=True)
```

### 获取中间层的输出

一种简单的方法是创建一个新的 `Model`，使得它的输出是你想要的那个输出：

```python
from keras.models import Model

model = ...  # 创建原模型

layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

此外，我们也可以建立一个 Keras 的函数来达到这一目的：

```python
from keras import backend as K

# Sequential模型
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]
```

### 在每个epoch后记录训练/测试的loss和正确率

`model.fit`在运行结束后返回一个`History`对象，其中含有的`history`属性包含了训练过程中损失函数的值以及其他度量指标。

```python
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```

> 本文整理自[《Keras中文文档》](http://keras-cn.readthedocs.io/en/latest/)

