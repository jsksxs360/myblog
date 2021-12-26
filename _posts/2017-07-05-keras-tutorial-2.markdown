---
layout: article
title: Keras 快速上手指南（中）：模型与网络层
tags:
    - Keras
mathjax: true
sidebar:
  nav: keras-note
---

## Keras 模型

Keras有两种类型的模型，**序贯模型(Sequential)**和**函数式模型(Model)**，函数式模型应用更为广泛，序贯模型是函数式模型的一种特殊情况。

两类模型有一些方法是相同的：

- `model.summary()`：打印出模型概况
- `model.get_config()`：返回包含模型配置信息的 Python 字典。模型也可以从它的 config 信息中重构回去

```python
config = model.get_config()

model = Model.from_config(config)
model = Sequential.from_config(config)
```

- `model.get_layer()`：依据层名或下标获得层对象
- `model.get_weights()`：返回模型权重张量的列表，类型为 numpy array
- `model.set_weights()`：从 numpy array 里将权重载入给模型，要求数组具有与 `model.get_weights()` 相同的形状。
- `model.save_weights(filepath)`：将模型权重保存到指定路径，文件类型是 HDF5（后缀是 .h5）
- `model.load_weights(filepath, by_name=False)`：从 HDF5 文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。如果想将权重载入不同的模型（有些层相同）中，则设置 `by_name=True`，只有名字匹配的层才会载入权重

## Sequential 模型接口

在[Sequntial 模型](/2017/07/04/article77/#4-sequntial-模型)中，我们已经介绍了模型的基本使用方法和实例，下面我们进一步对 Sequential 的 API 和参数做详细的介绍。

首先我们了解一下常用的 Sequential 属性：

- `model.layers` 是添加到模型上的层的 list

> 注：考虑到本文只是快速指南，因而省略了部分高级的属性和方法。

### add 添加层

向模型中添加一个层。

```python
add(self, layer)
```

- layer: Layer 对象

### pop 删除层

弹出模型最后的一层，无返回值

```python
pop(self)
```

### compile 编译模型 

```python
compile(self, optimizer, loss, metrics=None)
```

- optimizer：字符串（预定义优化器名）或优化器对象
- loss：字符串（预定义损失函数名）或目标函数
- metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是 `metrics=['accuracy']`

```python
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy'])
```

模型在使用前必须编译，否则在调用 `fit` 或 `evaluate` 时会抛出异常。

> 如果你只是载入模型并利用其 predict，可以不用进行 compile。在 Keras 中，compile 主要完成损失函数和优化器的一些配置，是为训练服务的。predict 会在内部进行符号函数的编译工作。

### fit 训练模型

将模型训练 `nb_epoch` 轮。

```python
fit(self, x, y, batch_size=32, epochs=10, verbose=1, validation_split=0.0, validation_data=None, shuffle=True)
```

- x：输入数据。如果模型只有一个输入，那么 x 的类型是 numpy array，如果模型有多个输入，那么 x 的类型应当为 list，list 的元素是对应于各个输入的 numpy array
- y：标签，numpy array
- batch_size：整数，指定进行梯度下降时每个 batch 包含的样本数。训练时一个 batch 的样本会被计算一次梯度下降，使目标函数优化一步。
- epochs：整数，训练的轮数，每个 epoch 会把训练集轮一遍。
- verbose：日志显示，0 为不在标准输出流输出日志信息，1 为输出进度条记录，2 为每个epoch输出一行记录
- validation_split：0~1 之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个 epoch 结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split 的划分在 shuffle 之前，因此如果你的数据本身是有序的，需要先手工打乱再指定 validation_split，否则可能会出现验证集样本不均匀。
- validation_data：形式为 (X，y) 的 tuple，是指定的验证集。此参数将覆盖 validation_spilt。
- shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。

`fit` 函数返回一个 `History` 的对象，其 `History.history` 属性记录了损失函数和其他指标的数值随 epoch 变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况。

### evaluate 测试模型

按 batch 计算在某些输入数据上模型的误差

```python
evaluate(self, x, y, batch_size=32, verbose=1)
```

- x：输入数据，与 `fit` 一样，是 numpy array 或 numpy array 的 list
- y：标签，numpy array
- batch_size：整数，指定进行测试时每个 batch 包含的样本数。
- verbose：日志显示，0 为不在标准输出流输出日志信息，1 为输出进度条记录

本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的 list（如果模型还有其他的评价指标）。`model.metrics_names` 将给出 list 中各个值的含义。

### predict 预测

按 batch 获得输入数据对应的预测结果，返回各个类别的可能性结果，是一个 n 维向量，n 等于类别的数量。

```python
predict(self, x, batch_size=32, verbose=0)
```

- x：输入数据，与 `fit` 一样，是 numpy array 或 numpy array 的 list
- batch_size：整数，每批次选取的数据集数量
- verbose：日志显示，0 为不在标准输出流输出日志信息，1 为输出进度条记录

### predict_classes 预测类别

按 batch 产生输入数据的类别预测结果，返回的是最可能的类别名称。

```
predict_classes(self, x, batch_size=32, verbose=1)
```

## 函数式模型接口

Keras 的函数式模型为 `Model`，即广义的拥有输入和输出的模型，我们使用 `Model` 来初始化一个函数式模型：

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

在这里，我们的模型以 `a` 为输入，以 `b` 为输出，同样我们可以构造拥有多输入和多输出的模型：

```python
model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])
```

在[Functional 模型](/2017/07/04/article77/#6-functional-模型)中，我们已经介绍了模型的基本使用方法和实例，下面我们进一步对函数式模型的 API 和参数做详细的介绍。

首先我们了解一下常用的 Modell 属性：

- `model.layers`：组成模型图的各个层
- `model.inputs`：模型的输入张量列表
- `model.outputs`：模型的输出张量列表

Model 模型的方法除了没有 add、pop 方法外与 Sequential 模型几乎是相同的，这里就不做赘述了。

## Keras 层

所有的 Keras 层对象都有如下方法：

- `layer.get_weights()`：返回层的权重（numpy array）
- `layer.set_weights(weights)`：从 numpy array 中将权重加载到该层中，要求 numpy array 的形状与 `layer.get_weights()` 的形状相同
- `layer.get_config()`：返回当前层配置信息的字典，层也可以借由配置信息重构:

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

或者：

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

如果层仅有一个计算节点（即该层不是共享层），则可以通过下列方法获得输入张量、输出张量、输入数据的形状和输出数据的形状：

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

如果该层有多个计算节点（参考[层计算节点和共享层](/2017/07/04/article77/#共享层)）。可以使用下面的方法：

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`

## 常用层

常用层对应于 core 模块，core 内部定义了一系列常用的网络层，包括全连接、激活层等。

### Dense 层

Dense 就是全连接层。

```python
keras.layers.core.Dense(units, activation=None)
```

- units：大于 0 的整数，代表该层的输出维度。
- activation：激活函数，为预定义的激活函数名。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：$a(x)=x$）

该层的输入为形如 (nb_samples, ..., input_shape[1]) 的 nD 张量，最常见的情况为 (nb_samples, input_dim) 的 2D 张量。输出为形如 (nb_samples, ..., units) 的 nD 张量，最常见的情况为 (nb_samples, output_dim) 的 2D 张量。

```python
# 作为sequential模型的第一层输入:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 模型输入shape为 (*, 16)，输出shape (*, 32)

# 第一层之后，就不需要指定输入大小了
model.add(Dense(32))
```

### Activation 层

激活层对一个层的输出施加激活函数。

```python
keras.layers.core.Activation(activation)
```

- activation：将要使用的激活函数，为预定义激活函数名或一个 Tensorflow/Theano 的函数。

该层输入 shape 任意，当使用激活层作为第一层时，要指定 `input_shape`。输出 shape 与输入 shape 相同。

### Dropout 层

Dropout 层将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，用于防止过拟合。

```python
keras.layers.core.Dropout(rate, seed=None)
```

- rate：0~1 的浮点数，控制需要断开的神经元的比例
- seed：整数，使用的随机数种子

### Flatten 层

Flatten 层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten 不影响 batch 的大小。

```python
keras.layers.core.Flatten()
```

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
# model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# model.output_shape == (None, 65536)
```

卷积操作将 (3,32,32) 格式的图像数据转换成 (64,32,32) 格式，Flatten 的结果则是 64 × 32 × 32 = 65536。

### Reshape 层

Reshape 层用来将输入 shape 转换为特定的 shape。

```python
keras.layers.core.Reshape(target_shape)
```

- target_shape：目标 shape，为整数的 tuple，不包含样本数目的维度（batch 大小）

该层的输入 shape 任意，但必须固定。当使用该层为模型首层时，需要指定 `input_shape` 参数。输出 shape 为 `(batch_size,)+target_shape`。

```python
# 作为Sequential模型的第一层
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# model.output_shape == (None, 3, 4)
# 注: `None` 是样本数目维

# 作为Sequential模型的中间层
model.add(Reshape((6, 2)))
# model.output_shape == (None, 6, 2)

# 也支持 `-1` 作为维度
model.add(Reshape((-1, 2, 2)))
# model.output_shape == (None, 3, 2, 2)
```

## 卷积层 

### Conv1D 层

一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数 `input_shape`。例如 `(10,128)` 代表一个长为 10 的序列，序列中每个信号为 128 向量。而 `(None, 128)` 代表变长的 128 维向量序列。

该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果 `use_bias=True`，则还会加上一个偏置项，若 `activation` 不为 None，则输出为经过激活函数的输出。

```python
keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None)
```

- filters：卷积核的数目（即输出的维度）
- kernel_size：整数或由单个整数构成的 list/tuple，卷积核的空域或时域窗长度
- strides：整数或由单个整数构成的 list/tuple，为卷积的步长。任何不为 1 的 strides 均与任何不为 1 的 dilation_rata 均不兼容
- padding：补 0 策略，为“valid”、“same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即 output[t] 不依赖于 input[t+1：]，当对不能违反时间顺序的时序信号建模时有用。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出 shape 与输入 shape 相同。
- activation：激活函数，为预定义的激活函数名。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：$a(x)=x$）
- dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rata均与任何不为1的strides均不兼容。

该层输入 shape 为形如 (samples，steps，input_dim) 的 3D 张量。输出 shape 为形如 (samples，new_steps，nb_filter) 的 3D 张量，因为有向量填充的原因，`steps` 的值会改变。

> 可以将 Convolution1D 看作 Convolution2D 的快捷版，对 (10，32) 的信号进行 1D 卷积相当于对其进行卷积核为 (filter_length, 32) 的 2D 卷积。

### Conv2D 层

二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供 `input_shape` 参数。例如 `input_shape = (128,128,3)` 代表 128×128 的彩色 RGB 图像（data_format='channels_last'）。

```python
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None)
```

- filters：卷积核的数目（即输出的维度）
- kernel_size：单个整数或由两个整数构成的 list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
- strides：单个整数或由两个整数构成的 list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为 1 的strides均与任何不为 1 的 dilation_rata 均不兼容
- padding：补 0 策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出 shape 与输入 shape 相同。
- activation：激活函数，为预定义的激活函数名。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：$a(x)=x$）
- dilation_rate：单个整数或由两个个整数构成的 list/tuple，指定 dilated convolution 中的膨胀比例。任何不为 1 的 dilation_rata 均与任何不为 1 的 strides 均不兼容。
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。以 128×128 的 RGB 图像为例，“channels_first”应将数据组织为 (3,128,128)，而“channels_last”应将数据组织为 (128,128,3)。该参数的默认值是 `~/.keras/keras.json` 中设置的值，若从未设置过，则为“channels_last”。

该层输入 shape 在“channels_first”模式下，输入形如 (samples,channels，rows，cols) 的 4D 张量，在“channels_last”模式下，输入形如 (samples，rows，cols，channels) 的 4D 张量。

> 注意：这里的输入 shape 指的是函数内部实现的输入 shape，而非函数接口应指定的 `input_shape`。

输出 shape 在“channels_first”模式下，为形如 (samples，nb_filter, new_rows, new_cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，new_rows, new_cols，nb_filter) 的 4D 张量。输出的行列数可能会因为填充方法而改变。

## 池化层

### MaxPooling1D 层

对时域 1D 信号进行最大值池化。

```python
keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

- pool_size：整数，池化窗口大小
- strides：整数或 None，下采样因子，例如设 2 将会使得输出 shape 为输入的一半，若为 None 则默认值为pool_size。
- padding：“valid”或者“same”

该层输入 shape 为形如 (samples，steps，features) 的 3D 张量，输出 shape 为形如 (samples，downsampled_steps，features) 的 3D 张量。

### MaxPooling2D 层

为空域信号施加最大值池化。

```python
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

- pool_size：整数或长为 2 的整数 tuple，代表在两个方向（竖直，水平）上的下采样因子，如取 (2，2) 将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
- strides：整数或长为 2 的整数 tuple，或者 None，步长值。
- border_mode：“valid”或者“same”
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。以 128×128 的 RGB 图像为例，“channels_first”应将数据组织为 (3,128,128)，而“channels_last”应将数据组织为 (128,128,3)。该参数的默认值是 `~/.keras/keras.json` 中设置的值，若从未设置过，则为“channels_last”。

该层输入 shape 在“channels_first”模式下，为形如 (samples，channels, rows，cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，rows, cols，channels) 的 4D 张量。输出 shape 在“channels_first”模式下，为形如 (samples，channels, pooled_rows, pooled_cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，pooled_rows, pooled_cols，channels) 的4D张量。

### AveragePooling1D 层

对时域 1D 信号进行平均值池化。

```python
keras.layers.pooling.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

- pool_size：整数，池化窗口大小
- strides：整数或 None，下采样因子，例如设 2 将会使得输出 shape 为输入的一半，若为 None 则默认值为 pool_size。
- padding：“valid”或者“same”

该层输入 shape 为形如 (samples，steps，features) 的 3D 张量，输出 shape 为形如 (samples，downsampled_steps，features) 的 3D 张量。

### AveragePooling2D 层

为空域信号施加平均值池化。

```python
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

- pool_size：整数或长为 2 的整数 tuple，代表在两个方向（竖直，水平）上的下采样因子，如取 (2，2) 将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
- strides：整数或长为 2 的整数tuple，或者 None，步长值。
- border_mode：“valid”或者“same”
- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。以 128×128 的 RGB 图像为例，“channels_first”应将数据组织为 (3,128,128)，而“channels_last”应将数据组织为 (128,128,3)。该参数的默认值是 `~/.keras/keras.json` 中设置的值，若从未设置过，则为“channels_last”。

该层输入 shape 在“channels_first”模式下，为形如 (samples，channels, rows，cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，rows, cols，channels) 的 4D 张量。输出 shape 在“channels_first”模式下，为形如 (samples，channels, pooled_rows, pooled_cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，pooled_rows, pooled_cols，channels) 的 4D 张量。

### GlobalMaxPooling1D 层

对于时间信号的全局最大池化。

```python
keras.layers.pooling.GlobalMaxPooling1D()
```

该层输入 shape 为形如 (samples，steps，features) 的 3D 张量，输出 shape 为形如 (samples, features) 的 2D 张量。

### GlobalAveragePooling1D 层

为时域信号施加全局平均值池化。

```python
keras.layers.pooling.GlobalAveragePooling1D()
```

该层输入 shape 为形如 (samples，steps，features) 的 3D 张量，输出 shape 为形如 (samples, features) 的 2D 张量。

### GlobalMaxPooling2D 层

为空域信号施加全局最大值池化。

```python
keras.layers.pooling.GlobalMaxPooling2D(data_format=None)
```

- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。以 128×128 的 RGB 图像为例，“channels_first”应将数据组织为 (3,128,128)，而“channels_last”应将数据组织为 (128,128,3)。该参数的默认值是 `~/.keras/keras.json` 中设置的值，若从未设置过，则为“channels_last”。

该层输入 shape 在“channels_first”模式下，为形如 (samples，channels, rows，cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，rows, cols，channels) 的 4D 张量。输出 shape 为形如 (nb_samples, channels) 的 2D 张量。

### GlobalAveragePooling2D 层

为空域信号施加全局平均值池化。

```python
keras.layers.pooling.GlobalAveragePooling2D(data_format=None)
```

- data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。以 128×128 的 RGB 图像为例，“channels_first”应将数据组织为 (3,128,128)，而“channels_last”应将数据组织为 (128,128,3)。该参数的默认值是 `~/.keras/keras.json` 中设置的值，若从未设置过，则为“channels_last”。

该层输入 shape 在“channels_first”模式下，为形如 (samples，channels, rows，cols) 的 4D 张量，在“channels_last”模式下，为形如 (samples，rows, cols，channels) 的 4D 张量。输出 shape 为形如 (nb_samples, channels) 的 2D 张量。

## 循环层

### Recurrent 层

这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类 LSTM、GRU 或 SimpleRNN，这些循环层都服从本层的性质，并接受本层指定的所有关键字参数。

```python
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False)
```

- return_sequences：布尔值，默认 `False`，控制返回类型。若为 `True` 则返回整个序列，否则仅返回输出序列的最后一个输出
- go_backwards：布尔值，默认为 `False`，若为 `True`，则逆向处理输入序列并返回逆序后的序列
- stateful：布尔值，默认为 `False`，若为 `True`，则一个 batch 中下标为 i 的样本的最终状态将会用作下一个 batch 同样下标的样本的初始状态
- input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定 input_shape)
- input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接 `Flatten` 层，然后又要连接 `Dense` 层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果循环层不是网络的第一层，你需要在网络的第一层中指定序列的长度（通过 `input_shap` 指 定）。

该层输入 shape 为形如 (samples，timesteps，input_dim) 的 3D 张量。输出 shape 如果 `return_sequences=True` 则返回形如 (samples，timesteps，output_dim) 的 3D 张量，否则返回形如 (samples，output_dim) 的 2D 张量。

```python
# 作为Sequential模型的第一层
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# model.output_shape == (None, 32)
# 注意: `None` 是batch维.

# 下面是完全相同的:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# 作为中间层, 不需要再去定义输入大小:
model.add(LSTM(16))

# 要堆叠循环层，必须设置输出到其他循环层的循环层 return_sequences=True
# 注意你只需在第一层上定义输入大小
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

循环层支持通过时间步变量对输入数据进行 Masking，如果想将输入数据的一部分屏蔽掉，请使用 Embedding 层并将参数 `mask_zero` 设为 True。

可以将 RNN 设置为“stateful”，意味着由每个 batch 计算出的状态都会被重用于初始化下一个 batch 的初始状态。状态 RNN 假设连续的两个 batch 之中，相同下标的元素有一一映射关系。要启用状态 RNN，请在实例化层对象时指定参数 `stateful=True`，并在 Sequential 模型使用固定大小的 batch：通过在模型的第一层传入 `batch_size=(...)` 和 `input_shape` 来实现。在函数式模型中，对所有的输入都要指定相同的 `batch_size`。

### SimpleRNN 层

全连接 RNN 网络，RNN 的输出会被回馈到输入。

```python
keras.layers.recurrent.SimpleRNN(units, activation='tanh', dropout=0.0, recurrent_dropout=0.0)
```

- units：输出维度
- activation：激活函数，为预定义的激活函数名
- dropout：0~1 之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1 之间的浮点数，控制循环状态的线性变换的神经元断开比例

### GRU 层

门限循环单元。

```python
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.0, recurrent_dropout=0.0)
```

- units：输出维度
- activation：激活函数，为预定义的激活函数名
- recurrent_activation: 为循环步施加的激活函数
- dropout：0~1 之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1 之间的浮点数，控制循环状态的线性变换的神经元断开比例

### LSTM 层

Keras 长短期记忆模型。

```python
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.0, recurrent_dropout=0.0)
```

- units：输出维度
- activation：激活函数，为预定义的激活函数名
- recurrent_activation: 为循环步施加的激活函数
- dropout：0~1 之间的浮点数，控制输入线性变换的神经元断开比例
- recurrent_dropout：0~1 之间的浮点数，控制循环状态的线性变换的神经元断开比例

## 嵌入层

### Embedding 层

嵌入层将正整数（下标）转换为具有固定大小的向量，如 [[4],[20]]->[[0.25,0.1],[0.6,-0.2]]。Embedding 层只能作为模型的第一层。

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, mask_zero=False, input_length=None)
```

- input_dim：大或等于 0 的整数，字典长度
- output_dim：大于 0 的整数，代表全连接嵌入的维度
- mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用循环层处理变长输入时有用。设置为 `True` 的话，模型中后续的层必须都支持 masking，否则会抛出异常。如果该值为 True，则下标 0 在字典中不可用，`input_dim` 应设置为 `|vocabulary| + 2`。
- input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接 `Flatten` 层，然后接 `Dense` 层，则必须指定该参数，否则 `Dense` 层的输出维度无法自动推断。

该层输入 shape 为形如 (samples，sequence_length) 的 2D 张量，输出 shape 为形如 (samples, sequence_length, output_dim) 的 3D 张量。

```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# 模型会接收一个(batch, input_length)的整数矩阵.
# 输入中的最大的整数 (词语索引) 应该不大于 999 (词表大小).
# model.output_shape == (None, 10, 64), None是batch维.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

## 融合层

Merge 层提供了一系列用于融合两个层或两个张量的层对象和方法。以大写首字母开头的是 Layer 类，以小写字母开头的是张量的函数。小写字母开头的张量函数在内部实际上是调用了大写字母开头的层。

### Add 层

该层接收一个列表的同 shape 张量，并返回它们的和，shape 不变。

```python
keras.layers.merge.Add()
```

该层的函数式包装为：`add(inputs)`，inputs 为长度至少为 2 的张量列表。

### Multiply 层

该层接收一个列表的同 shape 张量，并返回它们的逐元素积的张量，shape 不变。

```python
keras.layers.merge.Multiply()
```

该层的函数式包装为：`multiply(inputs)`，inputs 为长度至少为 2 的张量列表。

### Average 层

该层接收一个列表的同 shape 张量，并返回它们的逐元素均值，shape 不变。

```python
keras.layers.merge.Average()
```

该层的函数式包装为：`average(inputs)`，inputs 为长度至少为 2 的张量列表。

### Maximum 层

该层接收一个列表的同 shape 张量，并返回它们的逐元素最大值，shape 不变。

```python
keras.layers.merge.Maximum()
```

该层的函数式包装为：`maximum(inputs)`，inputs 为长度至少为 2 的张量列表。

### Concatenate 层

该层接收一个列表的同 shape 张量，并返回它们的按照给定轴相接构成的向量。

```python
keras.layers.merge.Concatenate(axis=-1)
```

- axis: 想接的轴

该层的函数式包装为：`concatenate(inputs, axis=-1))`，inputs 为长度至少为2的张量列，axis 为相接的轴。

### Dot 层

计算两个 tensor 中样本的张量乘积。例如，如果两个张量 `a` 和 `b` 的 shape 都为 (batch_size, n)，则输出为形如 (batch_size,1) 的张量，结果张量每个 batch 的数据都是 a[i,:] 和 b[i,:] 的矩阵（向量）点积。

```python
keras.layers.merge.Dot(axes, normalize=False)
```

- axes: 整数或整数的 tuple，执行乘法的轴。
- normalize: 布尔值，是否沿执行成绩的轴做 L2 规范化，如果设为 True，那么乘积的输出是两个样本的余弦相似性。

该层的函数式包装为：`dot(inputs, axes, normalize=False)`。inputs 为长度至少为2的张量列；axes 为整数或整数的 tuple，执行乘法的轴；normalize 为布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。

> 本文整理自[《Keras中文文档》](http://keras-cn.readthedocs.io/en/latest/)
