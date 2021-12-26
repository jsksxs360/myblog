---
layout: article
title: "将 PyTorch 版 bin 模型转换成 Tensorflow 版 ckpt"
author: 苏剑林
tags:
    - NLP
    - 机器学习
mathjax: false
---

最近由于工作上的需求，需要使用Tensorflow加载语言模型 [SpanBERT](https://arxiv.org/abs/1907.10529)（Facebook 发布的 BERT 模型的变体），但是作者只发布了 Pytorch 版的[预训练权重](https://github.com/facebookresearch/SpanBERT)，因此需要将其转换为 Tensorflow 可以加载的 checkpoint。

在 Pytorch 框架下，大多数开发者使用 Huggingface 发布的 Transformers 工具来加载语言模型，它同时支持加载 Pytorch 和 Tensorflow 版的模型。但是，目前基于 Tensorflow（或 Keras）的工具基本上都不支持加载 Pytorch 版的 bin 模型，转换代码在网上也很难找到，这带来了很多不便。

![bert](/img/article/pytorch-model-to-tensorflow-ckpt/bert.png)

通过搜索，目前能够找到的有以下几个转换代码片段可供参考：

- **[bin2ckpt](https://github.com/VXenomac/bin2ckpt/blob/master/bin2ckpt.py)**：用于转换 TinyBERT，但实测并不可用；
- **[convert_pytorch_checkpoint_to_tf](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_pytorch_checkpoint_to_tf2.py)**：Transformers 自带的转换脚本，但[官方文档](https://huggingface.co/transformers/)中并没有提及；
- **[pytorch_to_tf](https://github.com/mandarjoshi90/coref/blob/master/pytorch_to_tf.py)**：实体同指任务的一篇[论文](https://arxiv.org/abs/1908.09091)提供的转换脚本；
- [**PyTorch 版的 BERT 转换成 Tensorflow 版的 BERT**](https://zhuanlan.zhihu.com/p/349331135)：[VoidOc](https://www.zhihu.com/people/lidimeng) 编写的基于 Transformers 的转换脚本。

通过分析可以看到，将 PyTorch 版 bin 模型转换成 Tensorflow 版 ckpt 的过程并不复杂，可以分为以下几步：

1. 读取出模型中每一层网络结构的名称和参数；
2. 针对 PyTorch 和 Tensorflow 的模型格式差异对参数做一些调整；
3. 按照 Tensorflow 的格式保存模型。

## 读取和保存模型

PyTorch 和 Tensorflow 框架都提供了模型的读取和保存功能，因此读取和保存语言模型的过程非常简单。

读取模型直接使用 PyTorch 自带函数 `torch.load()` 或者 Transformers 提供的对应模型包的 `from_pretrained()` 函数就可以了；而保存模型则使用 Tensorflow 自带的模型保存器  `tf.train.Saver` 来完成。

以 BERT 模型为例，读取模型的过程就是：

```python
model = BertModel.from_pretrained(
    pretrained_model_name_or_path=pytorch_bin_path,
    state_dict=torch.load(os.path.join(pytorch_bin_path, pytorch_bin_model), map_location='cpu')
)
```

或者

```python
model = torch.load(os.path.join(pytorch_bin_path, pytorch_bin_model), map_location='cpu')
```

模型的保存过程则通过 Tensorflow 提供的保存器 `tf.train.Saver` 来完成：

```python
tf.reset_default_graph()
with tf.Session() as session:
    for var_name in state_dict:
        tf_name = to_tf_var_name(var_name) # 将层名称改为Tensorflow模型格式
        torch_tensor = state_dict[var_name].numpy()
        # 将参数矩阵改为Tensorflow模型格式
        if any([x in var_name for x in tensors_to_transpose]):
            torch_tensor = torch_tensor.T
        tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
        tf.keras.backend.set_value(tf_var, torch_tensor)
        tf_weight = session.run(tf_var)
        print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

    saver = tf.train.Saver(tf.trainable_variables())
    saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_").replace(".ckpt", "") + ".ckpt"))
```

整个过程就是先逐层读取模型的层名称和对应参数，然后将格式调整为 Tensorflow 模型的格式，再一次性写入到 checkpoint 文件中。

> **注意**：部分转换脚本忽略了 `reset_default_graph()` 这一操作，会导致生成的 meta 文件不仅保存网络结构，还会保存完整的网络参数，从而体积庞大。

## 调整模型格式

由于 PyTorch 和 Tensorflow 的模型格式定义有所差异，因此转换的关键就是对部分层的名称和参数矩阵进行调整。具体来说，首先需要构建名称映射字典，对部分层的名称进行调整：

```python
var_map = (
    ("layer.", "layer_"),
    ("word_embeddings.weight", "word_embeddings"),
    ("position_embeddings.weight", "position_embeddings"),
    ("token_type_embeddings.weight", "token_type_embeddings"),
    (".", "/"),
    ("LayerNorm/weight", "LayerNorm/gamma"),
    ("LayerNorm/bias", "LayerNorm/beta"),
    ("weight", "kernel"),
)

def to_tf_var_name(name: str):
    for patt, repl in iter(var_map):
        name = name.replace(patt, repl)
    return "bert/{}".format(name)
```

> **注意**：这里演示的是转换 BERT 模型，所以转换后的层名以 `bert/` 开头。如果转换的是其他模型，需要做相应的修改。

然后，由于 PyTorch 和 Tensorflow 模型中 `dense/kernel`、`attention/self/query`、`attention/self/key` 和 `attention/self/value` 层的参数矩阵互为转置，因此还需要对模型中的对应层的参数进行调整：

```python
tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

if any([x in var_name for x in tensors_to_transpose]):
    torch_tensor = torch_tensor.T
```

至此，转换过程就全部完成了。

## 完整的代码

综上所述，将 PyTorch 版 bin 模型转换成 Tensorflow 版 ckpt 的过程还是比较清晰的。本文对 [VoidOc](https://www.zhihu.com/people/lidimeng) 编写的脚本进行了进一步的简化，以转换 BERT 模型为例，完整的代码如下（[Github](https://github.com/jsksxs360/bin2ckpt)）：

```python
# coding=utf-8

"""
Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import torch
from transformers import BertModel
import os

def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported Huggingface models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return "bert/{}".format(name)

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_").replace(".ckpt", "") + ".ckpt"))

def convert(pytorch_bin_path: str, pytorch_bin_model: str, tf_ckpt_path: str, tf_ckpt_model: str):

    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=pytorch_bin_path,
        state_dict=torch.load(os.path.join(pytorch_bin_path, pytorch_bin_model), map_location='cpu')
    )

    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=tf_ckpt_path, model_name=tf_ckpt_model)

if __name__ == '__main__':
    bin_path = './pretrained_model/pytorch_model/'
    bin_model = 'pytorch_model.bin'
    ckpt_path = './pretrained_model/tensorflow_model/'
    ckpt_model = 'bert_model.ckpt'

    convert(bin_path, bin_model, ckpt_path, ckpt_model)
```

转换过程被包装为 `convert()` 函数，输入 PyTorch 版 bin 模型的路径和名称，以及 Tensorflow 版 ckpt 的保存路径和名称即可。

再次提醒一下，由于本文转换的 SpanBERT 只是 BERT 的一个变体，因此模型的层名称是与 BERT 模型完全一致的，如果需要转换其他模型，请自行修改 `to_tf_var_name()` 函数和 `tensors_to_transpose` 变量。

## 附

SpanBERT 目前已被实体同指在内的很多任务作为基准模型使用，因此 Facebook 没有提供 Tensorflow 版权重会给很多研究者和公司带来不便，尤其是目前很多商业项目是使用 Tensorflow 而不是 PyTorch 框架实现的。下面提供了转换好的 SpanBERT 模型权重给大家使用（包括 base 和 large 版）：

- SpanBERT (base & cased): 12-layer, 768-hidden, 12-heads , 110M parameters
- SpanBERT (large & cased): 24-layer, 1024-hidden, 16-heads, 340M parameters

下载地址：[百度盘](https://pan.baidu.com/s/1-VMYZ7KKxoCveokwIu_27g) (提取码: wtyr) \| [城通网盘](http://file.xiaosheng.run/d/4096332-43294170-42b59d) \| [GoogleDrive](https://drive.google.com/drive/folders/1W8MT99_SvECIaJ2rSthwCraSvM5XkGwH?usp=sharing)

