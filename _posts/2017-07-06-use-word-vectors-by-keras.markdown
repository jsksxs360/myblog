---
layout: article
title: "在 Keras 模型中使用预训练的词向量"
author: Francois Chollet
tags:
    - Keras
    - NLP
mathjax: false
---

> 转载自[《Using pre-trained word embeddings in a Keras model》](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)，作者：Francois Chollet，部分内容有修改。

通过本教程，你可以掌握技能：使用预先训练的词向量和卷积神经网络解决一个文本分类问题。本文代码已上传到 [Github](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py)。

## 什么是词向量?

“词向量”（词嵌入）是将一类将词的语义映射到向量空间中去的自然语言处理技术。即将一个词用特定的向量来表示，向量之间的距离（例如，任意两个向量之间的 L2 范式距离或更常用的余弦距离）一定程度上表征了的词之间的语义关系。由这些向量形成的几何空间被称为一个嵌入空间。

例如，“椰子”和“北极熊”是语义上完全不同的词，所以它们的词向量在一个合理的嵌入空间的距离将会非常遥远。但“厨房”和“晚餐”是相关的话，所以它们的词向量之间的距离会相对小。

理想的情况下，在一个良好的嵌入空间里，从“厨房”向量到“晚餐”向量的“路径”向量会精确地捕捉这两个概念之间的语义关系。在这种情况下，“路径”向量表示的是“发生的地点”，所以你会期望“厨房”向量 - “晚餐"向量（两个词向量的差异）捕捉到“发生的地点”这样的语义关系。基本上，我们应该有向量等式：晚餐 + 发生的地点 = 厨房（至少接近）。如果真的是这样的话，那么我们可以使用这样的关系向量来回答某些问题。例如，应用这种语义关系到一个新的向量，比如“工作”，我们应该得到一个有意义的等式，工作+ 发生的地点 = 办公室，来回答“工作发生在哪里？”。

词向量通过降维技术表征文本数据集中的词的共现信息。方法包括神经网络(“Word2vec”技术)，或矩阵分解。

## 文本使用的词向量和数据集

本文使用[GloVe词向量](http://nlp.stanford.edu/projects/glove/)。GloVe 是“Global Vectors for Word Representation”的缩写，一种基于共现矩阵分解的词向量。本文所使用的 GloVe 词向量是在 2014 年的英文维基百科上训练的，有 400k 个不同的词，每个词用 100 维向量表示。[点此下载](http://nlp.stanford.edu/data/glove.6B.zip) (词向量文件大小约为822M)。

本文使用的数据集是著名的“20 Newsgroup dataset”。该数据集共有 20 种新闻文本数据，我们将实现对该数据集的文本分类任务。数据集的说明和下载请参考[这里](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)。

不同类别的新闻包含大量不同的单词，在语义上存在极大的差别，一些新闻类别如下所示：

> comp.sys.ibm.pc.hardware
>
> comp.graphics
>
> comp.os.ms-windows.misc
>
> comp.sys.mac.hardware
>
> comp.windows.x
>
> rec.autos
>
> rec.motorcycles
>
> rec.sport.baseball
>
> rec.sport.hockey

## 实验方法

以下是我们如何解决分类问题的步骤：

- 将所有的新闻样本转化为词索引序列。所谓词索引就是为每一个词依次分配一个整数 ID。遍历所有的新闻文本，我们只保留最常见的 20,000 个词，而且每个新闻文本最多保留 1000 个词。
- 生成一个词向量矩阵。第 i 列表示词索引为 i 的词的词向量。
- 将词向量矩阵载入 Keras Embedding 层，设置该层的权重不可再训练（也就是说在之后的网络训练过程中，词向量不再改变）。
- Keras Embedding 层之后连接一个 1D 的卷积层，并用一个 softmax 全连接输出新闻类别。

### 数据预处理

我们首先遍历下语料文件下的所有文件夹，获得不同类别的新闻以及对应的类别标签，代码如下所示：

```python
texts = []  # 文本样例列表
labels_index = {}  # 字典，标签名称->数值id
labels = []  # 标签id列表
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath)
                texts.append(f.read())
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))
```

之后，我们可以新闻样本转化为神经网络训练所用的张量。所用到的 Keras 库是 `keras.preprocessing.text.Tokenizer` 和 `keras.preprocessing.sequence.pad_sequences`。代码如下所示：

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据集切分为训练集与验证集
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
```

### Embedding layer 设置

接下来，我们从 GloVe 文件中解析出每个词和它所对应的词向量，并用字典的方式存储：

```python
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
```

此时，我们可以根据得到的字典生成上文所定义的词向量矩阵：

```python
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # embedding index中没有的词为全零向量.
        embedding_matrix[i] = embedding_vector
```

现在我们将这个词向量矩阵加载到 Embedding 层中，注意，我们设置 trainable=False 使得这个编码层不可再训练。

```python
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
```

一个 Embedding 层的输入应该是一系列的整数序列，比如一个 2D 的输入，它的 shape 值为 (samples, indices)，也就是一个 samples 行，indeces 列的矩阵。每一次的 batch 训练的输入应该被 padded 成相同大小（尽管 Embedding 层有能力处理不定长序列，如果你不指定数列长度这一参数） dim)。所有的序列中的整数都将被对应的词向量矩阵中对应的列（也就是它的词向量）代替,比如序列 [1,2] 将被序列 [词向量[1],词向量[2]] 代替。这样，输入一个 2D 张量后，我们可以得到一个 3D 张量。

### 训练 1D 卷积

最后，我们可以使用一个小型的 1D 卷积解决这个新闻分类问题。

```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)
```

在两次迭代之后，这个模型最后可以达到 0.95 的分类准确率（4:1分割训练和测试集合）。你可以利用正则方法（例如 dropout）或在 Embedding 层上进行 fine-tuning 获得更高的准确率。

我们可以做一个对比实验，直接使用 Keras 自带的 Embedding 层训练词向量而不用 GloVe 向量。代码如下所示：

```python
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)
```

两次迭代之后，我们可以得到 0.9 的准确率。所以使用预训练的词向量作为特征是非常有效的。一般来说，在自然语言处理任务中，当样本数量非常少时，使用预训练的词向量是可行的（实际上，预训练的词向量引入了外部语义信息，往往对模型很有帮助）。

## 以下部分为译者添加

国内的 Rachel-Zhang 用 sklearn 对同样的数据集做过基于传统机器学习算法的实验，请点击[这里](http://blog.csdn.net/abcjennifer/article/details/23615947/)。 同时 Richard Socher 等在提出 GloVe 词向量的那篇论文中指出 GloVe 词向量比 word2vec 的性能更好[1]。之后的研究表示 word2vec 和 GloVe 其实各有千秋，例如 Schnabel 等提出了用于测评词向量的各项指标，测评显示 word2vec 在大部分测评指标优于 GloVe 和 C&W 词向量[2]。本文实现其实可以利用谷歌新闻的[word2vec词向量](http://pan.baidu.com/s/1kTCQqft)再做一组测评实验。

## 参考文献

[1] Pennington J, Socher R, Manning C D. Glove: Global Vectors for Word Representation[C]//EMNLP. 2014, 14: 1532-1543

[2] Schnabel T, Labutov I, Mimno D, et al. Evaluation methods for unsupervised word embeddings[C]//Proc. of EMNLP. 2015

> 转载自[《Using pre-trained word embeddings in a Keras model》](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)，作者：Francois Chollet，部分内容有修改。
