---
layout: article
title: "不可思议的 Word2Vec（下）：用 Tensorflow 和 Keras 来实现 Word2Vec"
author: 苏剑林
tags:
    - NLP
mathjax: true
sidebar:
  nav: incredible-word2vec
---

> 汇总整理自[《不可思议的Word2Vec》](http://spaces.ac.cn/archives/4299/)系列，作者苏剑林。部分内容有删改。

## Tensorflow 版的 Word2Vec

### 不同的地方

本文的主要模型还是 **CBOW** 或者 **Skip-Gram**，但在 loss 设计上有所不同。本文还是使用了完整的 softmax 结构，而不是 huffmax softmax 或者负采样方案，但是在训练 softmax 时，使用了基于随机负采样的交叉熵作为 loss。这种 loss 与已有的 nce_loss 和 sampled_softmax_loss 都不一样，这里姑且命名为 random softmax loss。

另外，在 softmax 结构中，一般是 $\text{softmax}(Wx+b)$ 这样的形式，考虑到 $W$ 矩阵的形状事实上跟词向量矩阵的形状是一样的，因此本文考虑了 softmax 层与词向量层共享权重的模型（这时候直接让 $b$ 为 0），这种模型等效于原有的 Word2Vec 的负采样方案，也类似于 glove 词向量的词共现矩阵分解，但由于使用了交叉熵损失，理论上收敛更快，而且训练结果依然具有 softmax 的预测概率意义（相比之下，已有的 Word2Vec 负样本模型训练完之后，最后模型的输出值是没有意义的，只有词向量是有意义的）。同时，由于共享了参数，因此词向量的更新更为充分，读者不妨多多测试这种方案。

所以，本文事实上也实现了 4 个模型组合：CBOW/Skip-Gram，是/否共享 softmax 层的参数，读者可以选择使用。

### loss 是怎么来的

前面已经说了，本文的主要目的之一就是测试新 loss 的有效性，下面简介一下这个 loss 的来源和形式。这要从 softmax 为什么训练难度大开始说起～

假设标签数（本文中也就是词表中词汇量）为 $n$，那么：

$$
\begin{aligned}(p_1,p_2,\dots,p_n) =& \text{softmax}(z_1,z_2,\dots,z_n)\\ 
=& \left(\frac{e^{z_1}}{Z}, \frac{e^{z_2}}{Z}, \dots, \frac{e^{z_n}}{Z}\right)\end{aligned}
$$

这里 $Z = e^{z_1} + e^{z_2} + \dots + e^{z_n}$。如果正确类别标签为 $t$，使用交叉熵为 loss，则：

$$
L=-\log \frac{e^{z_t}}{Z}
$$

梯度为：

$$
\nabla L=-\nabla z_t + \nabla (\log Z)=-\nabla z_t + \frac{\nabla Z}{Z}
$$

因为有 $Z$ 的存在，每次梯度下降时，都要计算完整的 $Z$ 来计算 $\nabla Z$，也就是每一个样本迭代一次的计算量就是 $\mathscr{O}(n)$ 了，对于 $n$ 比较大的情形，这是难以接受的，所以要寻求近似方案（huffman softmax 是其中的一种，但是它写起来会比较复杂，而且 huffman softmax 的结果通常略差于普通的 softmax，还有 huffman softmax 仅仅是训练快，但是如果预测时要找最大概率的标签，那么反而更加慢）。

让我们进一步计算 $\nabla L$：

$$
\begin{aligned}\nabla L=&-\nabla z_t + \frac{\sum_i e^{z_i}\nabla z_i}{Z}\\ 
=&-\nabla z_t + \frac{\sum_i e^{z_i}}{Z}\nabla z_i\\ 
=&-\nabla z_t + \sum_i p_i \nabla z_i\\ 
=&-\nabla z_t + \text{E}(\nabla z_i) 
\end{aligned}
$$

也就是说，最后的梯度由两项组成：一项是正确标签的梯度，一项是所有标签的梯度的均值，这两项反号，可以理解为这两项在“拉锯战”。计算量主要集中在第二项，因为要遍历所有才能算具体的均值。然而，均值本身就具有概率意义的，那么能不能直接就随机选取若干个来算这个梯度均值，而不用算全部梯度呢？如果可以的话，那每步更新的计算量就固定了，不会随着标签数的增大而快速增加。

但如果这样做的话，需要按概率来随机选取标签，这也并不容易写。然而，有个更巧妙的方法是不用我们直接算梯度，我们可以直接对 loss 动手脚。所以这就导致了本文的 loss：**对于每个“样本-标签”对，随机选取 nb_negative 个标签，然后与原来的标签组成 nb_negative+1 个标签，直接在这 nb_negative+1 个标签中算 softmax 和交叉熵。选取这样的 loss 之后再去算梯度，就会发现自然就是按概率来选取的梯度均值了**。

### 代码实现

自我感觉代码还是比较精炼的，单文件包含了所有内容。训练输出模仿了gensim中的Word2Vec。模型代码位于GitHub：
[https://github.com/bojone/tf_word2vec/blob/master/Word2Vec.py](https://github.com/bojone/tf_word2vec/blob/master/Word2Vec.py)

使用参考：

```python
from Word2Vec import *
import pymongo
db = pymongo.MongoClient().travel.articles
class texts:
    def __iter__(self):
        for t in db.find().limit(30000):
            yield t['words']

wv = Word2Vec(texts(), model='cbow', nb_negative=16, shared_softmax=True, epochs=2) #建立并训练模型
wv.save_model('myvec') #保存到当前目录下的myvec文件夹

#训练完成后可以这样调用
wv = Word2Vec() #建立空模型
wv.load_model('myvec') #从当前目录下的myvec文件夹加载模型
```

有几点需要说明的：

1. 训练的输入是分好词的句子，可以是列表，也可以是迭代器，注意，不能是生成器，这跟 gensim 版的 word2vec 的要求是一致的。因为生成器只能遍历一次，而训练 word2vec 需要多次遍历数据；
2. 模型不支持更新式训练，即训练完模型后，不能再用额外的文档更新原来的模型（不是不可以，是没必要，而且意义不大）；
3. 训练模型需要 tensorflow，推荐用 GPU 加速，训练完成后，重新加载模型并且使用模型，都不需要 tensorflow，只需要 numpy；
4. 对于迭代次数，一般迭代 1～2 次就够了，负样本个数 10～30 即可。其余参数如 batch_size，可以自己实验调整。

### 简单的对比实验

tensorflow 中，已有的两种近似训练 softmax 的 loss 是 nce_loss 和 sampled_softmax_loss，这里简单做一个比较。在一个旅游领域的语料中（两万多篇文章）训练同样的模型，并比较结果。模型 cbow，softmax 选择不共享词向量层，其余参数都采用相同的默认参数。

#### random_softmax_loss

耗时：8 分 19 秒（迭代次数 2 次，batch_size 为 8000）

相似度测试结果：

```python
>>> import pandas as pd
>>> pd.Series(wv.most_similar(u'水果'))
0 (食品, 0.767908)
1 (鱼干, 0.762363)
2 (椰子, 0.750326)
3 (饮料, 0.722811)
4 (食物, 0.719381)
5 (牛肉干, 0.715441)
6 (菠萝, 0.715354)
7 (火腿肠, 0.714509)
8 (菠萝蜜, 0.712546)
9 (葡萄干, 0.709274)
dtype: object

>>> pd.Series(wv.most_similar(u'自然'))
0 (人文, 0.645445)
1 (和谐, 0.634387)
2 (包容, 0.61829)
3 (大自然, 0.601749)
4 (自然环境, 0.588165)
5 (融, 0.579027)
6 (博大, 0.574943)
7 (诠释, 0.550352)
8 (野性, 0.548001)
9 (野趣, 0.545887)
dtype: object

>>> pd.Series(wv.most_similar(u'广州'))
0 (上海, 0.749281)
1 (武汉, 0.730211)
2 (深圳, 0.703333)
3 (长沙, 0.683243)
4 (福州, 0.68216)
5 (合肥, 0.673027)
6 (北京, 0.669859)
7 (重庆, 0.653501)
8 (海口, 0.647563)
9 (天津, 0.642161)
dtype: object

>>> pd.Series(wv.most_similar(u'风景'))
0 (景色, 0.825557)
1 (美景, 0.763399)
2 (景致, 0.734687)
3 (风光, 0.727672)
4 (景观, 0.57638)
5 (湖光山色, 0.573512)
6 (山景, 0.555502)
7 (美不胜收, 0.552739)
8 (明仕, 0.535922)
9 (沿途, 0.53485)
dtype: object

>>> pd.Series(wv.most_similar(u'酒楼'))
0 (酒家, 0.768179)
1 (排挡, 0.731749)
2 (火锅店, 0.729214)
3 (排档, 0.726048)
4 (餐馆, 0.722667)
5 (面馆, 0.715188)
6 (大排档, 0.709883)
7 (名店, 0.708996)
8 (松鹤楼, 0.705759)
9 (分店, 0.705749)
dtype: object

>>> pd.Series(wv.most_similar(u'酒店'))
0 (万豪, 0.722409)
1 (希尔顿, 0.713292)
2 (五星, 0.697638)
3 (五星级, 0.696659)
4 (凯莱, 0.694978)
5 (银泰, 0.693179)
6 (大酒店, 0.692239)
7 (宾馆, 0.67907)
8 (喜来登, 0.668638)
9 (假日, 0.662169)
```

#### nce_loss

耗时：4 分钟（迭代次数 2 次，batch_size 为 8000），然而相似度测试结果简直不堪入目，当然，考虑到用时变少了，为了公平，将迭代次数增加到 4 次，其余参数不变，重复跑一次。相似度测试结果依旧一塌糊涂，比如：

```python
>>> pd.Series(wv.most_similar(u'水果'))
0 (口, 0.940704)
1 (可, 0.940106)
2 (100, 0.939276)
3 (变, 0.938824)
4 (第二, 0.938155)
5 (：, 0.938088)
6 (见, 0.937939)
7 (不好, 0.937616)
8 (和, 0.937535)
9 (（, 0.937383)
dtype: object
```

有点怀疑是不是我使用姿势不对了～于是我再次调整，将 nb_negative 增加到 1000，然后迭代次数调回为 3，这样耗时为 9 分 17 秒，最后的 loss 比前面的要小一个数量级，比较相似度的结果有些靠谱了，但还是并非特别好，比如：

```python
>>> pd.Series(wv.most_similar(u'水果'))
0 (特产, 0.984775)
1 (海鲜, 0.981409)
2 (之类, 0.981158)
3 (食品, 0.980803)
4 (。, 0.980371)
5 (蔬菜, 0.979822)
6 (&, 0.979713)
7 (芒果, 0.979599)
8 (可, 0.979486)
9 (比如, 0.978958)
dtype: object

>>> pd.Series(wv.most_similar(u'自然'))
0 (与, 0.985322)
1 (地处, 0.984874)
2 (这些, 0.983769)
3 (夫人, 0.983499)
4 (里, 0.983473)
5 (的, 0.983456)
6 (将, 0.983432)
7 (故居, 0.983328)
8 (那些, 0.983089)
9 (这里, 0.983046)
dtype: object
```

#### sampled_softmax_loss

有了前面的经验，这次直接将 nb_negative 设为 1000，然后迭代次数为 3，这样耗时为 8 分 38 秒，相似度比较的结果是：

```python
>>> pd.Series(wv.most_similar(u'水果'))
0 (零食, 0.69762)
1 (食品, 0.651911)
2 (巧克力, 0.64101)
3 (葡萄, 0.636065)
4 (饼干, 0.62631)
5 (面包, 0.613488)
6 (哈密瓜, 0.604927)
7 (食物, 0.602576)
8 (干货, 0.601015)
9 (菠萝, 0.598993)
dtype: object

>>> pd.Series(wv.most_similar(u'自然'))
0 (人文, 0.577503)
1 (大自然, 0.537344)
2 (景观, 0.526281)
3 (田园, 0.526062)
4 (独特, 0.526009)
5 (和谐, 0.503326)
6 (旖旎, 0.498782)
7 (无限, 0.491521)
8 (秀美, 0.482407)
9 (一派, 0.479687)
dtype: object

>>> pd.Series(wv.most_similar(u'广州'))
0 (深圳, 0.771525)
1 (上海, 0.739744)
2 (东莞, 0.726057)
3 (沈阳, 0.687548)
4 (福州, 0.654641)
5 (北京, 0.650491)
6 (动车组, 0.644898)
7 (乘动车, 0.635638)
8 (海口, 0.631551)
9 (长春, 0.628518)
dtype: object

>>> pd.Series(wv.most_similar(u'风景'))
0 (景色, 0.8393)
1 (景致, 0.731151)
2 (风光, 0.730255)
3 (美景, 0.666185)
4 (雪景, 0.554452)
5 (景观, 0.530444)
6 (湖光山色, 0.529671)
7 (山景, 0.511195)
8 (路况, 0.490073)
9 (风景如画, 0.483742)
dtype: object
>>> pd.Series(wv.most_similar(u'酒楼'))
0 (酒家, 0.766124)
1 (菜馆, 0.687775)
2 (食府, 0.666957)
3 (饭店, 0.664034)
4 (川味, 0.659254)
5 (饭馆, 0.658057)
6 (排挡, 0.656883)
7 (粗茶淡饭, 0.650861)
8 (共和春, 0.650256)
9 (餐馆, 0.644265)
dtype: object

>>> pd.Series(wv.most_similar(u'酒店'))
0 (宾馆, 0.685888)
1 (大酒店, 0.678389)
2 (四星, 0.638032)
3 (五星, 0.633661)
4 (汉庭, 0.619405)
5 (如家, 0.614918)
6 (大堂, 0.612269)
7 (度假村, 0.610618)
8 (四星级, 0.609796)
9 (天域, 0.598987)
dtype: object
```

#### 总结

这个实验虽然不怎么严谨，但是应该可以说，在相同的训练时间下，从相似度任务来看，感觉上 random softmax 与 sampled softmax 效果相当，nce loss 的效果最差，进一步压缩迭代次数，调整参数也表明了类似结果，欢迎读者进一步测试。由于本文的 random softmax 对每一个样本都进行不同的采样，因此所需要采样的负样本数更少，并且采样更加充分。至于其他任务的比较，只能在以后的实践中进行了。

## Keras 版的 Word2Vec

### 代码

**Github：**[https://github.com/bojone/tf_word2vec/blob/master/word2vec_keras.py](https://github.com/bojone/tf_word2vec/blob/master/word2vec_keras.py)

```python
#! -*- coding:utf-8 -*-
#Keras版的Word2Vec，作者：苏剑林，http://kexue.fm
#Keras 2.0.6 ＋ Tensorflow 测试通过

import numpy as np
from keras.layers import Input,Embedding,Lambda
from keras.models import Model
import keras.backend as K

word_size = 128 #词向量维度
window = 5 #窗口大小
nb_negative = 16 #随机负采样的样本数
min_count = 10 #频数少于min_count的词将会被抛弃
nb_worker = 4 #读取数据的并发数
nb_epoch = 2 #迭代次数，由于使用了adam，迭代次数1～2次效果就相当不错
subsample_t = 1e-5 #词频大于subsample_t的词语，会被降采样，这是提高速度和词向量质量的有效方案
nb_sentence_per_batch = 20
#目前是以句子为单位作为batch，多少个句子作为一个batch（这样才容易估计训练过程中的steps参数，另外注意，样本数是正比于字数的。）

import pymongo
class Sentences: #语料生成器，必须这样写才是可重复使用的
    def __init__(self):
        self.db = pymongo.MongoClient().weixin.text_articles
    def __iter__(self):
        for t in self.db.find(no_cursor_timeout=True).limit(100000):
            yield t['words'] #返回分词后的结果

sentences = Sentences()
words = {} #词频表
nb_sentence = 0 #总句子数
total = 0. #总词频

for d in sentences:
    nb_sentence += 1
    for w in d:
        if w not in words:
            words[w] = 0
        words[w] += 1
        total += 1
    if nb_sentence % 10000 == 0:
        print u'已经找到%s篇文章'%nb_sentence

words = {i:j for i,j in words.items() if j >= min_count} #截断词频
id2word = {i+1:j for i,j in enumerate(words)} #id到词语的映射，0表示UNK
word2id = {j:i for i,j in id2word.items()} #词语到id的映射
nb_word = len(words)+1 #总词数（算上填充符号0）

subsamples = {i:j/total for i,j in words.items() if j/total > subsample_t}
subsamples = {i:subsample_t/j+(subsample_t/j)**0.5 for i,j in subsamples.items()} #这个降采样公式，是按照word2vec的源码来的
subsamples = {word2id[i]:j for i,j in subsamples.items() if j < 1.} #降采样表

def data_generator(): #训练数据生成器
    while True:
        x,y = [],[]
        _ = 0
        for d in sentences:
            d = [0]*window + [word2id[w] for w in d if w in word2id] + [0]*window
            r = np.random.random(len(d))
            for i in range(window, len(d)-window):
                if d[i] in subsamples and r[i] > subsamples[d[i]]: #满足降采样条件的直接跳过
                    continue
                x.append(d[i-window:i]+d[i+1:i+1+window])
                y.append([d[i]])
            _ += 1
            if _ == nb_sentence_per_batch:
                x,y = np.array(x),np.array(y)
                z = np.zeros((len(x), 1))
                yield [x,y],z
                x,y = [],[]
                _ = 0

#CBOW输入
input_words = Input(shape=(window*2,), dtype='int32')
input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs) #CBOW模型，直接将上下文词向量求和

#构造随机负样本，与目标组成抽样
target_word = Input(shape=(1,), dtype='int32')
negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'))(target_word)
samples = Lambda(lambda x: K.concatenate(x))([target_word,negatives]) #构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。

#只在抽样内做Dense和softmax
softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
softmax_biases = Embedding(nb_word, 1, name='b')(samples)
softmax = Lambda(lambda x: 
                    K.softmax((K.batch_dot(x[0], K.expand_dims(x[1],2))+x[2])[:,:,0])
                )([softmax_weights,input_vecs_sum,softmax_biases]) #用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能

#留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出

model = Model(inputs=[input_words,target_word], outputs=softmax)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(data_generator(), 
                    steps_per_epoch=nb_sentence/nb_sentence_per_batch, 
                    epochs=nb_epoch,
                    workers=nb_worker,
                    use_multiprocessing=True
                   )

model.save_weights('word2vec.model')

#通过词语相似度，检查我们的词向量是不是靠谱的
embeddings = model.get_weights()[0]
normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1,1))**0.5

def most_similar(w):
    v = normalized_embeddings[word2id[w]]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i],sims[i]) for i in sort[:10]]

import pandas as pd
pd.Series(most_similar(u'科学'))
```

### 要点

上面是 CBOW 模型的代码，如果需要 Skip-Gram，请自行修改，Keras 代码这么简单，改起来也容易。

纵观代码，就会发现搭建模型的部分还不到 10 行。事实上，CBOW 模型写起来是很简单的，唯一有难度的是为了提高效率而做的随机抽样版的 softmax（随机选若干个目标做 softmax，而不是完整 softmax）。在 Keras 中，实现的方式就是手写 Dense 层，而不是用自带的 Dense 层。具体步骤是：1、通过 random_uniform 生成随机整数，也就是负样本 id，然后跟目标输入拼在一起，构成一个抽样；2、通过 Embedding 层来存 softmax 的权重；3、把抽样中权重挑出来，组成一个小矩阵，然后用 K 后端做矩阵乘法，也就是实现抽样版本的 Dense 层了。反复看看代码就明白了。

最后，拼运行速度肯定拼不过 Gensim 版和原版的 Word2Vec 了，用 Keras 主要是灵活性强而已～这点大家需要留意哈。

> 汇总整理自[《不可思议的Word2Vec》](http://spaces.ac.cn/archives/4299/)系列，作者苏剑林。部分内容有删改。