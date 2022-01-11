---
layout: article
title: "GlobalPointer：用统一的方式处理嵌套和非嵌套NER"
author: 苏剑林
tags:
    - NLP
mathjax: true
---

> 转载自[《GlobalPointer：用统一的方式处理嵌套和非嵌套NER》](https://kexue.fm/archives/8373)，作者：苏剑林。

本文将介绍一个称为 GlobalPointer 的设计，它利用全局归一化的思路来进行命名实体识别 (NER)，可以无差别地识别嵌套实体和非嵌套实体，在非嵌套 (Flat NER) 的情形下它能取得媲美 CRF 的效果，而在嵌套 (Nested NER) 情形它也有不错的效果。还有，在理论上，GlobalPointer 的设计思想就比 CRF 更合理；而在实践上，它训练的时候不需要像 CRF 那样递归计算分母，预测的时候也不需要动态规划，是完全并行的，理想情况下时间复杂度是 $\mathscr{O}(1)$！

简单来说，就是更漂亮、更快速、更强大！真有那么好的设计吗？不妨继续看看。

![global-pointer](/img/article/global-pointer/global_pointer.png)

<center>GlobalPointer 多头识别嵌套实体示意图</center>

## GlobalPointer

常规的 Pointer Network 的设计在做实体识别或者阅读理解时，一般是用两个模块分别识别实体的首和尾，这会带来训练和预测时的不一致。而 GlobalPointer 就是针对这个不一致而设计的，它将首尾视为一个整体去进行判别，所以它更有“全局观”（更 Global）。

> 常规的 Pointer Network 在预测的时候以实体为单位的，实体对了才算对（首尾都对了才算对），训练的时候首尾单独训练评估。简单来说，训练的时候首尾是孤立的，预测的时候首尾是有联系的，这就是不一致性。

### 基本思路

具体来说，假设要识别文本序列长度为 $n$，简单起见先假定只有一种实体要识别，并且假定每个待识别实体是该序列的一个连续片段，长度不限，并且可以相互嵌套（两个实体之间有交集），那么该序列有多少个“候选实体”呢？不难得出，答案是 $n(n+1)/2$ 个，即长度为 $n$ 的序列有 $n(n+1)/2$ 个不同的连续子序列，这些子序列包含了所有可能的实体，而我们要做的就是从这 $n(n+1)/2$ 个“候选实体”里边挑出真正的实体，其实就是一个“$n(n+1)/2$ 选 $k$”的多标签分类问题。如果有 $m$ 种实体类型需要识别，那么就做成 $m$ 个“$n(n+1)/2$ 选 $k$”的多标签分类问题。这就是 GlobalPointer 的基本思想，以实体为基本单位进行判别，如本文开头的图片所示。

可能有读者会问：这种设计的复杂度明明就是 $\mathscr{O}(n^2)$ 呀，不会特别慢吗？如果现在还是 RNN/CNN 的时代，那么它可能就显得很慢了，但如今是 Transformer 遍布 NLP 的时代，Transformer 的每一层都是 $\mathscr{O}(n^2)$ 的复杂度，多 GlobalPointer 一层不多，少 GlobalPointer 一层也不少，关键是 $\mathscr{O}(n^2)$ 的复杂度仅仅是空间复杂度，如果并行性能好的话，时间复杂度甚至可以降到 $\mathscr{O}(1)$，所以不会有明显感知。

### 数学形式

设长度为 $n$ 的输入 $t$ 经过编码后得到向量序列 $[\boldsymbol{h}\_1,\boldsymbol{h}\_2,\cdots,\boldsymbol{h}\_n]$，通过变换 $\boldsymbol{q}\_{i,\alpha}=\boldsymbol{W}\_{q,\alpha}\boldsymbol{h}\_i+\boldsymbol{b}\_{q,\alpha}$ 和 $\boldsymbol{k}\_{i,\alpha}=\boldsymbol{W}\_{k,\alpha}\boldsymbol{h}\_i+\boldsymbol{b}\_{k,\alpha}$ 我们可以得到序列向量序列 $[\boldsymbol{q}\_{1,\alpha},\boldsymbol{q}\_{2,\alpha},\cdots,\boldsymbol{q}\_{n,\alpha}]$ 和 $[\boldsymbol{k}\_{1,\alpha},\boldsymbol{k}\_{2,\alpha},\cdots,\boldsymbol{k}\_{n,\alpha}]$，它们是识别第 $\alpha$ 种类型实体所用的向量序列。此时我们可以定义：

$$
s_{\alpha}(i,j) = \boldsymbol{q}_{i,\alpha}^{\top} \boldsymbol{k}_{j,\alpha} \tag{1}
$$

作为从 $i$ 到 $j$ 的连续片段是一个实体的打分。也就是说，用 $\boldsymbol{q}\_{i,\alpha}$ 与 $\boldsymbol{k}\_{j,\alpha}$ 的内积，作为片段 $t\_{[i:j]}$ 是类型为 $\alpha$ 的实体的打分 (logits)，这里的 $t\_{[i:j]}$ 指的是序列 $t$ 的第 $i$ 个到第 $j$ 个元素组成的连续子串。在这样的设计下，GlobalPointer 事实上就是 Multi-Head Attention 的一个简化版而已，有多少种实体就对应多少个 head，相比 Multi-Head Attention 去掉了 $\boldsymbol{V}$ 相关的运算。

### 相对位置

理论上来说，式 $(1)$ 这样的设计就足够了，但实际上训练语料比较有限的情况下，它的表现往往欠佳，因为它没有显式地包含相对位置信息。在后面的实验中我们将会看到，加不加相对位置信息，效果可以相差30个百分点以上！

比如，我们要识别出地名，输入是天气预报的内容“北京：21度；上海：22度；杭州：23度；广州：24度；...”，这时候要识别出来的实体有很多，如果没有相对位置信息输入的话，GlobalPointer 对实体的长度和跨度都不是特别敏感，因此很容易把任意两个实体的首尾组合都当成目标预测出来（即预测出“北京：21度；上海”这样的实体）。相反，有了相对位置信息之后，GlobalPointer 就会对实体的长度和跨度比较敏感，因此能更好地分辨出真正的实体出来。

用哪种相对位置编码呢？理论上来说，Transformer 里边所有的相对位置编码都可以考虑用（参考[《Transformer 位置编码》](/2021/05/30/transformer-position-encoding.html)），但真的要去落实就会发现一个问题，大多数相对位置编码都对相对位置进行了一个截断，虽然这个截断范围对我们要识别的实体来说基本都够用了，但未免有点不优雅，不截断又会面临可学参数太多的问题。想来想去，还是觉得笔者之前构思的旋转式位置编码 (RoPE) 比较适合。

RoPE 的其实就是一个变换矩阵 $\boldsymbol{\mathcal{R}}\_i$，满足关系 $\boldsymbol{\mathcal{R}}\_i^{\top}\boldsymbol{\mathcal{R}}\_j = \boldsymbol{\mathcal{R}}\_{j-i}$，这样一来我们分别应用到 $\boldsymbol{q},\boldsymbol{k}$ 中，就有

$$
s_{\alpha}(i,j) = (\boldsymbol{\mathcal{R}}_i\boldsymbol{q}_{i,\alpha})^{\top}(\boldsymbol{\mathcal{R}}_j\boldsymbol{k}_{j,\alpha}) = \boldsymbol{q}_{i,\alpha}^{\top} \boldsymbol{\mathcal{R}}_i^{\top}\boldsymbol{\mathcal{R}}_j\boldsymbol{k}_{j,\alpha} = \boldsymbol{q}_{i,\alpha}^{\top} \boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{k}_{j,\alpha}\tag{2}
$$

从而就显式地往打分 $s_{\alpha}(i,j)$ 注入了相对位置信息。

## 优化细节

在这部分内容中，我们会讨论关于 GlobalPointer 在训练过程中的一些细节问题，包括损失函数的选择以及评价指标的计算和优化等，从中我们可以看到，GlobalPointer 以实体为单位的设计有着诸多优雅和便利之处。

### 损失函数

到目前为止，我们已经设计好了打分 $s_{\alpha}(i,j)$，识别特定的类 $\alpha$ 的实体，则变成了共有 $n(n+1)/2$ 类的多标签分类问题。接下来的关键是损失函数的设计。最朴素的思路是变成 $n(n+1)/2$ 个二分类，然而实际使用时 $n$ 往往并不小，那么 $n(n+1)/2$ 更大，而每个句子的实体数不会很多（每一类的实体数目往往只是个位数），所以如果是 $n(n+1)/2$ 个二分类的话，会带来极其严重的类别不均衡问题。

这时候我们之前研究的[《将“softmax+交叉熵”推广到多标签分类问题》](https://kexue.fm/archives/7359)就可以派上用场了。简单来说，这是一个用于多标签分类的损失函数，它是单目标多分类交叉熵的推广，特别适合总类别数很大、目标类别数较小的多标签分类问题。其形式也不复杂，在 GlobalPointer 的场景，它为

$$
\log \left(1 + \sum\limits_{(i,j)\in P_{\alpha}} e^{-s_{\alpha}(i,j)}\right) + \log \left(1 + \sum\limits_{(i,j)\in Q_{\alpha}} e^{s_{\alpha}(i,j)}\right)\tag{3}
$$

其中 $P_{\alpha}$ 是该样本的所有类型为 $\alpha$ 的实体的首尾集合，$Q_{\alpha}$ 是该样本的所有非实体或者类型非 $\alpha$ 的实体的首尾集合，注意我们只需要考虑 $i\leq j$ 的组合，即

$$
\begin{aligned} 
\Omega=&\,\big\{(i,j)\,\big|\,1\leq i\leq j\leq n\big\}\\ 
P_{\alpha}=&\,\big\{(i,j)\,\big|\,t_{[i:j]}\text{是类型为}\alpha\text{的实体}\big\}\\ 
Q_{\alpha}=&\,\Omega - P_{\alpha} 
\end{aligned}\tag{4}
$$

而在解码阶段，所有满足 $s_{\alpha}(i,j) > 0$ 的片段 $t_{[i:j]}$ 都被视为类型为 $\alpha$ 的实体输出。可见，解码过程是及其简单的，并且在充分并行下解码效率就是 $\mathscr{O}(1)$！

### 评价指标

对于 NER 来说，常见的评价指标就是 F1，注意是实体级别的 F1，并非标注标签级别的 F1。在传统的 Pointer Network 或者 CRF 的设计下，我们并不容易在训练过程中直接计算实体级别的 F1，但是在 GlobalPointer 的设计下，不管是计算实体级别的 F1 还是 accuracy 都是很容易的，比如 F1 的计算如下：

```python
def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)
```

能有这么简单，主要就是因为 GlobalPointer 的“Global”，它的 `y_true` 和 `y_pred` 本身就已经是实体级别了，通过 `y_pred > 0` 我们就可以知道哪些实体被抽取出来的，然后做个匹配就可以算出各种（实体级别的）指标，达到了训练、评估、预测的一致性。

### 优化 F1 值

GlobalPointer 的“Global”还有一个好处，就是如果我们用它来做阅读理解的话，它可以直接优化阅读理解的 F1 指标！阅读理解的 F1 跟 NER 的 F1 有所不同，它是答案的一个模糊匹配程度，直接优化 F1 可能更有利于提高阅读理解的最终得分。将 GlobalPointer 用于阅读理解，相当于就只有一种实体类型的 NER，此时我们定义

$$
p(i,j) = \frac{e^{s(i,j)}}{\sum\limits_{i \leq j} e^{s(i,j)}}\tag{5}
$$

而有了 $p(i,j)$ 之后，用强化学习的思想（参考[《殊途同归的策略梯度与零阶优化》](https://kexue.fm/archives/7737)），优化 F1 就是以下述函数为损失：

$$
-\sum_{i\leq j} p(i,j) f_1(i,j) + \lambda \sum_{i\leq j}p(i,j)\log p(i,j)\tag{6}
$$

这里的 $f_1(i,j)$ 就是提前算好的片段 $t_{[i:j]}$ 与标准答案之间的 F1 相似度，$\lambda$ 是一个超参数。当然，算出所有的 $f_1(i,j)$ 成本可能会有点大，但它是一次性的，而且可以在计算时做些策略（比如首尾差别过大就直接置零），总的来说，可以控制在能接受的范围。如果为了提高阅读理解最终的 F1，这是一种比较直接的可以尝试的方案。（笔者在今年的百度 lic2021 阅读理解赛道上尝试过，确实能有一定的效果。）

## 实验结果

现在一切准备就绪，马上就能够开始实验了，实验代码整理如下：

> **开源地址：**[https://github.com/bojone/GlobalPointer](https://github.com/bojone/GlobalPointer)

目前 GlobalPointer 已经内置在 bert4keras>=0.10.6 中，bert4keras 的用户可以直接升级 bert4keras 使用。实验的三个任务均为中文 NER 任务，前两个为非嵌套 NER，第三个为嵌套 NER，它们的训练集文本长度统计信息为：

$$
\begin{array}{c|cc} 
\hline 
& \text{平均字数} & \text{字数标准差} \\ 
\hline 
\text{人民日报NER} & 46.93 & 30.08\\ 
\text{CLUENER} & 37.38 & 10.71\\ 
\text{CMeEE} & 54.15 & 80.27\\ 
\hline 
\end{array}
$$

### 人民日报

首先，我们验证一下在非嵌套场景 GlobalPointer 能否取代 CRF，语料是经典的人民日报语料，baseline 是 BERT+CRF 的组合，而对比的是 BERT+GlobalPointer 的组合，实验结果如下：

$$
\begin{array}{c} 
\text{人民日报NER实验结果} \\ 
{\begin{array}{c|cc|cc} 
\hline 
& \text{验证集F1} & \text{测试集F1} & \text{训练速度} & \text{预测速度}\\ 
\hline 
\text{CRF} & 96.39\% & 95.46\% & 1\text{x} & 1\text{x}\\ 
\text{GlobalPointer (w/o RoPE)} & 54.35\% & 62.59\% & 1.61\text{x} & 1.13\text{x} \\ 
\text{GlobalPointer (w/ RoPE)}& 96.25\% & 95.51\% & 1.56\text{x} & 1.11\text{x} \\ 
\hline 
\end{array}} 
\end{array}
$$

首先，表格中带来最大视觉冲击力的无疑是 GlobalPointer 有无 RoPE 的差距，达到了 30 个点以上！这说明了给 GlobalPointer 显式加入相对位置信息的重要性，后面的实验中我们将不再验证去掉 RoPE 的版本，默认都加上 RoPE。

从表格中还可以看出，在经典的非嵌套 NER 任务中，效果上 GlobalPointer 可以跟 CRF 相媲美，速度上 GlobalPointer 还更胜一筹，称得上是又快又好了。

### CLUENER

当然，可能因为人民日报这个经典任务的起点已经很高了，所以拉不开差距。为此，我们在测一下比较新的 [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020) 数据集，这个数据集也是非嵌套的，当前 SOTA 的 F1 是 81% 左右。BERT+CRF 与 BERT+GlobalPointer 的对比如下：

$$
\begin{array}{c} 
\text{CLUENER实验结果} \\ 
{\begin{array}{c|cc|cc} 
\hline 
& \text{验证集F1} & \text{测试集F1} & \text{训练速度} & \text{预测速度}\\ 
\hline 
\text{CRF} & 79.51\% & 78.70\% & 1\text{x} & 1\text{x}\\ 
\text{GlobalPointer}& 80.03\% & 79.44\% & 1.22\text{x} & 1\text{x} \\ 
\hline 
\end{array}} 
\end{array}
$$

这个实验结果说明了，当 NER 难度增加之后，哪怕只是非嵌套的场景，GlobalPointer 的效果能优于 CRF，这说明对于 NER 场景，GlobalPointer 其实比 CRF 更加好用。后面我们将对此做个简单的理论分析，进一步说明 GlobalPointer 相比 CRF 在理论上就更加合理。

至于速度方面，由于这个任务的文本长度普遍较短，因此 GlobalPointer 的速度增幅也没有那么明显。

### CMeEE

最后，我们来测一个嵌套的任务 (CMeEE)，它是去年 biendata 上的“[中文医学文本命名实体识别](https://www.biendata.xyz/competition/chip_2020_1/)”比赛，也是今年的“[中文医疗信息处理挑战榜 CBLUE](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414)”的任务1，简单来说就是医学方面的 NER，带有一定的嵌套实体。同样比较 CRF 和 GlobalPointer 的效果：

$$
\begin{array}{c} 
\text{CMeEE实验结果} \\ 
{\begin{array}{c|cc|cc} 
\hline 
& \text{验证集F1} & \text{测试集F1} & \text{训练速度} & \text{预测速度}\\ 
\hline 
\text{CRF} & 63.81\% & 64.39\% & 1\text{x} & 1\text{x}\\ 
\text{GlobalPointer}& 64.84\% & 65.98\% & 1.52\text{x} & 1.13\text{x} \\ 
\hline 
\end{array}} 
\end{array}
$$

可以看到效果上 GlobalPointer 明显地优于 CRF；速度方面，综合三个任务的结果，总的来说文本越长的任务， GlobalPointer 的训练加速就越明显，而预测速度通常也略有提升，但幅度没有训练阶段大。随后笔者以 RoBERTa large 为 encoder 继续捣鼓了一下，发现线上测试集就可以（不是太难地）达到 67% 以上，这说明 GlobalPointer 是一个“称职”的设计了。

当然，可能有读者会诟病：你拿非嵌套的 CRF 去做嵌套的 NER，这样跟 GlobalPointer 比肯定不公平呀。确实会有点，但是问题不大，一方面 CMeEE 目前的 F1 还比较低，嵌套的实体本来就不多，哪怕去掉嵌套部分当成非嵌套的来做，影响也不会太大；另一方面就是在嵌套 NER 方面，笔者还没发现比较简单明快的设计可以作为 baseline 跑跑的，所以就还是先跑个 CRF 看看了。欢迎读者报告其他设计的对比结果。

## 思考拓展

在本节中，我们将进一步对 CRF 和 GlobalPointer 做一个理论上的对比，并且介绍一些与 GlobalPointer 相关的工作，以方便读者更好地理解和定位 GlobalPointer。

### 相比 CRF

CRF (条件随机场，Conditional Random Field) 是序列标注的经典设计，由于大多数 NER 也能转化为序列标注问题，所以 CRF 也算是 NER 的经典方法，笔者也曾撰写过[《简明条件随机场CRF介绍（附带纯Keras实现）》](https://kexue.fm/archives/5542)和[《你的CRF层的学习率可能不够大》](https://kexue.fm/archives/7196)等文章来介绍 CRF。在之前的介绍中，我们介绍过，如果序列标注的标签数为 $k$，那么逐帧 softmax 和 CRF 的区别在于：

> 前者将序列标注看成是 $n$ 个 $k$ 分类问题，后者将序列标注看成是 $1$ 个 $k^n$ 分类问题。</center>

这句话事实上也说明了逐帧 softmax 和 CRF 用于 NER 时的理论上的缺点。怎么理解呢？逐帧 softmax 将序列标注看成是 $n$ 个 $k$ 分类问题，那是过于宽松了，因为某个位置上的标注标签预测对了，不代表实体就能正确抽取出来了，起码有一个片段的标签都对了才算对；相反，CRF 将序列标注看成是 $1$ 个 $k^n$ 分类问题，则又过于严格了，因为这意味着它要求所有实体都预测正确才算对，只对部分实体也不给分。虽然实际使用中我们用 CRF 也能出现部分正确的预测结果，但那只能说明模型本身的泛化能力好，CRF 本身的设计确实包含了“全对才给分”的意思。

> CRF 本质上跟单标签多分类相同，我们训练的时候（最大似然）要求的是“全对的标注序列得分越大越好”，但并没有提供“只对一部分的标注序列”的奖励信号。所以 CRF 有能力得到只对一部分的识别结果，纯粹是模型本身的泛化能力导致的，不是 CRF 本身导致的。

所以，逐帧 softmax 和 CRF 在理论上确实都存在不大合理的地方，而相比之下，GlobalPointer 则更加贴近使用和评测场景：它本身就是以实体为单位的，并且它设计为一个“多标签分类”问题，这样它的损失函数和评价指标都是实体颗粒度的，哪怕只对一部分也得到了合理的打分。因此，哪怕在非嵌套 NER 场景，GlobalPointer 能取得比 CRF 好也是“情理之中”的。

### 相关工作

如果读者比较关注实体识别、信息抽取的进展，那么应该可以发现，GlobalPointer 与前段时间的关系抽取新设计 [TPLinker](https://arxiv.org/abs/2010.13415) 很相似。但事实上，这种全局归一化的思想，还可以追溯到更远。

对于笔者来说，第一次了解到这种思想，是在百度 2017 年发表的一篇[《Globally Normalized Reader](https://arxiv.org/abs/1709.02828)》，里边提出了一种用于阅读理解的全局归一化设计（GNR），里边不单单将 (首, 尾) 视为一个整体了，而是 (句子, 首, 尾) 视为一个整体（它是按照先选句子，然后在句子中选首尾的流程，所以多了一个句子维度），这样一来组合数就非常多了，因此它还用了[《Sequence-to-Sequence Learning as Beam-Search Optimization》](https://arxiv.org/abs/1606.02960)里边的思路来降低计算量。

有了 GNR 作铺垫，其实 GlobalPointer 就很容易可以想到的，事实上早在前年笔者在做 LIC2019 的关系抽取赛道的时候，类似的想法就已经有了，但是当时还有几个问题没有解决。

第一，当时 Transformer 还没有流行起来，总觉得 $\mathscr{O}(n^2)$ 的复杂度很可怕；第二，当时[《将“softmax+交叉熵”推广到多标签分类问题》](https://kexue.fm/archives/7359)也还没想出来，所以多标签分类的不均衡问题没有很好的解决方案；第三，当时笔者对 NLP 各方面的理解也还浅，bert4keras 也没开发，一旦实验起来束手束脚的，出现问题也不知道往哪里调（比如开始没加上 RoPE，降低了 30 个点以上，如果是两年前，我肯定没啥调优方案了）。

所以，GlobalPointer 算是这两年来笔者经过各方面积累后的一个有点“巧合”但又有点“水到渠成”的工作。至于 TPLinker，它还真跟 GlobalPointer 起源没什么直接联系。当然，在形式上 GlobalPointer 确实跟 TPLinker 很相似，事实上 TPLinker 还可以追溯到更早的[《Joint entity recognition and relation extraction as a multi-head selection problem》](https://www.sciencedirect.com/science/article/abs/pii/S095741741830455X)，只不过这系列文章都主要是把这种 Global 的思想用于关系抽取了，没有专门针对 NER 优化。

### 加性乘性

在具体实现上，TPLinker 与 GlobalPointer 的一个主要区别是在 Multi-Head 上 TPLinker 用的是加性 Attention：

$$
s_{\alpha}(i,j) = \boldsymbol{W}_{o,\alpha}\tanh\left(\boldsymbol{W}_{h,\alpha}[\boldsymbol{h}_{i},\boldsymbol{h}_{j}]+\boldsymbol{b}_{h,\alpha}\right)+\boldsymbol{b}_{o,\alpha} 
\tag{7}
$$

目前尚不清楚该选择与式 $(1)$ 的效果差异有多大，但是相比式 $(1)$ 的乘性 Attention，虽然它们的理论复杂度相似，但实际实现上这种加性 Attention 的计算成本会大很多，尤其是空间成本（显存）会大很多～

所以笔者认为，就算加性效果确实比乘性好一些，也应该选择在乘性的基础上继续优化才行，因为加性的效率确实不行啊。此外，TPLinker 等文章也没有像本文一样报告过相对位置信息的重要性，难道在加性 Attention 中相对位置不那么重要了？这些暂时还不得而知。

## 本文小结

本文介绍了一种 NER 的新设计 GlobalPointer，它基于全局指针的思想，融合了笔者之前的一些研究结果，实现了用统一的方式处理嵌套和非嵌套 NER 的“理想设计”。实验结果显示，在非嵌套的情形下它能取得媲美 CRF 的效果，而在嵌套情形它也有不错的效果。

> 转载自[《GlobalPointer：用统一的方式处理嵌套和非嵌套NER》](https://kexue.fm/archives/8373)，作者：苏剑林。
