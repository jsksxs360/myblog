---
layout: article
title: 从 GlobalPointer 到 GPLinker
author: 苏剑林
tags:
    - NLP
mathjax: true
---

NLP 知名博主[苏剑林](https://kexue.fm/)在 21 年首次提出了用于命名实体识别 (NER) 任务的 [GlobalPointer](https://kexue.fm/archives/8373) 模型，然后提出了改进版 [Efficient GlobalPointer](https://kexue.fm/archives/8877) 模型，能够用更少的参数量取得更好的结果，后来又基于 GlobalPointer 提出了用于实体关系联合抽取的 [GPLinker](https://kexue.fm/archives/8888) 模型，并且进一步将其拓展用于完成事件联合抽取。

与近年来越来越繁琐的信息抽取模型相比，这些模型不仅思路清晰，而且用简单的结构就实现了优异的性能。本文将对这些模型进行简单的梳理，帮助读者快速了解其核心思想。

## GlobalPointer

GlobalPointer 利用全局归一化的思路来进行命名实体识别 (NER)，因此不仅可以识别非嵌套实体 (Flat NER)，还能识别嵌套实体 (Nested NER)。与常规的 Pointer Network 使用两个模块分别识别实体的首和尾相比，GlobalPointer 将首尾视为一个整体去进行判别，因此更有“全局观”（更 Global）。

具体来说，假设文本长度为 $n$，只有一种实体要识别，并且实体可以相互嵌套（两个实体之间有交集），那么就会有 $\frac{n(n+1)}{2}$ 个“候选实体”（连续子序列），我们需要从这 $\frac{n(n+1)}{2}$ 个“候选实体”里边挑出真正的实体，这可以看成是一个“$\frac{n(n+1)}{2}$ 选 $k$”的多标签分类问题。如果扩展到识别 $m$ 种实体类型，就可以看作是 $m$ 个“$\frac{n(n+1)}{2}$ 选 $k$”的多标签分类问题，这就是 GlobalPointer 的基本思想，以实体为基本单位进行判别。下图展示了 GlobalPoniter 多头识别嵌套实体的示意图：

![global_pointer](/img/article/globalpointer-to-gplinker/global_pointer.png)

设长度为 $n$ 的输入 $t$ 经过编码后得到向量序列 $[\boldsymbol{h}\_1,\boldsymbol{h}\_2,\cdots,\boldsymbol{h}\_n]$，通过变换 $\boldsymbol{q}\_{i,\alpha}=\boldsymbol{W}\_{q,\alpha}\boldsymbol{h}\_i+\boldsymbol{b}\_{q,\alpha}$ 和 $\boldsymbol{k}\_{i,\alpha}=\boldsymbol{W}\_{k,\alpha}\boldsymbol{h}\_i+\boldsymbol{b}\_{k,\alpha}$ 可以得到向量序列 $[\boldsymbol{q}\_{1,\alpha},\boldsymbol{q}\_{2,\alpha},\cdots,\boldsymbol{q}\_{n,\alpha}]$ 和 $[\boldsymbol{k}\_{1,\alpha},\boldsymbol{k}\_{2,\alpha},\cdots,\boldsymbol{k}\_{n,\alpha}]$，它们是识别第 $\alpha$ 种类型实体所用的向量序列。此时我们可以定义：

$$
s_{\alpha}(i,j) = \boldsymbol{q}_{i,\alpha}^{\top}\boldsymbol{k}_{j,\alpha}\label{eq:s} \tag{1}
$$

作为从 $i$ 到 $j$ 的连续片段 $t_{[i:j]}$ 是一个类型为 $\alpha$ 的实体的打分。但是由于没有显式地包含相对位置信息，在训练语料比较有限的情况下往往表现欠佳，模型很容易把任意两个实体的首尾组合都当成目标预测出来。这里作者使用了自己提出的旋转式位置编码 ([RoPE](https://kexue.fm/archives/8265))，通过在 $\boldsymbol{q},\boldsymbol{k}$ 上应用一个满足关系 $\boldsymbol{\mathcal{R}}\_i^{\top}\boldsymbol{\mathcal{R}}\_j = \boldsymbol{\mathcal{R}}\_{j-i}$ 的变换矩阵，使得：

$$
s_{\alpha}(i,j) = (\boldsymbol{\mathcal{R}}_i\boldsymbol{q}_{i,\alpha})^{\top}(\boldsymbol{\mathcal{R}}_j\boldsymbol{k}_{j,\alpha}) = \boldsymbol{q}_{i,\alpha}^{\top} \boldsymbol{\mathcal{R}}_i^{\top}\boldsymbol{\mathcal{R}}_j\boldsymbol{k}_{j,\alpha} = \boldsymbol{q}_{i,\alpha}^{\top} \boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{k}_{j,\alpha} \tag{2}
$$

从而显式地往打分 $s_{\alpha}(i,j)$ 注入了相对位置信息。有了相对位置信息之后，GlobalPointer 就会对实体的长度和跨度比较敏感，因此能更好地分辨出真正的实体出来。

设计好打分 $s_{\alpha}(i,j)$ 之后，识别特定的类 $\alpha$ 的实体就变成了共有 $\frac{n(n+1)}{2}$ 类的多标签分类问题。损失函数最朴素的思路是变成 $\frac{n(n+1)}{2}$ 个二分类，然而实际使用时 $n$ 往往并不小，而每个句子的实体数不会很多（每一类的实体数目甚至只是个位数），所以会带来极其严重的类别不均衡问题。

这里作者基于自己研究的[将“softmax+交叉熵”推广到多标签分类问题](https://kexue.fm/archives/7359)，将单目标多分类交叉熵推广到多标签分类损失函数，适用于总类别数很大、目标类别数较小的多标签分类问题。对于 GlobalPointer，就为：

$$
\log \left(1 + \sum\limits_{(i,j)\in P_{\alpha}} e^{-s_{\alpha}(i,j)}\right) + \log \left(1 + \sum\limits_{(i,j)\in Q_{\alpha}} e^{s_{\alpha}(i,j)}\right)\tag{3}
$$

其中 $P_{\alpha}$ 是该样本的所有类型为 $\alpha$ 的实体的首尾集合，$Q_{\alpha}$ 是该样本的所有非实体或者类型非 $\alpha$ 的实体的首尾集合，即：

$$
\begin{aligned} 
\Omega=&\,\big\{(i,j)\,\big|\,1\leq i\leq j\leq n\big\}\\ 
P_{\alpha}=&\,\big\{(i,j)\,\big|\,t_{[i:j]}\text{ 是类型为 }\alpha\text{ 的实体}\big\}\\ 
Q_{\alpha}=&\,\Omega - P_{\alpha} 
\end{aligned} \tag{4}
$$

在解码阶段，所有满足 $s_{\alpha}(i,j) > 0$ 的片段 $t_{[i:j]}$ 都被视为类型为 $\alpha$ 的实体输出，在充分并行下解码效率就是 $\mathscr{O}(1)$！

作者在人民日报、[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)、[CMeEE](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414) 等多个数据集上对非嵌套/嵌套 NER 都进行了实验。结果显示，GlobalPointer 在非嵌套场景下能取得媲美 CRF 的效果，并且在速度上更胜一筹，在嵌套场景下也有不错的效果。

代码地址为：

- Keras 版：[https://github.com/bojone/GlobalPointer](https://github.com/bojone/GlobalPointer)
- Pytorch 版：[https://github.com/gaohongkui/GlobalPointer_pytorch](https://github.com/gaohongkui/GlobalPointer_pytorch)

### 思考拓展

如果序列标注的标签数为 $k$，那么逐帧 softmax 和 CRF 的区别在于：前者将序列标注看成是 $n$ 个 $k$ 分类问题，后者将序列标注看成是 $1$  个 $k^n$ 分类问题。但是，逐帧 softmax 将序列标注看成是 $n$ 个 $k$ 分类问题过于宽松了，因为某个位置上的标注标签预测对了，不代表就能正确抽取出实体，起码有一个片段的标签都对了才算对；相反，CRF 将序列标注看成是 $1$ 个 $k^n$ 分类问题又过于严格了，这意味着它要求所有实体都预测正确才算对，只对部分实体也不给分。

相比之下，GlobalPointer 则更加贴近使用和评测场景：它本身就是以实体为单位，并且设计为一个“多标签分类”问题，这样它的损失函数和评价指标都是实体颗粒度的，哪怕只对一部分也得到了合理的打分。因此，哪怕在非嵌套 NER 场景下，GlobalPointer 能取得比 CRF 更好的结果。

### Efficient GlobalPointer

针对 GlobalPointer 参数利用率不高的问题，作者又提出了 Efficient GlobalPointer 模型，并且通过实验证明参数量更少的 Efficient GlobalPointer 反而能取得更好的结果。

如式 $(1)$ 所示，对于每一种实体类型 $\alpha$，GlobalPointer 首先需要通过 $\boldsymbol{q}\_{i,\alpha}=\boldsymbol{W}\_{q,\alpha}\boldsymbol{h}\_i$ 和 $\boldsymbol{k}\_{i,\alpha}=\boldsymbol{W}\_{k,\alpha}\boldsymbol{h}\_i$ 对输入向量序列进行变换（这里暂时省略偏置项），因此有多少种实体，就有多少个 $\boldsymbol{W}\_{q,\alpha}$ 和 $\boldsymbol{W}\_{k,\alpha}$。设 $\boldsymbol{W}\_{q,\alpha},\boldsymbol{W}\_{k,\alpha}\in\mathbb{R}^{D\times d}$，那么每新增一种实体，就要新增 $2Dd$ 个参数；而用 CRF+BIO 标注的话，每新增一种实体，只需要增加 $2D$ 个参数（转移矩阵参数较少，忽略不计），所以 GlobalPointer 的参数量远远大于 CRF。

事实上，不同类型 $\alpha$ 的打分矩阵 $s_{\alpha}(i,j)$ 有很多相似之处，因为大多数 token-pair 都是“非实体”，这些非实体的正确打分都是负的，所以我们没有必要为每种实体类型都设计独立的 $s_{\alpha}(i,j)$，它们应当包含更多的共性。

怎么突出 $s_{\alpha}(i,j)$ 的共性呢？以 NER 为例，NER 实际上可以分解为“抽取”和“分类”两个步骤，即先抽取出实体片段，然后确定每个实体的类型。其中，“抽取”步骤相当于只有一种实体类型的 NER，用一个打分矩阵 $(\boldsymbol{W}\_q\boldsymbol{h}\_i)^{\top}(\boldsymbol{W}\_k\boldsymbol{h}\_j)$ 就可以完成，而“分类”步骤则可以用“特征拼接+Dense层”来完成，即 $\boldsymbol{w}\_{\alpha}^{\top}[\boldsymbol{h}_i;\boldsymbol{h}_j]$，因此新的打分函数为：

$$
s_{\alpha}(i,j) = (\boldsymbol{W}_q\boldsymbol{h}_i)^{\top}(\boldsymbol{W}_k\boldsymbol{h}_j) + \boldsymbol{w}_{\alpha}^{\top}[\boldsymbol{h}_i;\boldsymbol{h}_j]\label{eq:EGP-1}\tag{5}
$$

这样，“抽取”部分的参数是所有事件类型共享的，每新增一种实体类型，只需要新增对应的 $\boldsymbol{w}_{\alpha}\in\mathbb{R}^{2D}$，参数量也只是 $2D$。我们记 $\boldsymbol{q}_i=\boldsymbol{W}_q\boldsymbol{h}_i, \boldsymbol{k}_i=\boldsymbol{W}_k\boldsymbol{h}_i$，然后为了进一步地减少参数量，我们可以用 $[\boldsymbol{q}_i;\boldsymbol{k}_i]$ 来代替 $\boldsymbol{h}_i$，此时：

$$
s_{\alpha}(i,j) = \boldsymbol{q}_i^{\top}\boldsymbol{k}_j + \boldsymbol{w}_{\alpha}^{\top}[\boldsymbol{q}_i;\boldsymbol{k}_i;\boldsymbol{q}_j;\boldsymbol{k}_j]\label{eq:EGP} \tag{6}
$$

此时 $\boldsymbol{w}_{\alpha}\in\mathbb{R}^{4d}$，因此每新增一种实体类型所增加的参数量为 $4d$，由于通常 $d \ll D$，所以式 $(6)$ 的参数量更少。

作者在人民日报、CLUENER 和 CMeEE 数据集上的实验结果显示，除了在人民日报任务上有轻微下降外，其他两个任务都获得了一定提升，并且整体而言提升的幅度大于下降的幅度。作者初步推断：实体类别越多、任务越难时，Efficient GlobalPointer 越有效。

代码地址为：

- Keras 版：已经内置在 `bert4keras>=0.10.9` 中，只需要 

  `from bert4keras.layers import EfficientGlobalPointer as GlobalPointer`，

  就可以切换 Efficient GlobalPointer；

- Pytorch 版：[https://github.com/powerycy/Efficient-GlobalPointer](https://github.com/powerycy/Efficient-GlobalPointer)

## GPLinker

GPLinker 是作者在参考了 CasRel 之后的一些 SOTA 设计之后，提出的一个基于 GlobalPointer 的实体关系抽取模型。

### 关系抽取

关系抽取乍看之下是三元组 $(s,p,o)$（即 subject, predicate, object) 的抽取，但具体实现时实际上是“五元组” $(s_h,s_t,p,o_h,o_t)$ 的抽取，其中 $s_h,s_t$ 分别是 $s$ 的首、尾位置，而 $o_h,o_t$ 则分别是 $o$ 的首、尾位置。因此，从概率图角度可以构建模型：

1. 设计一个五元组的打分函数 $S(s_h,s_t,p,o_h,o_t)$；
2. 训练时让标注的五元组 $S(s_h,s_t,p,o_h,o_t) > 0$，其余五元组则 $S(s_h,s_t,p,o_h,o_t) < 0$；
3. 预测时枚举所有可能的五元组，输出 $S(s_h,s_t,p,o_h,o_t) > 0$ 的部分。

然而，直接枚举所有的五元组数目太多，假设句子长度为 $l$，$p$ 的总数为 $n$，即便加上 $s_h\leq s_t$ 和 $o_h\leq o_t$ 的约束，所有五元组的数目也是长度的四次方级别：

$$
n\times \frac{l(l+1)}{2}\times \frac{l(l+1)}{2}=\frac{1}{4}nl^2(l+1)^2 \tag{7}
$$

目前的算力一般最多能接受长度平方级别的计算量，所以我们每次顶多能识别“一对”首或尾。为此，作者基于对任务的理解采用以下分解：

$$
S(s_h,s_t,p,o_h,o_t) = S(s_h,s_t) + S(o_h,o_t) + S(s_h,o_h| p) + S(s_t, o_t| p)\label{eq:factor}\tag{8}
$$

其中，$S(s_h,s_t)$ 和 $S(o_h,o_t)$ 分别是 subject 和 object 的首尾打分，通过 $S(s_h,s_t) > 0$ 和 $S(o_h,o_t) > 0$ 来析出所有的 subject 和 object；后两项则是 predicate 的匹配，$S(s_h,o_h\|p)$ 代表以 subject 和 object 的首特征作为它们自身的表征来进行一次匹配，如果能确保 subject 内和 object 内是没有嵌套实体的，那么理论上 $S(s_h,o_h\|p) > 0$ 就足够析出所有的 predicate 了，但考虑到存在嵌套实体的可能，所以还要对实体的尾再进行一次匹配，即 $S(s_t, o_t\|p)$。

此时，训练和预测过程变为：

1. 训练时让标注的五元组 $S(s_h,s_t) > 0$、$S(o_h,o_t) > 0$、$S(s_h,o_h\| p) > 0$、$S(s_t, o_t\| p) > 0$，其余五元组则 $S(s_h,s_t) < 0$、$S(o_h,o_t) < 0$、$S(s_h,o_h\| p) < 0$、$S(s_t, o_t\| p) < 0$；
2. 预测时枚举所有可能的五元组，逐次输出 $S(s_h,s_t) > 0$、$S(o_h,o_t) > 0$、$S(s_h,o_h\| p) > 0$、$S(s_t, o_t\| p) > 0$ 的部分，然后取它们的交集作为最终的输出（即同时满足 4 个条件）。

在实现上，由于 $S(s_h,s_t)$ 和 $S(o_h,o_t)$ 用来识别 subject 和 object 对应的实体，相当于两种实体类型的 NER 任务，所以可以用一个 GlobalPointer 来完成；$S(s_h,o_h\| p)$ 用来识别 predicate 为 $p$ 的 $(s_h,o_h)$ 对，作者同样用 GlobalPointer 来完成，不过它不需要 NER 任务 $s_h \leq o_h$ 的约束，所以要去掉 GlobalPointer 默认的下三角 mask；$S(s_t, o_t\|p)$ 与 $S(s_h,o_h\| p)$ 同理，不再赘述。

> 虽然 GlobalPointer 提出时是用于识别嵌套/非嵌套实体，但是它是基于 token-pair 的识别来进行的，所以 GlobalPointer 实际上是一个通用的 token-pair 识别模型，而不是只能局限于 NER 任务。因此上述 $S(s_h,s_t)$、$S(o_h,o_t)$、$S(s_h,o_h\| p)$、$S(s_t, o_t\|p)$ 都可以用 GlobalPointer 来完成，只是需要根据任务需求决定要不要加下三角 mask。

损失函数则继续使用 GlobalPointer 默认使用的[多标签交叉熵](https://kexue.fm/archives/7359)，它的一般形式为：

$$
\log \left(1 + \sum\limits_{i\in \mathcal{P}} e^{-S_i}\right) + \log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right)\label{eq:loss-1}\tag{9}
$$

其中 $\mathcal{P},\mathcal{N}$ 分别是正、负类别的集合。但是由于使用“multi hot”向量来标记正、负类别，在 $S(s_h,o_h\| p)$ 和 $S(s_t, o_t\|p)$ 的场景下，我们各需要一个 $n\times l\times l$ 的矩阵来标记，再算上 batch_size，总维度就是 $2bnl^2$，这导致无论是创建还是传输成本都很大。所以作者实现一个“稀疏版”的多标签交叉熵，每次只需要传输正类所对应的的下标，这就大大减少了标签矩阵的尺寸。

“稀疏版”多标签交叉熵意味着要在只知道 $\mathcal{P}$ 和 $\mathcal{A}=\mathcal{P}\cup\mathcal{N}$ 的前提下去实现式 $(9)$，作者使用的实现方式是：

$$
\begin{aligned} 
&\,\log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right) = \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i} - \sum\limits_{i\in \mathcal{P}} e^{S_i}\right) \\ 
=&\, \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right) + \log \left(1 - \left(\sum\limits_{i\in \mathcal{P}} e^{S_i}\right)\Bigg/\left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right)\right) 
\end{aligned}\tag{10}
$$

令 $a = \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right),b=\log \left(\sum\limits_{i\in \mathcal{P}} e^{S_i}\right)$，就可以写为：

$$
\log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right) = a + \log\left(1 - e^{b - a}\right)\tag{11}
$$

这样就通过 $\mathcal{P}$ 和 $\mathcal{A}$ 算出了负类对应的损失，而正类部分的损失保持不变就好。由于正类个数不定，可以将类的下标从 1 开始，将 0 作为填充标签使得每个样本的标签矩阵大小一致，最后在 loss 的实现上对 0 类进行 mask 处理即可。相应的实现已经内置在 [bert4keras](https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272) 中。

作者在 LIC2019 上的实验结果显示，GPLinker 比 CasRel 更加有效，而且 Efficient GlobalPointer 能在更少参数的情况下媲美标准版 GlobalPointer 的效果。

代码地址为：

- Keras 版：[task_relation_extraction_gplinker.py](https://github.com/bojone/bert4keras/tree/master/examples/task_relation_extraction_gplinker.py)
- PyTorch 版：[https://github.com/xhw205/GPLinker_torch](https://github.com/xhw205/GPLinker_torch)

> GPLinker 虽然借鉴了 [TPLinker](https://arxiv.org/abs/2010.13415)，但是与 TPLinker 存在如下区别：
>
> 1. TPLinker 的 token-pair 分类特征是首尾特征后拼接做 Dense 变换得到的，其思想来源于 Additive Attention；GPLinker 则是用 GlobalPointer 实现，其思想来源于 Scaled Dot-Product Attention，拥有更少的显存占用和更快的计算速度。
> 2. GPLinker 分开识别 subject 和 object 的实体，而 TPLinker 将 subject 和 object 混合起来统一识别。
> 3. 在 $S(s_h,o_h\|p)$ 和 $S(s_t,o_t\|p)$，TPLinker 将其转化为了 $\frac{l(l+1)}{2}$ 个 3 分类问题，这会有明显的类别不平衡问题；而 GPLinker 使用作者提出的多标签交叉熵，不会存在不平衡问题。改进后的 [TPLinker-plus](https://github.com/131250208/TPlinker-joint-extraction/tree/master/tplinker_plus) 也用到了该多标签交叉熵。

### 事件抽取

后来，作者又在关系抽取模型 GPLinker 的基础上，结合完全子图搜索，设计了一个比较简单但相对完备的事件联合抽取模型。

一个标准的事件抽取样本如下：

<img src="/img/article/globalpointer-to-gplinker/event_extraction.png" width="800px" style="display: block; margin: auto;"/>

每个事件会有一个事件类型以及相应的触发词，并且配有不同角色的论元。其中，事件类型和论元角色是在约定的有限集合 (schema) 中选择，触发词和论元则是输入句子的片段。

传统的事件抽取模型一般分为“触发词检测”、“事件/触发词类型识别”、“事件论元检测”和“论元角色识别”四个子任务，要先检测触发词然后基于触发词做进一步的处理，所以如果训练集没有标注触发词就无法进行了。因此，作者把触发词也当作事件的一个论元角色，这样有无触发词就是增减一个论元而已，重点还是在于论元识别和事件划分。

- 对于论元识别，作者将 `(事件类型, 论元角色)` 组合成一个大类从而转化为 NER 问题，然后使用 GlobalPointer 来完成，这样还解决了实体嵌套的问题。

- 对于事件划分，最简单的做法就是直接把具有同一事件类型的论元聚合起来作为一个事件，但是同一个输入可能包含多个同一类型的事件，甚至同一个触发词都可能会触发多个事件，所以作者设计了一个额外的模块来做事件划分。

  一个事件论元之间的联系可以用无向图来描述，具体地，可以将每个论元看成是图上的一个节点，同一事件任意两个论元节点之间存在边，如果两个论元从未出现在同一事件中，那么节点之间则没有边。这样同一事件的任意两个节点都是相邻的，即为完全图 (Complete Graph) 或团 (Clique)，事件划分就转化为图上的完全子图搜索。如下图所示：

  <img src="/img/article/globalpointer-to-gplinker/complete_graph.png" width="450px" style="display: block; margin: auto;"/>

构建无向图可以沿用 TPLinker 的做法，如果两个论元实体有关系，那么它们的 `(首, 首)` 和 `(尾, 尾)` 都能匹配上，可以像前面关系抽取的 GPLinker 一样，用 GlobalPointer 来预测它们的匹配关系。这里由于只需要构建一个无向图，所以可以 mask 掉下三角部分，所有的边都只用上三角部分描述。

在构建好无向图之后，接下来就需要搜索出其中所有的完全子图了，比如上图中的 8 个节点就可以搜索出两个完全子图。作者为此提出了一种递归搜索算法：

1. 枚举图上的所有节点对，如果所有节点对都相邻，那么该图本身就是完全图，直接返回；如果有不相邻的节点对，那么执行步骤 2；
2. 对于每一对不相邻的节点，分别找出与之相邻的所有节点集（包含自身）构成子图，然后对每个子图集分别执行步骤 1。

> 以上图为例，可以找出 $B$、$E$ 是一对不相邻节点，那么我们可以分别找出其相邻集为 $\{A,B,C,D\}$ 和$\{D,E,F,G,H\}$，然后继续找 $\{A,B,C,D\}$ 和 $\{D,E,F,G,H\}$ 的不相邻节点对，发现找不到，所以 $\{A,B,C,D\}$ 和 $\{D,E,F,G,H\}$ 都是完全子图。
>
> 这个算法不依赖于不相邻节点对的顺序，因为对于“所有”不相邻节点对都要进行同样的操作，比如又找到$A$、$F$ 是一对不相邻节点，那我们同样要找其相邻集 $\{A,B,C,D\}$ 和 $\{D,E,F,G,H\}$ 并递归执行。因此在整个过程中可能会得到很多重复结果，最后再去重即可。

每次搜索的时候，只需要对同一事件类型的节点进行搜索，而多数情况下同一事件类型的论元只有个位数，所以上述算法看似复杂，实际运行速度还是很快的。

作者在 DuEE 和 DuEE-fin 上进行了实验，证明了模型的有效性。

代码地址为：

- Keras 版：[https://github.com/bojone/GPLinker](https://github.com/bojone/GPLinker)

## 参考

[[1]](https://kexue.fm/archives/8373) GlobalPointer：用统一的方式处理嵌套和非嵌套NER  
[[2]](https://kexue.fm/archives/8265) Transformer升级之路：2、博采众长的旋转式位置编码  
[[3]](https://kexue.fm/archives/7359) 将“softmax+交叉熵”推广到多标签分类问题  
[[4]](https://kexue.fm/archives/8877) Efficient GlobalPointer：少点参数，多点效果  
[[5]](https://kexue.fm/archives/8888) GPLinker：基于GlobalPointer的实体关系联合抽取  
[[6]](https://kexue.fm/archives/8926) GPLinker：基于GlobalPointer的事件联合抽取