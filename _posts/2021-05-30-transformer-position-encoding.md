---
layout: article
title: "Transformer 位置编码"
author: 苏剑林
tags:
    - NLP
mathjax: true
---

> 转载自[《让研究人员绞尽脑汁的Transformer位置编码》](https://kexue.fm/archives/8130)和[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://kexue.fm/archives/8265)，作者：苏剑林，部分内容有修改。

不同于 RNN、CNN 等模型，对于 Transformer 模型来说，位置编码的加入是必不可少的，因为纯粹的 Attention 模块是无法捕捉输入顺序的，即无法区分不同位置的 Token。为此我们大体有两个选择：1、想办法将位置信息融入到输入中，这构成了**绝对位置编码**的一般做法；2、想办法微调一下 Attention 结构，使得它有能力分辨不同位置的 Token，这构成了**相对位置编码**的一般做法。

虽然说起来主要就是绝对位置编码和相对位置编码两大类，但每一类其实又能衍生出各种各样的变种，为此研究人员可算是煞费苦心、绞尽脑汁了，此外还有一些不按套路出牌的位置编码。本文就让我们来欣赏一下研究人员为了更好地表达位置信息所构建出来的“八仙过海，各显神通”般的编码方案。

## 1. 绝对位置编码

形式上来看，绝对位置编码是相对简单的一种方案，但即便如此，也不妨碍各路研究人员的奇思妙想，也有不少的变种。一般来说，绝对位置编码会加到输入中：在输入的第 $k$ 个向量 $\boldsymbol{x}_k$ 中加入位置向量 $\boldsymbol{p}_k$ 变为 $\boldsymbol{x}_k + \boldsymbol{p}_k$，其中 $\boldsymbol{p}_k$ 只依赖于位置编号 $k$。

### 训练式

很显然，绝对位置编码的一个最朴素方案是不特意去设计什么，而是直接**将位置编码当作可训练参数**，比如最大长度为512，编码维度为768，那么就初始化一个 $512\times 768$ 的矩阵作为位置向量，让它随着训练过程更新。现在的 BERT、GPT 等模型所用的就是这种位置编码，事实上它还可以追溯得更早，比如 2017 年 Facebook 的[《Convolutional Sequence to Sequence Learning》](https://arxiv.org/abs/1705.03122)就已经用到了它。

对于这种训练式的绝对位置编码，一般的认为它的缺点是没有外推性，即如果预训练最大长度为 512 的话，那么最多就只能处理长度为 512 的句子，再长就处理不了了。当然，也可以将超过 512 的位置向量随机初始化，然后继续微调。但[《层次分解位置编码，让BERT可以处理超长文本》](https://kexue.fm/archives/7947)表明，通过层次分解的方式，可以使得绝对位置编码能外推到足够长的范围，同时保持还不错的效果。因此，其实外推性也不是绝对位置编码的明显缺点。

### 三角式

三角函数式位置编码，一般也称为 **Sinusoidal 位置编码**，是 Google 的论文[《Attention is All You Need》](https://arxiv.org/abs/1706.03762)所提出来的一个显式解：

$$
\left\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\ 
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big) 
\end{aligned}\right.\tag{1}
$$

其中 $\boldsymbol{p}\_{k,2i},\boldsymbol{p}_{k,2i+1}$ 分别是位置 $k$ 的编码向量的第 $2i,2i+1$ 个分量，$d$ 是位置向量的维度。

很明显，三角函数式位置编码的特点是有显式的生成规律，因此可以期望于它有一定的外推性。另外一个使用它的理由是：由于 $\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$ 以及 $\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$，这表明位置 $\alpha+\beta$ 的向量可以表示成位置 $\alpha$ 和位置 $\beta$ 的向量组合，这提供了表达相对位置信息的可能性。但很奇怪的是，现在我们很少能看到直接使用这种形式的绝对位置编码的工作，原因不详。

### 递归式

原则上来说，RNN 模型不需要位置编码，它在结构上就自带了学习到位置信息的可能性（因为**递归就意味着我们可以训练一个“数数”模型**），因此，如果在输入后面先接一层 RNN，然后再接 Transformer，那么理论上就不需要加位置编码了。同理，我们也可以用 RNN 模型来学习一种绝对位置编码，比如从一个向量 $\boldsymbol{p}\_0$ 出发，通过递归格式 $\boldsymbol{p}_{k+1}=f(\boldsymbol{p}_k)$ 来得到各个位置的编码向量。

ICML 2020 的论文[《Learning to Encode Position for Transformer with Continuous Dynamical Model》](https://arxiv.org/abs/2003.09229)把这个思想推到了极致，它提出了用微分方程 (ODE) $d\boldsymbol{p}_t/dt=\boldsymbol{h}(\boldsymbol{p}_t,t)$ 的方式来建模位置编码，该方案称之为 FLOATER。显然，FLOATER 也属于递归模型，函数 $\boldsymbol{h}(\boldsymbol{p}_t,t)$ 可以通过神经网络来建模，因此这种微分方程也称为神经微分方程，关于它的工作最近也逐渐多了起来。

理论上来说，基于递归模型的位置编码也具有比较好的外推性，同时它也比三角函数式的位置编码有更好的灵活性（比如容易证明三角函数式的位置编码就是 FLOATER 的某个特解）。但是很明显，递归形式的位置编码牺牲了一定的并行性，可能会带速度瓶颈。

### 相乘式

刚才我们说到，输入 $\boldsymbol{x}_k$ 与绝对位置编码 $\boldsymbol{p}_k$ 的组合方式一般是 $\boldsymbol{x}_k + \boldsymbol{p}_k$，那有没有“不一般”的组合方式呢？比如 $\boldsymbol{x}_k \otimes \boldsymbol{p}_k$（逐位相乘）？我们平时在搭建模型的时候，对于融合两个向量有多种方式，相加、相乘甚至拼接都是可以考虑的，怎么大家在做绝对位置编码的时候，都默认只考虑相加了？

很抱歉，笔者也不知道答案。可能大家默认选择相加是因为向量的相加具有比较鲜明的几何意义，但是对于深度学习模型来说，这种几何意义其实没有什么实际的价值。最近笔者看到的一个实验显示，似乎将“加”换成“乘”，也就是 $\boldsymbol{x}_k \otimes \boldsymbol{p}_k$ 的方式，似乎比 $\boldsymbol{x}_k + \boldsymbol{p}_k$ 能取得更好的结果。具体效果笔者也没有完整对比过，只是提供这么一种可能性。关于实验来源，可以参考[《中文语言模型研究：(1) 乘性位置编码》](https://zhuanlan.zhihu.com/p/183234823)。

## 2. 相对位置编码

相对位置并没有完整建模每个输入的位置信息，而是在算 Attention 的时候考虑当前位置与被 Attention 的位置的相对距离，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。对于相对位置编码来说，它的灵活性更大，更加体现出了研究人员的“天马行空”。

### 经典式

相对位置编码起源于 Google 的论文[《Self-Attention with Relative Position Representations》](https://arxiv.org/abs/1803.02155)，华为开源的 NEZHA 模型也用到了这种位置编码，后面各种相对位置编码变体基本也是依葫芦画瓢的简单修改。

一般认为，相对位置编码是由绝对位置编码启发而来，考虑一般的带绝对位置编码的 Attention：

$$
\left\{\begin{aligned} 
\boldsymbol{q}_i =&\, (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q \\ 
\boldsymbol{k}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K \\ 
\boldsymbol{v}_j =&\, (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_V \\ 
a_{i,j} =&\, softmax\left(\boldsymbol{q}_i \boldsymbol{k}_j^{\top}\right)\\ 
\boldsymbol{o}_i =&\, \sum_j a_{i,j}\boldsymbol{v}_j 
\end{aligned}\right. \tag{2}
$$

其中 $softmax$ 对 $j$ 那一维归一化，这里的向量都是指行向量。我们初步展开 $\boldsymbol{q}_i \boldsymbol{k}_j^{\top}$：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \left(\boldsymbol{x}_i + \boldsymbol{p}_i\right)\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\left(\boldsymbol{x}_j + \boldsymbol{p}_j\right)^{\top} = \left(\boldsymbol{x}_i \boldsymbol{W}_Q + \boldsymbol{p}_i \boldsymbol{W}_Q\right)\left(\boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}\right) \tag{3}
$$

为了引入相对位置信息，Google 把第一项位置去掉，第二项 $\boldsymbol{p}\_j \boldsymbol{W}_K$ 改为二元位置向量 $\boldsymbol{R}\_{i,j}^{K}$，变成

$$
a_{i,j} = softmax\left(\boldsymbol{x}_i \boldsymbol{W}_Q\left(\boldsymbol{x}_j\boldsymbol{W}_K + \color{green}{\boldsymbol{R}_{i,j}^K}\right)^{\top}\right) \tag{4}
$$

以及 $\boldsymbol{o}\_i =\sum\limits\_j a\_{i,j}\boldsymbol{v}\_j = \sum\limits\_j a\_{i,j}(\boldsymbol{x}_j\boldsymbol{W}_V + \boldsymbol{p}_j\boldsymbol{W}_V)$ 中的 $\boldsymbol{p}\_j \boldsymbol{W}_V$ 换成 $\boldsymbol{R}\_{i,j}^{V}$：

$$
\boldsymbol{o}_i = \sum_j a_{i,j}\left(\boldsymbol{x}_j\boldsymbol{W}_V + \color{green}{\boldsymbol{R}_{i,j}^{V}}\right) \tag{5}
$$

所谓相对位置，是将本来依赖于二元坐标 $(i,j)$ 的向量 $\boldsymbol{R}\_{i,j}^{K},\boldsymbol{R}\_{i,j}^{V}$，改为只依赖于相对距离 $i−j$，并且通常来说会进行截断，以适应不同任意的距离

$$
\begin{aligned} 
\boldsymbol{R}_{i,j}^{K} = \boldsymbol{p}_K\left[\text{clip}(i-j, p_{\min}, p_{\max})\right]\\ 
\boldsymbol{R}_{i,j}^{V} = \boldsymbol{p}_V\left[\text{clip}(i-j, p_{\min}, p_{\max})\right] 
\end{aligned}\label{eq:rp-clip} \tag{6}
$$

这样一来，**只需要有限个位置编码，就可以表达出任意长度的相对位置（因为进行了截断）**，不管 $\boldsymbol{p}_K,\boldsymbol{p}_V$ 是选择可训练式的还是三角函数式的，都可以达到处理任意长度文本的需求。

### XLNET 式

XLNET 式位置编码其实源自 Transformer-XL 的论文[《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://arxiv.org/abs/1901.02860)，只不过因为使用了 Transformer-XL 架构的 [XLNET](https://arxiv.org/abs/1906.08237) 模型并在一定程度上超过了 BERT 后，Transformer-XL 才算广为人知，因此这种位置编码通常也被冠以 XLNET 之名。

XLNET 式位置编码源于对上述 $\boldsymbol{q}_i \boldsymbol{k}_j^{\top}$ 的完全展开：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}\label{eq:qk-exp} \tag{7}
$$

Transformer-XL 的做法很简单，直接将 $\boldsymbol{p}\_j$ 替换为相对位置向量 $\boldsymbol{R}\_{i-j}$，至于两个 $\boldsymbol{p}\_i$，则干脆替换为两个可训练的向量 $\boldsymbol{u},\boldsymbol{v}$：

$$
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i-j}^{\top}} +  \color{red}{\boldsymbol{u}}\boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \color{red}{\boldsymbol{v}} \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i-j}^{\top}} 
\tag{8}
$$

该编码方式中的 $\boldsymbol{R}\_{i-j}$ 没有像式 $(6)$ 那样进行截断，而是直接用了 Sinusoidal 式的生成方案。此外，$\boldsymbol{v}\_j$ 上的位置偏置就直接去掉了，即直接令 $\boldsymbol{o}\_i = \sum\limits_j a_{i,j}\boldsymbol{x}\_j\boldsymbol{W}\_V$。**似乎从这个工作开始，后面的相对位置编码都只加到 Attention 矩阵上去，而不加到 $\boldsymbol{v}_j$ 上去了。**

### T5 式

T5模型出自文章[《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://arxiv.org/abs/1910.10683)，里边用到了一种更简单的相对位置编码。思路依然源自展开式 $(7)$，如果非要分析每一项的含义，那么可以分别理解为“**输入-输入**”、“**输入-位置**”、“**位置-输入**”、“**位置-位置**”四项注意力的组合。如果我们认为输入信息与位置信息应该是独立（解耦）的，那么它们就不应该有过多的交互，所以“输入-位置”、“位置-输入”两项 Attention 可以删掉，而 $\boldsymbol{p}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{p}_j^{\top}$ 实际上只是一个只依赖于 $(i,j)$ 的标量，我们可以直接将它作为参数训练出来，即简化为

$$
\boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \color{green}{\boldsymbol{\beta}_{i,j}} \tag{9}
$$

说白了，它仅仅是在 Attention 矩阵的基础上**加一个可训练的偏置项**而已，而跟 XLNET 式一样，在 $\boldsymbol{v}_j$ 上的位置偏置则直接被去掉了。包含同样的思想的还有微软在ICLR 2021的论文[《Rethinking Positional Encoding in Language Pre-training》](https://arxiv.org/abs/2006.15595)中提出的 TUPE 位置编码。

比较“别致”的是，不同于常规位置编码对将 $\boldsymbol{\beta}_{i,j}$ 视为 $i−j$ 的函数并进行截断的做法，T5 对相对位置进行了一个“分桶”处理，即相对位置是 $i−j$ 的位置实际上对应的是 $f(i−j)$ 位置，映射关系如下：

$$
\begin{array}{c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c} 
\hline 
i - j & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15\\ 
\hline 
f(i-j) & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 8 & 8 & 8 & 9 & 9 & 9 & 9 \\ 
\hline 
i - j & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 & 30 & \cdots\\ 
\hline 
f(i-j) & 10 & 10 & 10 & 10 & 10 & 10 & 10 & 11 & 11 & 11 & 11 & 11 & 11 & 11 & 11 & \cdots \\ 
\hline\end{array}
$$

具体的映射代码，读者自行看源码就好。这个设计的思路其实也很直观，就是比较邻近的位置（0～7），我们需要比较得精细一些，所以给它们都分配一个独立的位置编码，至于稍远的位置（比如 8～11），我们不用区分得太清楚，所以它们可以共用一个位置编码，距离越远，共用的范围就可以越大，直到达到指定范围再 clip。

### DeBERTa 式

DeBERTa 也是微软搞的，去年 6 月就发出来了，论文为[《DeBERTa: Decoding-enhanced BERT with Disentangled Attention》](https://arxiv.org/abs/2006.03654)，最近又小小地火了一把，一是因为它正式中了 ICLR 2021，二则是它登上 [SuperGLUE](https://super.gluebenchmark.com/) 的榜首，成绩稍微超过了 T5。

其实 DeBERTa 的主要改进也是在位置编码上，同样还是从展开式 $(7)$ 出发，T5 是干脆去掉了第 2、3 项，只保留第 4 项并替换为相对位置编码，而 DeBERTa 则刚刚相反，它扔掉了第 4 项，保留第 2、3 项并且替换为相对位置编码（果然，科研就是枚举所有的排列组合看哪个最优）：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^{\top} = \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} + \boldsymbol{x}_i \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\color{green}{\boldsymbol{R}_{i,j}^{\top}} + \color{green}{\boldsymbol{R}_{j,i}} \boldsymbol{W}_Q \boldsymbol{W}_K^{\top}\boldsymbol{x}_j^{\top} \tag{10}
$$

至于 $\boldsymbol{R}_{i,j}$ 的设计也是像式 $(6)$ 那样进行截断的，没有特别的地方。

不过，DeBERTa 比较有意思的地方，是提供了使用相对位置和绝对位置编码的一个新视角，它指出 NLP 的大多数任务可能都只需要相对位置信息，但确实有些场景下绝对位置信息更有帮助，于是它将整个模型分为两部分来理解。以 Base 版的 MLM 预训练模型为例，它一共有 13 层，前 11 层只是用相对位置编码，这部分称为 Encoder，后面 2 层加入绝对位置信息，这部分它称之为 Decoder，还弄了个简称 EMD（Enhanced Mask Decoder）；至于下游任务的微调截断，则是使用前 11 层的 Encoder 加上 1 层的 Decoder 来进行。

SuperGLUE 上的成绩肯定了 DeBERTa 的价值，但是它论文的各种命名真的是让人觉得极度不适，比如它自称的“Encoder”、“Decoder”就很容易让人误解这是一个 Seq2Seq 模型，比如 EMD 这个简称也跟 Earth Mover's Distance 重名。虽然有时候重名是不可避免的，但它重的名都是 ML 界大家都比较熟悉的对象，相当容易引起误解，真不知道作者是怎么想的...

## 3. 博采众长的旋转式位置编码

前面我们对原始的 Sinusoidal 位置编码做了较为详细的推导和理解，总的感觉是 Sinusoidal位 置编码是一种“想要成为相对位置编码的绝对位置编码”。一般来说，绝对位置编码具有实现简单、计算速度快等优点，而相对位置编码则直接地体现了相对位置信号，跟我们的直观理解吻合，实际性能往往也更好。由此可见，如果可以通过绝对位置编码的方式实现相对位置编码，那么就是“集各家之所长”、“鱼与熊掌兼得”了。Sinusoidal 位置编码隐约做到了这一点，但并不够好。

本文将会介绍我们自研的 Rotary Transformer (RoFormer) 模型，它的主要改动是应用了笔者构思的“旋转式位置编码 (Rotary Position Embedding，RoPE)”，这是一种配合 Attention 机制能达到“绝对位置编码的方式实现相对位置编码”的设计。而也正因为这种设计，它还是目前唯一一种可用于线性 Attention 的相对位置编码。

### 基本思路

在旋转式位置编码 RoPE 中，我们的出发点就是“通过绝对位置编码的方式实现相对位置编码”，这样做既有理论上的优雅之处，也有实践上的实用之处，比如它可以拓展到线性 Attention 中就是主要因为这一点。

为了达到这个目的，我们假设通过下述运算来给 $\boldsymbol{q},\boldsymbol{k}$ 添加绝对位置信息：

$$
\tilde{\boldsymbol{q}}_m = \boldsymbol{f}(\boldsymbol{q}, m), \quad\tilde{\boldsymbol{k}}_n = \boldsymbol{f}(\boldsymbol{k}, n) \tag{11}
$$

也就是说，我们分别为 $\boldsymbol{q},\boldsymbol{k}$ 设计操作 $\boldsymbol{f}(\cdot, m),\boldsymbol{f}(\cdot, n)$，使得经过该操作后，$\tilde{\boldsymbol{q}}_m,\tilde{\boldsymbol{k}}_n$ 就带有了位置 $m,n$ 的绝对位置信息。Attention 的核心运算是内积，所以我们希望的内积的结果带有相对位置信息，因此假设存在恒等关系：

$$
\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n) \tag{12}
$$

所以我们要求出该恒等式的一个（尽可能简单的）解。求解过程还需要一些初始条件，显然我们可以合理地设 $\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$ 和 $\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$。

### 求解过程

我们先考虑二维情形，然后借助复数来求解。在复数中有 $\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]$， $\text{Re}[]$ 代表复数的实部，所以我们有

$$
\text{Re}[\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n)] = g(\boldsymbol{q},\boldsymbol{k},m-n) \tag{13}
$$

简单起见，我们假设存在复数 $\boldsymbol{g}(\boldsymbol{q},\boldsymbol{k},m-n)$，使得 $\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n) = \boldsymbol{g}(\boldsymbol{q},\boldsymbol{k},m-n)$，然后我们用复数的指数形式，设

$$
\begin{aligned} 
\boldsymbol{f}(\boldsymbol{q}, m) =&\, R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} \\ 
\boldsymbol{f}(\boldsymbol{k}, n) =&\, R_f (\boldsymbol{k}, n)e^{\text{i}\Theta_f(\boldsymbol{k}, n)} \\ 
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n)e^{\text{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)} \\ 
\end{aligned} \tag{14}
$$

那么代入方程后就得到方程组

$$
\begin{aligned} 
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n) \\ 
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, n) =&\, \Theta_g (\boldsymbol{q}, \boldsymbol{k}, m-n) 
\end{aligned} \tag{15}
$$

对于第一个方程，代入 $m=n$ 得到

$$
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, m) = R_g (\boldsymbol{q}, \boldsymbol{k}, 0) = R_f (\boldsymbol{q}, 0) R_f (\boldsymbol{k}, 0) = \Vert \boldsymbol{q}\Vert \Vert \boldsymbol{k}\Vert \tag{16}
$$

最后一个等号源于初始条件 $\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$ 和 $\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$。所以现在我们可以很简单地设 $R_f (\boldsymbol{q}, m)=\Vert \boldsymbol{q}\Vert, R_f (\boldsymbol{k}, m)=\Vert \boldsymbol{k}\Vert$，即它不依赖于 $m$。至于第二个方程，同样代入 $m=n$ 得到

$$
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, m) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f (\boldsymbol{q}, 0) - \Theta_f (\boldsymbol{k}, 0) =  \Theta (\boldsymbol{q}) - \Theta (\boldsymbol{k}) \tag{17}
$$

这里的 $\Theta (\boldsymbol{q}),\Theta (\boldsymbol{k})$ 是 $\boldsymbol{q},\boldsymbol{k}$ 本身的幅角，最后一个等号同样源于初始条件。根据上式得到 $\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q}) = \Theta_f (\boldsymbol{k}, m) - \Theta (\boldsymbol{k})$，所以 $\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q})$ 应该是一个只与 $m$ 相关、跟 $\boldsymbol{q}$ 无关的函数，记为 $\varphi(m)$，即 $\Theta_f (\boldsymbol{q}, m) = \Theta (\boldsymbol{q}) + \varphi(m)$。接着代入 $n=m−1$，整理得到

$$
\varphi(m) - \varphi(m-1) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 1) + \Theta (\boldsymbol{k}) - \Theta (\boldsymbol{q}) \tag{18}
$$

即 $\{\varphi(m)\}$ 是等差数列，设右端为 $\theta$，那么就解得 $\varphi(m)=m\theta$。

### 编码形式

综上，我们得到二维情况下用复数表示的 RoPE：

$$
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} 
= \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}\tag{19}
$$

根据复数乘法的几何意义，该变换实际上对应着向量的旋转，所以我们称之为“旋转式位置编码”，它还可以写成矩阵形式：

$$
 \boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix} \tag{20}
$$

由于内积满足线性叠加性，因此任意偶数维的 RoPE，我们都可以表示为二维情形的拼接，即

$$
\scriptsize{\underbrace{\begin{pmatrix} 
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ 
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}} \tag{21}
$$

也就是说，给位置为 $m$ 的向量 $\boldsymbol{q}$ 乘上矩阵 $\boldsymbol{\mathcal{R}}_m$、位置为 $n$ 的向量 $\boldsymbol{k}$ 乘上矩阵 $\boldsymbol{\mathcal{R}}_n$，用变换后的 $\boldsymbol{Q},\boldsymbol{K}$ 序列做 Attention，那么 Attention 就自动包含相对位置信息了，因为成立恒等式：

$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k} \tag{22}
$$

值得指出的是，$\boldsymbol{\mathcal{R}}_m$ 是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

由于 $\boldsymbol{\mathcal{R}}_m$ 的稀疏性，所以直接用矩阵乘法来实现会很浪费算力，推荐通过下述方式来实现 RoPE：

$$
\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix} \tag{23}
$$

其中 $\otimes$ 是逐位对应相乘，即 Numpy、Tensorflow 等计算框架中的 $*$ 运算。从这个实现也可以看到，RoPE 可以视为是乘性位置编码的变体。

### 远程衰减

可以看到，RoPE 形式上和 Sinusoidal 位置编码有点相似，只不过 Sinusoidal 位置编码是加性的，而 RoPE 可以视为乘性的。在 $\theta_i$ 的选择上，我们同样沿用了 Sinusoidal 位置编码的方案，即 $\theta_i = 10000^{-2i/d}$，它可以带来一定的远程衰减性。

具体证明如下：将 $\boldsymbol{q},\boldsymbol{k}$ 两两分组后，它们加上 RoPE 后的内积可以用复数乘法表示为

$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \text{Re}\left[\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i}\right]\tag{24}
$$

记 $h\_i = \boldsymbol{q}\_{[2i:2i+1]}\boldsymbol{k}\_{[2i:2i+1]}^*, S\_j = \sum\limits\_{i=0}^{j-1} e^{\text{i}(m-n)\theta\_i}$，并约定 $h\_{d/2}=0,S\_0=0$，那么由 [Abel 变换（分部求和法）](https://zh.wikipedia.org/wiki/分部求和法)可以得到：

$$
\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i} = \sum_{i=0}^{d/2-1} h_i (S_{i 
+1} - S_i)  = -\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\tag{25}
$$

所以

$$
\begin{aligned} 
\left|\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i}\right| =&\, \left|\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\right| \\ 
\leq&\, \sum_{i=0}^{d/2-1} |S_{i+1}| |h_{i+1} - h_i| \\ 
\leq&\, \left(\max_i |h_{i+1} - h_i|\right)\sum_{i=0}^{d/2-1} |S_{i+1}| 
\end{aligned}\tag{26}
$$

因此我们可以考察 $\frac{1}{d/2}\sum\limits\_{i=1}^{d/2} \|S\_i\|$ 随着相对距离的变化情况来作为衰减性的体现，Mathematica 代码如下：

```mathematica
d = 128;
\[Theta][t_] = 10000^(-2*t/d);
f[m_] = Sum[
    Norm[Sum[Exp[I*m*\[Theta][i]], {i, 0, j}]], {j, 0, d/2 - 1}]/(d/2);
Plot[f[m], {m, 0, 256}, AxesLabel -> {相对距离, 相对大小}]
```

结果如下图：

<img src="/img/article/transformer-position-encoding/attenuation_of_rope.png" width="500px" style="display: block; margin: auto;"/>

<center>RoPE 的远程衰减性（d=128）</center>

从图中我们可以可以看到随着相对距离的变大，内积结果有衰减趋势的出现。因此，选择 $\theta_i = 10000^{-2i/d}$，确实能带来一定的远程衰减性。当然，同上一篇文章说的一样，能带来远程衰减性的不止这个选择，几乎任意的光滑单调函数都可以，这里只是沿用了已有的选择而已。笔者还试过以 $\theta_i = 10000^{-2i/d}$ 为初始化，将 $\theta_i$ 视为可训练参数，然后训练一段时间后发现 $\theta_i$ 并没有显著更新，因此干脆就直接固定 $\theta_i = 10000^{-2i/d}$ 了。

### 线性场景

最后，我们指出，RoPE 是目前唯一一种可以用于线性 Attention 的相对位置编码。这是因为其他的相对位置编码，都是直接基于 Attention 矩阵进行操作的，但是线性 Attention 并没有事先算出 Attention 矩阵，因此也就不存在操作 Attention 矩阵的做法，所以其他的方案无法应用到线性 Attention 中。而对于 RoPE 来说，它是用绝对位置编码的方式来实现相对位置编码，不需要操作 Attention 矩阵，因此有了应用到线性 Attention 的可能性。

关于线性 Attention 的介绍，这里不再重复，有需要的读者请参考[《线性Attention的探索：Attention必须有个Softmax吗？》](https://kexue.fm/archives/7546)。线性 Attention 的常见形式是：

$$
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)} = \frac{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\tag{27}
$$

其中 $\phi,\varphi$ 是值域非负的激活函数。可以看到，线性 Attention 也是基于内积的，所以很自然的想法是可以将 RoPE 插入到内积中：

$$
\frac{\sum\limits_{j=1}^n [\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]\boldsymbol{v}_j}{\sum\limits_{j=1}^n [\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]}\tag{28}
$$

但这样存在的问题是，内积 $[\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]$ 可能为负数，因此它不再是常规的概率注意力，而且分母有为 0 的风险，可能会带来优化上的不稳定。考虑到 $\boldsymbol{\mathcal{R}}_i,\boldsymbol{\mathcal{R}}_j$ 都是正交矩阵，它不改变向量的模长，因此我们可以抛弃常规的概率归一化要求，使用如下运算作为一种新的线性 Attention：

$$
\frac{\sum\limits_{j=1}^n [\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\tag{29}
$$

也就是说，RoPE 只插入分子中，而分母则不改变，这样的注意力不再是基于概率的（注意力矩阵不再满足非负归一性），但它某种意义上来说也是一个归一化方案，而且也没有证据表明非概率式的注意力就不好（比如 [Nyströmformer](https://kexue.fm/archives/8180) 也算是没有严格依据概率分布的方式构建注意力），所以我们将它作为候选方案之一进行实验，而我们初步的实验结果显示这样的线性 Attention 也是有效的。

此外，笔者在[《线性Attention的探索：Attention必须有个Softmax吗？》](https://kexue.fm/archives/7546)中还提出过另外一种线性 Attention 方案：$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \left( \frac{\boldsymbol{q}_i}{\Vert \boldsymbol{q}_i\Vert}\right)^{\top}\left(\frac{\boldsymbol{k}_j}{\Vert \boldsymbol{k}_j\Vert}\right)$，它不依赖于值域的非负性，而 RoPE 也不改变模长，因此 RoPE 可以直接应用于此类线性 Attention，并且不改变它的概率意义。

### 模型开源

RoFormer 的第一版模型，我们已经完成训练并开源到了 Github 中：

> **RoFormer：**[https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)

简单来说，RoFormer 是一个绝对位置编码替换为 RoPE 的 [WoBERT](https://github.com/ZhuiyiTechnology/WoBERT) 模型，它跟其他模型的结构对比如下：

$$
\begin{array}{c|cccc} 
\hline 
& \text{BERT} & \text{WoBERT} & \text{NEZHA} & \text{RoFormer} \\ 
\hline 
\text{token单位} & \text{字} & \text{词} & \text{字} & \text{词} & \\ 
\text{位置编码} & \text{绝对位置} & \text{绝对位置} & \text{经典式相对位置} & \text{RoPE}\\ 
\hline 
\end{array}
$$

在预训练上，我们以 WoBERT Plus 为基础，采用了多个长度和 batch size 交替训练的方式，让模型能提前适应不同的训练场景：

$$
\begin{array}{c|ccccc} 
\hline 
& \text{maxlen} & \text{batch size} & \text{训练步数} & \text{最终loss} & \text{最终acc}\\ 
\hline 
1 & 512 & 256 & 20\text{万} & 1.73 & 65.0\%\\ 
2 & 1536 & 256 & 1.25\text{万} & 1.61 & 66.8\%\\ 
3 & 256 & 256 & 12\text{万} & 1.75 & 64.6\%\\ 
4 & 128 & 512 & 8\text{万} & 1.83 & 63.4\%\\ 
5 & 1536 & 256 & 1\text{万} & 1.58 & 67.4\%\\ 
6 & 512 & 512 & 3\text{万} & 1.66 & 66.2\%\\ 
\hline 
\end{array}
$$

从表格还可以看到，增大序列长度，预训练的准确率反而有所提升，这侧面体现了 RoFormer 长文本语义的处理效果，也体现了 RoPE 具有良好的外推能力。在短文本任务上，RoFormer 与 WoBERT 的表现类似，RoFormer 的主要特点是可以直接处理任意长的文本。下面是我们在 [CAIL2019-SCM](https://arxiv.org/abs/1911.08962) 任务上的实验结果：

$$
\begin{array}{c|cc} 
\hline 
& \text{验证集} & \text{测试集} \\ 
\hline 
\text{BERT-512} & 64.13\% & 67.77\% \\ 
\text{WoBERT-512} & 64.07\% & 68.10\% \\ 
\text{RoFormer-512} & 64.13\% & 68.29\% \\ 
\text{RoFormer-1024} & \textbf{66.07%} & \textbf{69.79%} \\ 
\hline 
\end{array}
$$

其中--后面的参数是微调时截断的 maxlen，可以看到 RoFormer 确实能较好地处理长文本语义，至于设备要求，在 24G 显存的卡上跑 maxlen=1024，batch_size 可以跑到 8 以上。目前中文任务中笔者也就找到这个任务比较适合作为长文本能力的测试，所以长文本方面只测了这个任务，欢迎读者进行测试或推荐其他评测任务。

当然，尽管理论上 RoFormer 能处理任意长度的序列，但目前 RoFormer 还是具有平方复杂度的，我们也正在训练基于线性 Attention 的 RoFormer 模型，实验完成后也会开源放出，请大家期待。

（注：RoPE 和 RoFormer 已经整理成文[《RoFormer: Enhanced Transformer with Rotary Position Embedding》](https://arxiv.org/abs/2104.09864)提交到了 Arxiv，欢迎使用和引用哈哈～）

> 本文转载自[《让研究人员绞尽脑汁的Transformer位置编码》](https://kexue.fm/archives/8130)和[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://kexue.fm/archives/8265)，作者：苏剑林，部分内容有修改。
