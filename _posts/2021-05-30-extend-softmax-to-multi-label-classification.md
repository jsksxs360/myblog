---
layout: article
title: 将“softmax+交叉熵”推广到多标签分类问题
author: 苏剑林
tags:
    - 机器学习
mathjax: true
---

> 转载自[《将“softmax+交叉熵”推广到多标签分类问题》](https://kexue.fm/archives/7359)，作者：苏剑林。

一般来说，在处理常规的多分类问题时，我们会在模型的最后用一个全连接层输出每个类的分数，然后用 softmax 激活并用交叉熵作为损失函数。在这篇文章里，我们尝试将“softmax+交叉熵”方案推广到多标签分类场景，希望能得到用于多标签分类任务的、不需要特别调整类权重和阈值的 loss。

![mushroom](/img/article/extend-softmax-to-multi-label-classification/mushroom.jpg)

<center>类别不平衡</center>

## 单标签到多标签

一般来说，多分类问题指的就是单标签分类问题，即从 $n$ 个候选类别中选 $1$ 个目标类别。假设各个类的得分分别为 $s_1,s_2,  \dots,s_n$，目标类为 $t\in\{1,2,\dots,n\}$，那么所用的 loss 为

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}= - s_t + \log \sum\limits_{i=1}^n e^{s_i}\label{eq:log-softmax}\tag{1}
$$

这个 loss 的优化方向是让目标类的得分 $s_t$ 变为 $s_1,s_2,\dots,s_t$ 中的最大值。

> 关于 softmax 的相关内容，还可以参考[《寻求一个光滑的最大值函数》](https://kexue.fm/archives/3290)、[《函数光滑化杂谈：不可导函数的可导逼近》](https://kexue.fm/archives/6620)等文章。

现在我们转到多标签分类问题，即从 $n$ 个候选类别中选 $k$ 个目标类别。这种情况下我们一种朴素的做法是用 sigmoid 激活，然后变成 $n$ 个二分类问题，用二分类的交叉熵之和作为 loss。显然，当 $n\gg k$ 时，这种做法会面临着严重的类别不均衡问题，这时候需要一些平衡策略，比如手动调整正负样本的权重、[focal loss](https://kexue.fm/archives/4733) 等。训练完成之后，还需要根据验证集来进一步确定最优的阈值。

这时候，一个很自然的困惑就是：**为什么“$n$ 选 $k$”要比“$n$ 选 $1$”多做那么多工作？**

笔者认为这是很不科学的事情，毕竟直觉上 $n$ 选 $k$ 应该只是 $n$ 选 $1$ 自然延伸，所以不应该要比 $n$ 要多做那么多事情，就算 $n$ 选 $k$ 要复杂一些，难度也应该是慢慢过渡的，但如果变成多个二分类的话，$n$ 选 $1$ 反而是最难的，因为这时候类别最不均衡。而从形式上来看，单标签分类比多标签分类要容易，就是因为单标签有“softmax+交叉熵”可以用，它不会存在类别不平衡的问题，而多标签分类中的“sigmoid+交叉熵”就存在不平衡的问题。

所以，理想的解决办法应该就是将“softmax+交叉熵”推广到多标签分类上去。

## 众里寻她千百度

为了考虑这个推广，笔者进行了多次尝试，也否定了很多结果，最后确定了一个相对来说比较优雅的方案：构建组合形式的 softmax 来作为单标签 softmax 的推广。在这部分内容中，我们会先假设 $k$ 是一个固定的常数，然后再讨论一般情况下 $k$ 的自动确定方案，最后确实能得到一种有效的推广形式。

### 组合 softmax

首先，我们考虑 $k$ 是一个固定常数的情景，这意味着预测的时候，我们直接输出得分最高的 $k$ 个类别即可。那训练的时候呢？作为 softmax 的自然推广，我们可以考虑用下式作为 loss：

$$
-\log \frac{e^{s_{t_1}+s_{t_2}+\dots+s_{t_k}}}{\sum\limits_{1\leq i_1 < i_2 < \cdots < i_k\leq n}e^{s_{i_1}+s_{i_2}+\dots+s_{i_k}}}=\log Z_k - (s_{t_1}+s_{t_2}+\dots+s_{t_k})\tag{2}
$$

其中 $t_1,t_2,\dots,t_k$ 是 $k$ 个目标标签，$Z_k = \sum\limits_{1\leq i_1 < i_2 < \cdots < i_k\leq n}e^{s_{i_1}+s_{i_2}+\dots+s_{i_k}}$ 是配分函数。很显然，上式是以任何 $k$ 个类别总得分 $s_{i_1}+s_{i_2}+\dots+s_{i_k}$ 为基本单位所构造的 softmax，所以它算是单标签 softmax 的合理推广。又或者理解为还是一个单标签分类问题，只不过这是 $C_n^k$ 选 $1$ 问题。

在这个方案之中，比较困难的地方是 $Z_k$ 的计算，它是 $C_n^k$ 项总得分的指数和。不过，我们可以利用[牛顿恒等式](https://en.wikipedia.org/wiki/Newton's_identities)来帮助我们递归计算。设 $S_k = \sum\limits_{i=1}^n e^{k s_i}$，那么

$$
\begin{aligned} 
Z_1 =&\, S_1\\ 
2Z_2 =&\, Z_1 S_1  - S_2\\ 
3Z_3 = &\, Z_2 S_1 - Z_1 S_2 + S_3\\ 
\vdots\\ 
k Z_k = &\, Z_{k-1} S_1 - Z_{k-2} S_2 + \dots + (-1)^{k-2} Z_1 S_{k-1} + (-1)^{k-1} S_k 
\end{aligned}\tag{3}
$$

所以为了计算 $Z_k$，我们只需要递归计算 $k$ 步，这可以在合理的时间内计算出来。预测阶段，则直接输出分数最高的 $k$ 个类就行。

### 自动确定阈值

上述讨论的是输出数目固定的多标签分类问题，但一般的多标签分类的目标标签数是不确定的。为此，我们确定一个最大目标标签数 $K\geq k$，并添加一个 $0$ 标签作为填充标签，此时 loss 变为

$$
\log \overline{Z}_K - (s_{t_1}+s_{t_2}+\dots+s_{t_k}+\underbrace{s_0+\dots+s_0}_{K-k\text{个}})\tag{4}
$$

而

$$
\begin{aligned} 
\overline{Z}_K =&\, \sum\limits_{1\leq i_1 < i_2 < \cdots < i_K\leq n}e^{s_{i_1}+s_{i_2}+\dots+s_{i_K}} + \sum\limits_{0 = i_1 = \dots = i_j < i_{j+1} < \cdots < i_K\leq n}e^{s_{i_1}+s_{i_2}+\dots+s_{i_K}}\\ 
=&\, Z_K + e^{s_0} \overline{Z}_{K-1} 
\end{aligned}\tag{5}
$$

看上去很复杂，其实很简单，还是以 $K$ 个类别总得分为基本单位，但是允许且仅允许 $0$ 类重复出现。预测的时候，仍然是输出分数最大的 $K$ 个类，但允许重复输出 $0$ 类，等价的效果是**以 $s_0$ 为阈值，只输出得分大于 $s_0$ 的类**。最后的式子显示 $\overline{Z}_K$ 也可以通过递归来计算，所以实现上是没有困难的。

## 暮然回首阑珊处

看上去“众里寻她千百度”终究是有了结果：理论有了，实现也不困难，接下来似乎就应该做实验看效果了吧？效果好的话，甚至可以考虑发 paper 了吧？看似一片光明前景呢！然而～

幸运或者不幸，在验证了它的有效性的同时，笔者请教了一些前辈大神，在他们的提示下翻看了之前没细看的 [Circle Loss](https://arxiv.org/abs/2002.10857)，看到了它里边统一的 loss 形式（原论文的公式(1)），然后意识到了这个统一形式蕴含了一个更简明的推广方案。

所以，不幸的地方在于，已经有这么一个现成的更简明的方案了，所以不管如何“众里寻她千百度”，都已经没有太大意义了；而幸运的地方在于，还好找到了这个更好的方案，要不然屁颠屁颠地把前述方案写成文章发出来，还不如现成的方案简单有效，那时候丢人就丢大发了～

### 统一的 loss 形式

让我们换一种形式看单标签分类的交叉熵 $(1)$：

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}=-\log \frac{1}{\sum\limits_{i=1}^n e^{s_i-s_t}}=\log \sum\limits_{i=1}^n e^{s_i-s_t}=\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)\tag{6}
$$

为什么这个 loss 会有效呢？在文章[《寻求一个光滑的最大值函数》](https://kexue.fm/archives/3290)、[《函数光滑化杂谈：不可导函数的可导逼近》](https://kexue.fm/archives/6620)中我们都可以知道，$\text{logsumexp}$ 实际上就是 $\max$ 的光滑近似，所以我们有：

$$
\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)\approx max\begin{pmatrix}0 \\ s_1 - s_t \\ \vdots \\ s_{t-1} - s_t \\ s_{t+1} - s_t \\ \vdots \\ s_n - s_t\end{pmatrix}\tag{7}
$$

这个 loss 的特点是，所有的非目标类得分 $\{s_1,\cdots,s_{t-1},s_{t+1},\cdots,s_n\}$ 跟目标类得分 $\{s_t\}$ 两两作差比较，它们的差的最大值都要尽可能小于零，所以实现了“目标类得分都大于每个非目标类的得分”的效果。

所以，假如是有多个目标类的多标签分类场景，我们也希望“每个目标类得分都不小于每个非目标类的得分”，所以下述形式的 loss 就呼之欲出了：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:unified}\tag{8}
$$

其中 $\Omega_{pos},\Omega_{neg}$ 分别是正负样本的类别集合。这个 loss 的形式很容易理解，就是我们希望 $s_i < s_j$，就往 $\log$ 里边加入 $e^{s_i - s_j}$ 这么一项。如果补上缩放因子 $\gamma$ 和间隔 $m$，就得到了 Circle Loss 论文里边的统一形式：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{\gamma(s_i-s_j + m)}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{\gamma (s_i + m)}\sum\limits_{j\in\Omega_{pos}} e^{-\gamma s_j}\right)\tag{9}
$$

说个题外话，上式就是 Circle Loss 论文的公式 (1)，但原论文的公式 (1) 不叫 Circle Loss，原论文的公式 (4) 才叫 Circle Loss，所以不能把上式叫做 Circle Loss。但笔者认为，整篇论文之中最有意思的部分还数公式 (1)。

### 用于多标签分类

$\gamma$ 和 $m$ 一般都是度量学习中才会考虑的，所以这里我们还是只关心式 $(8)$。如果 $n$ 选 $k$ 的多标签分类中 $k$ 是固定的话，那么直接用式 $(8)$ 作为 loss 就行了，然后预测时候直接输出得分最大的 $k$ 个类别。

对于 $k$ 不固定的多标签分类来说，我们就需要一个阈值来确定输出哪些类。为此，我们同样引入一个额外的 $0$ 类，希望目标类的分数都大于 $s_0$，非目标类的分数都小于 $s_0$，而前面已经已经提到过，“希望 $s_i < s_j$ 就往 $\log$ 里边加入 $e^{s_i - s_j}$”，所以现在式 $(8)$ 变成：

$$
\begin{aligned} 
&\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}+\sum\limits_{i\in\Omega_{neg}} e^{s_i-s_0}+\sum\limits_{j\in\Omega_{pos}} e^{s_0-s_j}\right)\\ 
=&\log \left(e^{s_0} + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(e^{-s_0} + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\\ 
\end{aligned}\tag{10}
$$

如果指定阈值为 0，那么就简化为

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(1 + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:final}\tag{11}
$$

这便是我们最终得到的 Loss 形式了——“softmax + 交叉熵”在多标签分类任务中的自然、简明的推广，它没有类别不均衡现象，因为它不是将多标签分类变成多个二分类问题，而是变成目标类别得分与非目标类别得分的两两比较，并且借助于 $\text{logsumexp}$ 的良好性质，自动平衡了每一项的权重。

这里给出 Keras 下的参考实现：

```python
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss
```

## 所以，结论就是

所以，最终结论就是式 $(11)$，它就是本文要寻求的多标签分类问题的统一 loss，欢迎大家测试并报告效果。笔者也实验过几个多标签分类任务，均能媲美精调权重下的二分类方案。

要提示的是，除了标准的多标签分类问题外，还有一些常见的任务形式也可以认为是多标签分类，比如基于0/1标注的序列标注，典型的例子是笔者的[“半指针-半标注”标注设计](https://kexue.fm/archives/6671)。因此，从这个角度看，能被视为多标签分类来测试式 $(11)$ 的任务就有很多了，笔者也确实在之前的三元组抽取例子 [task_relation_extraction.py](https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py) 中尝试了 $(11)$，最终能取得跟[这里](https://kexue.fm/archives/7161#类别失衡)一致的效果。

当然，最后还是要说明一下，虽然理论上式 $(11)$ 作为多标签分类的损失函数能自动地解决很多问题，但终究是不存在绝对完美、保证有提升的方案，所以当你用它替换掉你原来多标签分类方案时，也不能保证一定会有提升，尤其是当你原来已经通过精调权重等方式处理好类别不平衡问题的情况下，式 $(11)$ 的收益是非常有限的。毕竟式 $(11)$ 的初衷，只是让我们在不需要过多调参的的情况下达到大部分的效果。

> 转载自[《将“softmax+交叉熵”推广到多标签分类问题》](https://kexue.fm/archives/7359)，作者：苏剑林。

