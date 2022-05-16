---
layout: article
title: 多标签“Softmax+交叉熵”
author: 苏剑林
tags:
    - 机器学习
mathjax: true
---

单标签分类问题负责从 $n$ 个候选类别中选出 $1$ 个目标类别，常规的操作就是在最后一个全连接层输出每个类的分数，然后用 softmax 激活并用交叉熵作为损失函数。假设各个类的得分分别为 $s_1, s_2, ..., s_n$，目标类为 $t\in\{1,2,\dots,n\}$，那么所用的 loss 就为：

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}= - s_t + \log \sum\limits_{i=1}^n e^{s_i}\label{eq:log-softmax}\tag{1}
$$

这个 loss 的优化方向是让目标类的得分 $s_t$ 变为 $s_1,s_2,\dots,s_n$ 中的最大值。

而多标签分类问题则是从 $n$ 个候选类别中选 $k$ 个目标类别，最简单的做法是用 sigmoid 激活，然后变成 $n$ 个二分类问题，用二分类的交叉熵之和作为 loss。但是，当 $n\gg k$ 时，这种做法会面临着严重的类别不均衡问题，通常需要采取一些平衡策略（比如手动调整正负样本的权重、focal loss 等），还需要根据验证集来确定最优的阈值。

幸运的是，知名博主[苏剑林](https://kexue.fm/)将“softmax+交叉熵”方案推广到了多标签分类场景，并且 loss 不需要特别调整类权重和阈值，是一个非常实用的设计。下面本文将简单介绍其核心思想。

## Circle Loss

首先，让我们换一种形式看单标签分类的交叉熵 $(1)$：

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}=-\log \frac{1}{\sum\limits_{i=1}^n e^{s_i-s_t}}=\log \sum\limits_{i=1}^n e^{s_i-s_t}=\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)\tag{2}
$$

而 $\text{logsumexp}$ 实际上就是 $\max$ 的光滑近似，所以有：

$$
\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)\approx \max\begin{pmatrix}0 \\ s_1 - s_t \\ \vdots \\ s_{t-1} - s_t \\ s_{t+1} - s_t \\ \vdots \\ s_n - s_t\end{pmatrix}\tag{3}
$$

因此，交叉熵 loss 相当于将所有非目标类得分 $\{s_1,\cdots,s_{t-1},s_{t+1},\cdots,s_n\}$ 跟目标类得分 $\{s_t\}$ 两两作差比较，然后要求差的最大值都要尽可能小于零，这样就实现了“目标类得分大于每个非目标类的得分”的效果。

对于有多个目标类的多标签分类场景，实际上我们也是希望“每个目标类得分都不小于每个非目标类的得分”，所以下述形式的 loss 就呼之欲出了：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:unified}\tag{4}
$$

其中 $\Omega_{pos},\Omega_{neg}$ 分别是样本的正负类别集合，即如果要 $s_i < s_j$，就往 $\log$ 里边加入一项 $e^{s_i - s_j}$。如果补上缩放因子 $\gamma$ 和间隔 $m$，就得到了 [Circle Loss](https://arxiv.org/abs/2002.10857) 论文里边的统一形式：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{\gamma(s_i-s_j + m)}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{\gamma (s_i + m)}\sum\limits_{j\in\Omega_{pos}} e^{-\gamma s_j}\right)\tag{5}
$$

$\gamma$ 和 $m$ 一般都是度量学习中才会考虑的，所以这里我们只关心式 $(4)$。

## 用于多标签分类

如果 $n$ 选 $k$ 的多标签分类中 $k$ 是固定的话，那么直接用式 $(4)$ 作为 loss 就行了，然后预测时候直接输出得分最大的 $k$ 个类别。对于 $k$ 不固定的多标签分类来说，就需要一个阈值来确定输出哪些类。

为此，作者引入了一个额外的 $0$ 类，希望目标类的分数都大于 $s_0$，非目标类的分数都小于 $s_0$。前面已经提过，希望 $s_i < s_j$ 就往 $\log$ 里边加入 $e^{s_i - s_j}$，所以现在式 $(4)$ 变成：

$$
\begin{aligned} 
&\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}+\sum\limits_{i\in\Omega_{neg}} e^{s_i-s_0}+\sum\limits_{j\in\Omega_{pos}} e^{s_0-s_j}\right)\\ 
=&\log \left(e^{s_0} + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(e^{-s_0} + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\\ 
\end{aligned} \tag{6}
$$

如果指定阈值为 0，上式就简化为：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(1 + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:final}\tag{7}
$$

这就是作者提出的 Loss 形式了——“softmax+交叉熵”在多标签分类任务中的自然、简明的推广，它没有类别不均衡现象，并且借助于 $\text{logsumexp}$ 的良好性质，自动平衡了每一项的权重。

Keras 下的参考实现为：

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

## 软标签版本

从式 $(7)$ 可以看到，这个损失函数只适用于“硬标签”，这意味着没法使用 label smoothing、[mixup](https://kexue.fm/archives/5693) 等技巧。因此，作者后来又提出了该损失函数的一个软标签版本。

前面说过，多标签分类的经典方案就是转化为多个二分类问题，每个类别用 sigmoid 函数 $\sigma(x)=\frac{1}{(1+e^{-x})}$ 激活，然后各自用二分类交叉熵损失。当正负类别极其不平衡时，这种做法的表现通常会比较糟糕。实际上，多个“sigmoid+二分类交叉熵”可以适当地改写成：

$$
\begin{aligned} 
&\,-\sum_{j\in\Omega_{pos}}\log\sigma(s_j)-\sum_{i\in\Omega_{neg}}\log(1-\sigma(s_i))\\ 
=&\,\log\prod_{j\in\Omega_{pos}}(1+e^{-s_j})+\log\prod_{i\in\Omega_{neg}}(1+e^{s_i})\\ 
=&\,\log\left(1+\sum_{j\in\Omega_{pos}}e^{-s_j}+\cdots\right)+\log\left(1+\sum_{i\in\Omega_{neg}}e^{s_i}+\cdots\right) 
\end{aligned}\label{eq:link}\tag{8}
$$

可以看到，式 $(7)$ 正好是上式去掉了 $\cdots$ 所表示的高阶项！在正负类别不平衡时，这些高阶项占据了过高的权重，加剧了不平衡问题；而去掉这些高阶项后，并没有改变损失函数的作用（希望正类得分大于0、负类得分小于0），同时因为括号内的求和数跟类别数是线性关系，因此正负类各自的损失差距不会太大。

这个联系告诉我们，要寻找式 $(7)$ 的软标签版本，同样可以从多个“sigmoid+二分类交叉熵”的软标签版本出发，然后尝试去掉高阶项。所谓软标签，指的是标签不再是 0 或 1，而是 0～1 之间的任意实数，表示属于该类的可能性。对于二分类交叉熵，它的软标签版本很简单：

$$
p\log\sigma(s)-(1-p)\log(1-\sigma(s))\tag{9}
$$

这里 $p$ 就是软标签，而 $s$ 就是对应的打分。模仿过程 $(8)$ 可以得到：

$$
\begin{aligned} 
&\,-\sum_i p_i\log\sigma(s_i)-\sum_i (1-p_i)\log(1-\sigma(s_i))\\ 
=&\,\log\prod_i(1+e^{-s_i})^{p_i}+\log\prod_i (1+e^{s_i})^{1-p_i}\\ 
=&\,\log\prod_i(1+p_i e^{-s_i} + \cdots)+\log\prod_i (1+(1-p_i)e^{s_i}+\cdots)\\ 
=&\,\log\left(1+\sum_i p_i e^{-s_i}+\cdots\right)+\log\left(1+\sum_i(1-p_i)e^{s_i}+\cdots\right) 
\end{aligned}\tag{10}
$$

如果去掉高阶项，那么就得到：

$$
\log\left(1+\sum_i p_i e^{-s_i}\right)+\log\left(1+\sum_i(1-p_i)e^{s_i}\right)\label{eq:soft}\tag{11}
$$

并且作者通过推导证明了这就是式 $(7)$ 的软标签版本，当 $p_i\in\{0,1\}$ 时就退化为式 $(7)$ 的。

> 如果要将结果输出为 0～1 的概率值，正确做法应该是 $\sigma(2s_i)$ 而不是直觉中的 $\sigma(s_i)$。

作者同时放出了 Keras 版的实现：[multilabel_categorical_crossentropy](https://github.com/bojone/bert4keras/blob/5f5d493fe7be9ff2bd0e303e78ed945d386ed8fd/bert4keras/backend.py#L331)，注意该 loss 在实现时需要一些技巧，可以参见[实现技巧](https://kexue.fm/archives/9064#%E5%AE%9E%E7%8E%B0%E6%8A%80%E5%B7%A7)。

## 参考

[[1]](https://kexue.fm/archives/7359) 将“softmax+交叉熵”推广到多标签分类问题  
[[2]](https://kexue.fm/archives/9064) 多标签“Softmax+交叉熵”的软标签版本