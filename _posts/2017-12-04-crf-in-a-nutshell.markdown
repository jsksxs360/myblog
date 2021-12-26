---
layout: article
title: "果壳中的条件随机场"
author: 苏剑林
tags:
    - 机器学习
mathjax: true
---

> 转载自[《果壳中的条件随机场(CRF In A Nutshell)》](http://kexue.fm/archives/4695/)，作者：苏剑林。

本文希望用尽可能简短的语言把 CRF（条件随机场，Conditional Random Field）的原理讲清楚，这里 In A Nutshell 在英文中其实有“导论”、“科普”等意思（霍金写过一本《果壳中的宇宙》，这里东施效颦一下）。

网上介绍 CRF 的文章，不管中文英文的，基本上都是先说一些概率图的概念，然后引入特征的指数公式，然后就说这是 CRF。所谓“概率图”，只是一个形象理解的说法，然而如果原理上说不到点上，你说太多形象的比喻，反而让人糊里糊涂，以为你只是在装逼。（说到这里我又想怼一下了，求解神经网络，明明就是求一下梯度，然后迭代一下，这多好理解，偏偏还弄个装逼的名字叫“反向传播”，如果不说清楚它的本质是求导和迭代求解，一下子就说反向传播，有多少读者会懂？）

好了，废话说完了，来进入正题。

## 逐标签 Softmax

CRF 常见于序列标注相关的任务中。假如我们的模型输入为 $Q$，输出目标是一个序列 $a_1,a_2,\dots,a_n$，那么按照我们通常的建模逻辑，我们当然是希望目标序列的概率最大：

$$
P(a_1,a_2,\dots,a_n \mid Q)
$$

不管用传统方法还是用深度学习方法，直接对完整的序列建模是比较艰难的，因此我们通常会使用一些假设来简化它，比如直接使用朴素假设，就得到：

$$
P(a_1,a_2,\dots,a_n \mid Q)=P(a_1 \mid Q)P(a_2 \mid Q)\dots P(a_n \mid Q)
$$

注意这里的 $Q$ 不一定是原始输入，比如它可能是经过多层 LSTM 之后的隐藏输出 $q_1,q_2,\dots,q_n$，并且我们认为全局的关联意境由前面的模型捕捉完成了，因此在最后一步，我们可以认为特征之间互不相关，那么：

$$
\begin{aligned}P(a_1 \mid Q)=&P(a_1 \mid q_1,q_2,\dots,q_n)=P(a_1 \mid q_1)\\ 
P(a_2 \mid Q)=&P(a_2 \mid q_1,q_2,\dots,q_n)=P(a_2 \mid q_2)\\ 
&\quad\vdots\\ 
P(a_n \mid Q)=&P(a_n \mid q_1,q_2,\dots,q_n)=P(a_n \mid q_n)\\ 
\end{aligned}
$$

从而：

$$
P(a_1,a_2,\dots,a_n \mid Q)=P(a_1 \mid q_1)P(a_2 \mid q_2)\dots P(a_n \mid q_n)
$$

这就得到了我们最常用的方案：**直接逐标签输出最大概率的那个标签**。而前面的模型通常是多层的双向 LSTM。

## 条件随机场

逐标签 softmax 是一种简单有效的方法，但有时候会出现不合理的结果。比如我们用 sbme 来做 4 标签分词时，逐标签 softmax 无法排除出现 bbbb 这样的序列的可能性，但这个序列是违反了我们的解码规则（b 后面只能接 m 或 e）。因此，有人说逐标签 softmax 不需要动态规划，那是不对的，这种情况下，我们至少需要一个“非0即1”的转移矩阵，直接把不合理的转移概率设为 0（如 $P(b \mid b)=0$），然后通过动态规划保证得到合理的序列。

上述方案会出问题，归根结底就是我们在建模的时候，使用了输出完全独立的朴素假设（一元模型），但我们真实的输出序列又是上下文有关的，因此造成了优化目标与模型假设不吻合。能不能直接把上下文考虑进去呢？很简单，使用二元模型即可。

$$
\begin{aligned}P_Q(a_1,a_2,\dots,a_n)&=P_Q(a_1) P_Q(a_2 \mid a_1) P_Q(a_3 \mid a_1,a_2)\dots P_Q(a_n \mid a_1,\dots,a_{n-1})\\ 
&=P_Q(a_1) P_Q(a_2 \mid a_1) P_Q(a_3 \mid a_2)\dots P_Q(a_n \mid a_{n-1}) 
\end{aligned}
$$

这里为了使得表达式更好看一些，我把输入 $Q$ 放到了下标中。这**已经很接近 CRF**了！

继续观察上式，上面是一个转移概率的乘积，然而我们为什么非要定义每一项都是转移概率呢？于是，CRF 的做法非常一般，首先它定义了一个函数 $f(x,y;Q)$（它可能是一些简单的特征函数之和，但具体的形式其实不重要），然后直接令

$$
P_Q(a_1,a_2,\dots,a_n) = \frac{1}{Z}\exp\left(\sum_k f(a_{k-1},a_k;Q)\right)
$$

其中 $Z$ 是归一化因子。跟前一式相比，两者的区别在于，$P_Q(a_k \mid a_{k-1})$ 是有概率意义的（条件概率），而单项的 $e^{f(a_{k-1},a_k;Q)}/Z$ 是没有概率意义的，更有代表性。

这就是 CRF 的全部了。

一个更完整的参考链接：[https://zhuanlan.zhihu.com/p/28465510](https://zhuanlan.zhihu.com/p/28465510)

## 线性链 CRF

What？你在逗我？这就是CRF？这一个个 $P_Q(a_k \mid a_{k-1})$ 是什么鬼？还有 $f(x,y;Q)$ 呢？

这位读者，你可问倒我了，我也不知道是什么鬼呀。不信您到再看看网上的一些教程，它们给出的公式大概是这样的（直接抄自[这里](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)）：

$$
\begin{aligned}p(l \mid s) =& \frac{exp[score(l \mid s)]}{\sum_{l’} exp[score(l’ \mid s)]} \\ 
=& \frac{exp[\sum_{j = 1}^m \sum_{i = 1}^n \lambda_j f_j(s, i, l_i, l_{i-1})]}{\sum_{l’} exp[\sum_{j = 1}^m \sum_{i = 1}^n \lambda_j f_j(s, i, l’_i, l’_{i-1})]}\end{aligned}
$$

这里的 $f$ 都是未知的“特征函数”，需要根据具体问题具体设计，那还不是等价于说是未知的 $P_Q(a_k \mid a_{k-1})$。所以，我确实不知道是什么鬼呀。

好吧，就算你是对的，你好歹也教教我怎么用吧？

这里介绍一个经常用的版本——线性链 CRF，它就是 tensorflow 自带的版本。我们先写出：

$$
\begin{aligned}&P_Q(a_1,a_2,\dots,a_n)\\ 
=&P_Q(a_1) P_Q(a_2 \mid a_1) P_Q(a_3 \mid a_2)\dots P_Q(a_n \mid a_{n-1})\\ 
=&P_Q(a_1) \frac{P_Q(a_1, a_2)}{P_Q(a_1) P_Q(a_2)} P_Q(a_2) \frac{P_Q(a_2, a_3)}{P_Q(a_2) P_Q(a_3)}P_Q(a_3) \dots \frac{P_Q(a_{n-1}, a_n)}{P_Q(a_{n-1}) P_Q(a_n)} P_Q(a_n) 
\end{aligned}
$$

是不是感觉还挺好看的？根据 CRF 的一般思路，我们放弃每一项的概率意义，直接写出：

$$
\begin{aligned}&P_Q(a_1,a_2,\dots,a_n)\\ 
=&\frac{1}{Z} exp \Big[f(a_1;Q)+g(a_1, a_2;Q) + f(a_2;Q) +\dots + g(a_{n-1}, a_n;Q) + f(a_n;Q)\Big] 
\end{aligned}
$$

所谓线性链，就是直接认为函数 $g$ 实际上跟 $Q$ 没关系，任何情况都共用一个 $g(a_{k-1},a_k)$，这样它不过是个待确定的矩阵而已。剩下的则跟逐标签 softmax 的情形差不多了，认为 $f(a_k;Q)\equiv f(a_k;q_k)$。按照极大似然的思想，loss 应该取为：

$$
\begin{aligned} &-\log P_Q(a_1,a_2,\dots,a_n)\\ 
=& - \sum_{k=1}^n f(a_k;q_k) - \sum_{k=2}^n g(a_{k-1},a_k) + \log Z 
\end{aligned}
$$

如果前面模型用双向 LSTM 来得到特征 $q_k$，那么就得到了序列标注任务中最经典的 BiLSTM-CRF 了。

所以，现在也就不难理解 tensorflow 中自带的 [CRF 的函数](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf)了。相对于逐标签 softmax，CRF 不过是换了一个 loss 罢了，当然，还多了一个互信息矩阵，并且解码的时候需要用到 viterbi 算法。但这都不重要，因为 tensorflow 都帮我们写好了。

## 再设计？

线性链 CRF 可以说是一个化简的模版，我们是不是可能参照这个模版，设计一个改进的 CRF？比如，用模型生成一个跟 $Q$ 有关的互信息矩阵？也许是可能的。

剥开 nutshell，才能更好地品尝 nut。知其然，知其所以然，一切就没那么神秘了。

> 转载自[《果壳中的条件随机场(CRF In A Nutshell)》](http://kexue.fm/archives/4695/)，作者：苏剑林。