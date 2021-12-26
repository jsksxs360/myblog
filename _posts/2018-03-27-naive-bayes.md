---
layout: article
title: "朴素贝叶斯分类器：假设所有属性相互独立"
tags:
    - 机器学习
mathjax: true
---

## 贝叶斯决策论

### 贝叶斯判定准则

假设有 $N$ 种可能的类别标记，即 $\mathcal{Y} = \{c_1,c_2,…,c_N\}$，$\lambda_{ij}$ 是将一个真实标记为 $c_j$ 的样本误分类为 $c_i$ 所产生的损失。基于后验概率 $P(c_i\mid\boldsymbol{x})$ 可获得将样本 $\boldsymbol{x}$ 分类为 $c_i$ 所产生的期望损失 (expected loss)，即在样本 $\boldsymbol{x}$ 上的“条件风险” (conditional risk)

$$
R(c_i\mid\boldsymbol{x}) = \sum_{j=1}^N \lambda_{ij} P(c_j \mid \boldsymbol{x}) \tag{1.1}
$$

我们的任务是寻找一个判定准则 $h:\mathcal{X} \mapsto \mathcal{Y}$ 以最小化总体风险

$$
R(h) = \mathbb{E}_{\boldsymbol{x}} [R(h(\boldsymbol{x})\mid\boldsymbol{x})] \tag{1.2}
$$

显然，对每个样本 $\boldsymbol{x}$，若 $h$ 能最小化条件风险 $R(h(\boldsymbol{x}) \mid \boldsymbol{x})$，则总体风险 $R(h)$ 也将被最小化。这就产生了**贝叶斯判定准则** (Bayes decision rule)：为最小化总体风险，只需在每个样本上选择那个能使条件风险 $R(c \mid \boldsymbol{x})$ 最小的类别标记，即

$$
h^*(\boldsymbol{x}) = \mathop{\arg\min}_{c \in \mathcal{Y}} R(c\mid \boldsymbol{x}) \tag{1.3}
$$

此时 $h^*$ 称为贝叶斯最优分类器，与之对应的总体风险 $R(h^*)$ 称为贝叶斯风险。

若目标是最小化分类错误率，并且选择 $0/1$ 损失函数，则误判损失 $\lambda_{ij}$ 可写为

$$
\lambda_{ij} = 
\begin{cases}
0, &\text{if }i=j;\\
1,&\text{otherwise}
\end{cases}
$$

此时条件风险

$$
R(c\mid\boldsymbol{x}) = 1-P(c\mid\boldsymbol{x}) \tag{1.4}
$$

于是，最小化分类错误率的贝叶斯最优分类器为

$$
h^*(\boldsymbol{x}) = \mathop{\arg\max}_{c\in\mathcal{Y}} P(c\mid \boldsymbol{x}) \tag{1.5}
$$

即对每个样本 $\boldsymbol{x}$，选择能使后验概率 $P(c\mid\boldsymbol{x})$ 最大的类别标记。

### 生成式模型

不难看出，欲使用贝叶斯判定准则来最小化决策风险，首先要获得后验概率 $P(c\mid\boldsymbol{x})$，然而在现实任务中这通常难以直接获得。从这个角度来看，机器学习所要实现的是基于有限的训练样本集尽可能准确地估计后验概率 $P(c\mid\boldsymbol{x})$。大体上来说，主要有两种策略：

- 给定 $\boldsymbol{x}$，可通过直接建模 $P(c\mid\boldsymbol{x})$ 来预测 $c$，这样得到的是**判别式模型**；
- 也可以先对联合概率分布 $P(\boldsymbol{x},c)$ 建模，然后再由此获得 $P(c\mid\boldsymbol{x})$，这样得到的是**生成式模型**。

对生成式模型来说，必然考虑

$$
P(c\mid\boldsymbol{x}) = \frac{P(\boldsymbol{x},c)}{P(\boldsymbol{x})} \tag{1.6}
$$

基于贝叶斯定理，$P(c\mid\boldsymbol{x})$ 可写为

$$
P(c\mid\boldsymbol{x}) = \frac{P(c)P(\boldsymbol{x}\mid c)}{P(\boldsymbol{x})} \tag{1.7}
$$

其中，$P(c)$ 是类先验概率，$P(\boldsymbol{x}\mid c)$ 是样本 $\boldsymbol{x}$ 相对于类标记 $c$ 的类条件概率，$P(\boldsymbol{x})$ 是用于归一化的证据因子。对给定样本 $\boldsymbol{x}$，证据因子 $P(\boldsymbol{x})$ 与类标记无关，因此估计 $P(c\mid\boldsymbol{x})$ 的问题就转化为如何基于训练数据 $D$ 来估计先验 $P(c)$ 和条件概率 $P(\boldsymbol{x}\mid c)$。

- 类先验概率 $P(c)$ 表达了样本空间中各类样本所占的比例，根据大数定律，当训练集包含充足的独立同分布样本时，$P(c)$ 可通过各类样本出现的频率来进行估计。
- 类条件概率 $P(\boldsymbol{x}\mid c)$ 由于涉及关于 $\boldsymbol{x}$ 所有属性的联合概率，直接根据样本出现的频率来估计将会遇到严重的困难。例如，假设样本的 $d$ 个属性都是二值的，则样本空间将有 $2^d$ 种可能的取值，这个值往往大于训练样本数 $m$，因此很多样本取值在训练集中根本没有出现。

## 朴素贝叶斯法的参数估计

### 极大似然估计

估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计。具体地，记关于类别 $c$ 的类条件概率为 $P(\boldsymbol{x}\mid c)$，假设 $P(\boldsymbol{x}\mid c)$ 具有确定的形式并且被参数向量 $\boldsymbol{\theta}_c$ 唯一确定，则我们的任务就是利用训练集 $D$ 估计参数 $\boldsymbol{\theta}_c$。为明确起见，我们将 $P(\boldsymbol{x}\mid c)$ 记为 $P(\boldsymbol{x} \mid \boldsymbol{\theta}_c)$。

事实上，概率模型的训练过程就是参数估计过程。对于参数估计，统计学界的两个学派分别提供了不同的解决方案：

- 频率派认为参数虽然未知，但却是客观存在的固定值，因此，可通过优化似然函数等准则来确定参数值
- 贝叶斯派则认为参数是未观察到的随机变量，其本身也可有分布，因此，可假定参数服从一个先验分布，然后基于观察到的数据来计算参数的后验分布。

本节介绍源自频率主义学派的**极大似然估计** (MLE)，这是根据数据采样来估计概率分布参数的经典方法。

令 $D_c$ 表示训练集 $D$ 中第 $c$ 类样本组成的集合，假设这些样本是独立同分布的，则参数 $\boldsymbol{\theta}_c$ 对于数据集 $D_c$ 的似然是

$$
P(D_c\mid \boldsymbol{\theta}_c) = \prod_{\boldsymbol{x} \in D_c} P(\boldsymbol{x}\mid \boldsymbol{\theta}_c) \tag{2.1}
$$

对 $\boldsymbol{\theta}_c$ 进行极大似然估计，就是去寻找能最大化似然 $P(D_c\mid\boldsymbol{\theta}_c)$ 的参数值 $\hat{\boldsymbol{\theta}}_c$。直观上看，极大似然估计是试图在 $\boldsymbol{\theta}_c$ 所有可能的取值中，找到一个能使数据出现的“可能性”最大的值。

由于 $(2.1)$ 中的连乘操作易造成下溢，通常使用对数似然 (log-likelihood)

$$
\begin{align}
LL(\boldsymbol{\theta}_c) &= \log P(D_c\mid\boldsymbol{\theta}_c)\\
&=\sum_{\boldsymbol{x}\in D_c} \log P(\boldsymbol{x}\mid\boldsymbol{\theta}_c)
\end{align} \tag{2.2}
$$

此时参数 $\boldsymbol{\theta}_c$ 的极大估计 $\hat{\boldsymbol{\theta}}_c$ 为

$$
\hat{\boldsymbol{\theta}}_c = \mathop{\arg\max}_{\boldsymbol{\theta}_c} LL(\boldsymbol{\theta}_c) \tag{2.3}
$$

> 极大似然估计虽然使类条件概率估计变得相对简单，但估计结果的准确性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布。

### 朴素贝叶斯分类器 

基于贝叶斯公式 $1.7$ 来估计后验概率 $P(c\mid\boldsymbol{x})$ 的主要困难在于：类条件概率 $P(\boldsymbol{x}\mid c)$ 是所有属性上的联合概率，难以从有限的训练样本直接估计得到。为避开这个障碍，朴素贝叶斯分类器采用了“属性条件独立性假设”：对已知类别，假设所有属性相互独立。

基于属性条件独立性假设，$(1.7)$ 可以重写为

$$
P(c\mid\boldsymbol{x}) = \frac{P(c)P(\boldsymbol{x}\mid c)}{P(\boldsymbol{x})} = \frac{P(c)}{P(\boldsymbol{x})}\prod_{i=1}^d P(x_i\mid c) \tag{2.4}
$$

其中 $d$ 为属性数目，$x_i$ 为 $\boldsymbol{x}$ 在第 $i$ 个属性上的取值。由于对所有类别来说 $P(\boldsymbol{x})$ 相同，因此基于 $(1.5)$ 的贝叶斯判定准则有

$$
h_{nb}(\boldsymbol{x}) = \mathop{\arg\max}_{c \in \mathcal{Y}} P(c)\prod_{i=1}^d P(x_i\mid c) \tag{2.5}
$$

这就是朴素贝叶斯分类器的表达式。

显然，朴素贝叶斯分类器的训练过程就是基于训练集 $D$ 来估计类先验概率 $P(c)$，并为每个属性估计条件概率 $P(x_i\mid c)$。

令 $D_c$ 表示训练集 $D$ 中第 $c$ 类样本组成的集合，若有充足的独立同分布样本，则可以容易地估计出类先验概率

$$
P(c) = \frac{\vert D_c\vert}{\vert D\vert} \tag{2.6}
$$

对离散属性而言，令 $D_{c,x_i}$ 表示 $D_c$ 中在第 $i$ 个属性上取值为 $x_i$ 的样本组成的集合，则条件概率 $P(x_i\mid c)$ 可估计为

$$
P(x_i\mid c) = \frac{\vert D_{c,x_i}\vert}{\vert D_c \vert} \tag{2.7}
$$

对连续属性可考虑概率密度函数，假定 $p(x_i\mid c)\sim\mathcal{N}(\mu_{c,i},\sigma_{c,i}^2)$，其中 $\mu_{c,i}$ 和 $\sigma_{c,i}^2$ 分别是第 $c$ 类样本在第 $i$ 个属性上取值的均值和方差，则有

$$
p(x_i\mid c) = \frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp\Bigg(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\Bigg) \tag{2.8}
$$

> 一个简单的例子
>
> 尝试由下表的训练数据学习一个朴素贝叶斯分类器并确定 $\boldsymbol{x} = (2,S)^\top$ 的类标记。表中 $X_1 \in\{1,2,3\}$ 和 $X_2\in\{S,M,L\}$ 为特征，$Y\in\{1,-1\}$ 为标记。
> 
> $$
> \begin{array}{c|c} 
> \hline
> & 1&2&3&4&5&6&7&8&9&10&11&12&13&14&15\\
> \hline
> X_1 & 1&1&1&1&1&2&2&2&2&2&3&3&3&3&3\\
> \hline
> X_2&S&M&M&S&S&S&M&M&L&L&L&M&M&L&L\\
> \hline
> Y&-1&-1&1&1&-1&-1&-1&1&1&1&1&1&1&1&-1\\
> \hline
> \end{array}
> $$
> 
> 对于给定的 $\boldsymbol{x} = (2,S)^\top$ 计算：
>
> $$
> \begin{align}
> &P(Y=1)P(X_1=2\mid Y=1)P(X_2=S\mid Y=1) = \frac{9}{15}\cdot\frac{3}{9}\cdot\frac{1}{9}=\frac{1}{45}\\
> &P(Y=-1)P(X_1=2\mid Y=-1)P(X_2 = S \mid Y=-1) = \frac{6}{15}\cdot\frac{2}{6}\cdot\frac{3}{6}=\frac{1}{15}
> \end{align}
> $$
> 
> 因为 $P(Y=-1)P(X_1=2\mid Y=-1)P(X_2=S\mid Y=-1)$ 最大，所以 $y=-1$。

### 平滑

需注意，若某个属性值在训练集中没有与某个类同时出现过，则直接基于 $(2.7)$ 进行概率估计就会得到条件概率为零，再根据 $(2.5)$ 进行判别，就会得到后验概率值为零，使分类产生偏差。

为了避免其他属性携带的信息被训练集中未出现的属性值“抹去”，在估计概率值时通常要进行**平滑** (smoothing)，常用**拉普拉斯修正** (Laplacian correction)。具体来说，令 $N$ 表示训练集 $D$ 中可能的类别数，$N_i$ 表示第 $i$ 个属性可能的取值数，则式 $(2.6)$ 和 $(2.7)$ 分别修正为

$$
\hat{P}(c) = \frac{\vert D_c\vert + 1}{\vert D\vert + N} \tag{2.9}
$$

$$
\hat{P}(x_i\mid c) = \frac{\vert D_{c,x_i}\vert + 1}{\vert D_c \vert + N_i} \tag{2.10}
$$

显然，拉普拉斯修正避免了因训练集样本不充分而导致概率估值为零的问题，并且在训练集变大时，修正过程所引入的先验的影响也会逐渐变得可忽略，使得估值渐趋向于实际概率值。

> 拉普拉斯修正实质上假设了属性值与类别均匀分布，这是在朴素贝叶斯学习过程中额外引入的关于数据的先验。

在现实任务中朴素贝叶斯分类器有很多种使用方式：

- 若任务对预测速度要求较高，则对给定训练集，可将朴素贝叶斯分类器涉及的所有概率估值事先计算好存储起来，预测时只需“查表”即可进行判别；
- 若任务数据更替频繁，则可采用“懒惰学习”方式，不进行任何训练，待收到预测请求时再根据当前数据集进行概率估值；
- 若数据集不断增加，则可在现有估值基础上，仅对新增样本的属性值所涉及的概率估值进行计数修正即可实现增量学习。

## 参考

李航《统计学习方法》  
周志华《机器学习》