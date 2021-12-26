---
layout: article
title: "机器学习基础（四）：最大似然估计和贝叶斯统计"
tags:
    - 机器学习
mathjax: true
sidebar:
  nav: machine-learning-note
---

> 本文内容摘取自 [《Deep Learning》](http://www.deeplearningbook.org/)，部分内容有修改。

在《机器学习基础（三）》[估计](http://xiaosheng.me/2018/02/21/article126/#2-%E4%BC%B0%E8%AE%A1)一节中我们已经看过常用估计的定义，并分析了它们的性质，但是这些估计是从哪里来的呢？我们希望有些准则可以让我们从不同模型中得到特定函数作为好的估计，而不是猜测某些函数可能是好的估计，然后分析其偏差和方差。

最常用的准则是最大似然估计。

## 最大似然估计 

考虑一组含有 $m$ 个样本的数据集 $\mathbb{X} = \{\boldsymbol{x}^{(1)}, . . . , \boldsymbol{x}^{(m)}\}$，独立地由未知的真实数据分布 $p_{\text{data}}(\boldsymbol{x})$ 生成。令 $p_{\text{model}}(\boldsymbol{x}; \boldsymbol{\theta})$ 是一族由 $\boldsymbol{\theta}$ 确定在相同空间上的概率分布，将任意输入 $\boldsymbol{x}$ 映射到实数来估计真实概率 $p_{\text{data}}(\boldsymbol{x})$。

对 $\boldsymbol{\theta}$ 的最大似然估计被定义为：

$$
\begin{align}
\boldsymbol{\theta}_{\text{ML}} &= \mathop{\arg\max}_\boldsymbol{\theta} p_{\text{model}}(\mathbb{X}; \boldsymbol{\theta})\\
&=\mathop{\arg\max}_\boldsymbol{\theta} \prod_{i=1}^m p_{\text{model}}(\boldsymbol{x}^{(i)}; \boldsymbol{\theta})
\end{align}
$$

多个概率的乘积会因很多原因不便于计算（例如可能出现数值下溢）。我们观察到似然对数不会改变其 $\arg \max$ 但是将乘积转化成了便于计算的求和形式：

$$
\boldsymbol{\theta}_{\text{ML}} = \mathop{\arg \max}_\boldsymbol{\theta} \sum_{i=1}^m \log p_{\text{model}}(\boldsymbol{x}^{(i)};\boldsymbol{\theta}) \tag{1.1}
$$

因为当我们重新缩放代价函数时 $\arg \max$ 不会改变，我们可以除以 $m$ 得到和训练数据经验分布 $\hat{p}_{\text{data}}$ 相关的期望作为准则：

$$
\boldsymbol{\theta}_{\text{ML}} = \mathop{\arg\max}_\boldsymbol{\theta} \mathbb{E}_{\boldsymbol{x}\sim \hat{p}_{\text{data}}} \log p_{\text{model}}(\boldsymbol{x};\boldsymbol{\theta}) \tag{1.2}
$$

一种解释最大似然估计的观点是将它看作最小化训练集上的经验分布 $\hat{p}_{\text{data}}$ 和模型分布之间的差异，两者之间的差异程度可以通过 KL 散度度量。KL 散度被定义为：

$$
D_{\text{KL}}(\hat{p}_{\text{data}} \parallel p_{\text{model}}) = \mathbb{E}_{\boldsymbol{x}\sim\hat{p}_{\text{data}}}[\log\hat{p}_{\text{data}}(\boldsymbol{x}) - \log p_{\text{model}}(\boldsymbol{x})]
$$

左边一项仅涉及到数据生成过程，和模型无关。这意味着当我们训练模型最小化 KL 散度时，我们只需要最小化

$$
-\mathbb{E}_{\boldsymbol{x}\sim\hat{p}_{\text{data}}}[\log p_{\text{model}}(\boldsymbol{x})] \tag{1.3}
$$

当然，这和式 $(1.2)$ 中最大化是相同的。

最小化 KL 散度其实就是在最小化分布之间的交叉熵。**任何一个由负对数似然组成的损失都是定义在训练集上的经验分布和定义在模型上的概率分布之间的交叉熵。**例如，均方误差是经验分布和高斯模型之间的交叉熵。

> 许多作者使用术语 “交叉熵’’ 特定表示伯努利或 softmax 分布的负对数似然，是用词不当的。

我们可以将最大似然看作是使模型分布尽可能地和经验分布 $\hat{p}\_{\text{data}}$ 相匹配的尝试。理想情况下，我们希望匹配真实的数据生成分布 $p_{\text{data}}$，但我们没法直接知道这个分布。

最优 $\boldsymbol{\theta}$ 在最大化似然或是最小化 KL 散度时是相同的，我们通常将两者都称为最小化代价函数。因此最大化似然变成了最小化负对数似然 (NLL)，或者等价的是最小化交叉熵。

### 条件对数似然和均方误差

最大似然估计很容易扩展到估计条件概率 $P (\boldsymbol{y} \mid \boldsymbol{x}; \boldsymbol{\theta})$，从而给定 $\boldsymbol{x}$ 预测 $\boldsymbol{y}$，这构成了大多数监督学习的基础。如果 $\boldsymbol{X}$ 表示所有的输入，$\boldsymbol{Y}$ 表示我们观测到的目标，那么条件最大似然估计是：

$$
\boldsymbol{\theta}_{\text{ML}} = \mathop{\arg\max}_\boldsymbol{\theta} P(\boldsymbol{Y}\mid \boldsymbol{X};\boldsymbol{\theta})
$$

如果假设样本是独立同分布的，那么这可以分解成

$$
\boldsymbol{\theta}_{\text{ML}} = \mathop{\arg\max}_\boldsymbol{\theta}\sum_{i=1}^m \log P(\boldsymbol{y}^{(i)}\mid\boldsymbol{x}^{(i)};\boldsymbol{\theta}) \tag{1.4}
$$

例如，我们在《机器学习基础（一）》中介绍的[线性回归](http://xiaosheng.me/2018/01/24/article124/#4-%E7%A4%BA%E4%BE%8B%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92)，可以被看作是最大似然过程。之前我们将线性回归作为学习从输入 $\boldsymbol{x}$ 映射到输出 $\hat{y}$ 的算法，从 $\boldsymbol{x}$ 到 $\hat{y}$ 的映射选自最小化均方误差。现在，我们以最大似然估计的角度重新审视线性回归。

我们现在希望模型能够得到条件概率 $p(y \mid \boldsymbol{x})$，而不只是得到一个单独的预测 $\hat{y}$。学习算法的目标是拟合分布 $p(y \mid \boldsymbol{x})$ 到和 $\boldsymbol{x}$ 相匹配的不同的 $y$。为了得到我们之前推导出的相同的线性回归算法，我们定义 $p(y \mid \boldsymbol{x}) = \mathcal{N} (y; \hat{y}(\boldsymbol{x}; \boldsymbol{w}), \sigma^2)$，函数 $\hat{y}(\boldsymbol{x}; \boldsymbol{w})$ 预测高斯的均值，方差是用户固定的某个常量 $\sigma^2$。由于假设样本是独立同分布的，条件对数似然(式 $(1.4)$ ) 如下

$$
\sum_{i=1}^m\log p(y^{(i)}\mid\boldsymbol{x}^{(i)};\boldsymbol{\theta})\\
=-m\log\sigma - \frac{m}{2}\log(2\pi) - \sum_{i=1}^m\frac{\Vert\hat{y}^{(i)}-y^{(i)}\Vert^2
}{2\sigma^2}
$$

其中 $\hat{y}^{(i)}$ 是线性回归在第 $i$ 个输入 $\boldsymbol{x}^{(i)}$ 上的输出，$m$ 是训练样本的数目。对比对数似然和均方误差，

$$
\text{MSE}_{\text{train}} = \frac{1}{m}\sum_{i=1}^m \Vert\hat{y}^{(i)}-y^{(i)}\Vert^2
$$

我们可以看出最大化关于 $\boldsymbol{w}$ 的对数似然和最小化均方误差会得到相同的参数估计 $\boldsymbol{w}$。这验证了 MSE 可以用于最大似然估计。

### 最大似然的性质

最大似然估计最吸引人的地方在于，它被证明当样本数目 $m \to \infty$ 时，就收敛率而言是最好的渐近估计。

在合适的条件下，最大似然估计具有一致性，意味着训练样本数目趋向于无穷大时，参数的最大似然估计会收敛到参数的真实值。这些条件是：

- 真实分布 $p_\text{data}$ 必须在模型族 $p_\text{model}(\cdot; \theta)$ 中。否则，没有估计可以还原 $p_\text{data}$。
- 真实分布 $p_\text{data}$ 必须刚好对应一个 $\boldsymbol{\theta}$ 值。否则，最大似然估计恢复出真实分布 $p_\text{data}$ 后，也不能决定数据生成过程使用哪个 $\boldsymbol{\theta}$。

除了最大似然估计，还有其他的归纳准则，其中许多共享一致估计的性质。然而，一致估计的**统计效率** (statistic efficiency) 可能区别很大。某些一致估计可能会在固定数目的样本上获得一个较低的泛化误差，或者等价地，可能只需要较少的样本就能达到一个固定程度的泛化误差。

因为一致性和统计效率，最大似然通常是机器学习中的首选估计。 当样本数目小到会发生过拟合时，正则化策略如权重衰减可用于获得训练数据有限时方差较小的最大似然有偏版本。

## 贝叶斯统计

前面我们已经讨论了**频率派统计** (frequentist statistics) 和基于估计单一值 $\boldsymbol{\theta}$ 的方法，然后基于该估计作所有的预测，另一种方法是在做预测时会考虑所有可能的 $\boldsymbol{\theta}$，后者属于**贝叶斯统计** (Bayesian statistics) 的范畴。

> 频率派的视角是真实参数 $\boldsymbol{\theta}$ 是未知的定值，点估计 $\hat{\boldsymbol{\theta}}$ 是考虑数据集上函数的随机变量，而贝叶斯统计的视角则完全不同。贝叶斯用概率反映知识状态的确定性程度。数据集能够被直接观测到，因此不是随机的。另一方面，真实参数 $\boldsymbol{\theta}$ 是未知或不确定的， 因此可以表示成随机变量。

在观察到数据前，我们将 $\boldsymbol{\theta}$ 的已知知识表示成**先验概率分布** (prior probability distribution)，$p(\boldsymbol{\theta})$。一般实践中会选择一个相当宽泛的（高熵的）先验分布，反映在观测到任何数据前参数 $\boldsymbol{\theta}$ 的高度不确定性。例如，我们可能会假设先验 $\boldsymbol{\theta}$ 在有限区间中均匀分布。

现在假设我们有一组数据样本 $\{x^{(1)},…,x^{(m)}\}$。通过贝叶斯规则结合数据似然 $p(x^{(1)}, . . . , x^{(m)} \mid \boldsymbol{\theta})$ 和先验，我们可以恢复数据对我们关于 $\boldsymbol{\theta}$ 信念的影响：

$$
p(\boldsymbol{\theta}\mid x^{(1)},...,x^{(m)}) = \frac{p(x^{(1)},...,x^{(m)}\mid \boldsymbol{\theta})p(\boldsymbol{\theta})}{p(x^{(1)},...,x^{(m)})} \tag{2.1}
$$

在贝叶斯估计常用的情景下，先验开始是相对均匀的分布或高熵的高斯分布，观测数据通常会使后验的熵下降，并集中在参数的几个可能性很高的值。

### 与最大似然的区别

相对于最大似然估计，贝叶斯估计有两个重要区别。第一，不像最大似然方法预测时使用 $\boldsymbol{\theta}$ 的点估计，贝叶斯方法使用 $\boldsymbol{\theta}$ 的全分布。例如，在观测到 $m$ 个样本后，下一个数据样本 $x^{(m+1)}$ 的预测分布如下：

$$
p(x^{(m+1)}\mid x^{(1)},...,x^{(m)}) = \int p(x^{(m+1)}\mid \boldsymbol{\theta})p(\boldsymbol{\theta}\mid x^{(1)},...,x^{(m)}) \text{ }d\boldsymbol{\theta}
$$

这里，每个具有正概率密度的 $\boldsymbol{\theta}$ 的值有助于下一个样本的预测，其中贡献由后验密度本身加权。在观测到数据集 $\{x^{(1)}, . . . , x^{(m)}\}$ 之后，如果我们仍然非常不确定 $\boldsymbol{\theta}$ 的值，那么这个不确定性会直接包含在我们所做的任何预测中。

频率派方法解决给定点估计 $\boldsymbol{\theta}$ 的不确定性的方法是评估方差，估计的方差评估了观测数据重新从观测数据中采样后，估计可能如何变化。对于如何处理估计不确定性的这个问题，贝叶斯派的答案是积分，这往往会防止过拟合。

贝叶斯方法和最大似然方法的第二个最大区别是由贝叶斯先验分布造成的。先验能够影响概率质量密度朝参数空间中偏好先验的区域偏移。实践中，先验通常表现为偏好更简单或更光滑的模型。对贝叶斯方法的批判认为先验是人为主观判断影响预测的来源。

> 当训练数据很有限时，贝叶斯方法通常泛化得更好，但是当训练样本数目很大时，通常会有很大的计算代价。

### 最大后验 (MAP) 估计

原则上，我们应该使用参数 $\boldsymbol{\theta}$ 的完整贝叶斯后验分布进行预测，但单点估计常常也是需要的。我们可以让先验影响点估计的选择来利用贝叶斯方法的优点，而不是简单地回到最大似然估计。

> 使用点估计的一个常见原因是，对于大多数有意义的模型而言，大多数涉及到贝叶斯后验的计算是非常棘手的，点估计提供了一个可行的近似解。

一种能够做到这一点的合理方式是选择**最大后验** (Maximum A Posteriori, MAP) 点估计。MAP 估计选择后验概率最大的点 (或在 $\boldsymbol{\theta}$ 是连续值情况下，概率密度最大的点)：

$$
\begin{align}
\boldsymbol{\theta}_{\text{MAP}} &= \mathop{\arg\max}_\boldsymbol{\theta} p(\boldsymbol{\theta}\mid\boldsymbol{x}) \\&=\mathop{\arg\max}_\boldsymbol{\theta} \frac{p(\boldsymbol{x\mid\boldsymbol{\theta}})p(\boldsymbol{\theta})}{p(\boldsymbol{x})}\\
&=\mathop{\arg\max}_\boldsymbol{\theta} p(\boldsymbol{x}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})
\\&= \mathop{\arg\max}_\boldsymbol{\theta}\log p(\boldsymbol{x}\mid\boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) \end{align}\tag{2.2}
$$

我们可以认出上式右边的 $\log p(\boldsymbol{x} \mid \boldsymbol{\theta})$ 对应着标准的对数似然项，$\log p(\boldsymbol{\theta})$ 对应着先验分布。

正如全贝叶斯推断，MAP 贝叶斯推断的优势是能够利用来自先验的信息，这些信息无法从训练数据中获得。该附加信息有助于减少最大后验点估计的方差（相比 于 ML 估计）。然而，这个优点的代价是增加了偏差。

> 本文内容摘取自 [《Deep Learning》](http://www.deeplearningbook.org/)，部分内容有修改。