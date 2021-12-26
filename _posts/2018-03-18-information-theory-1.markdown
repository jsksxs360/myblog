---
layout: article
title: "概率与信息论基础（上）：基础概念"
tags:
    - 机器学习
mathjax: true
sidebar:
  nav: machine-learning-note
---

概率论中最基本的概念是随机变量，**随机变量** (random variable) 就是可以随机地取不同值的变量。 一个随机变量只是对可能的状态的描述，它必须伴随着一个概率分布来指定每个状态的可能性。随机变量可以是离散的或者连续的。离散随机变量拥有有限或者可数无限多的状态（这些状态不一定是整数，也可能只是一些被命名的没有数值的状态）。连续随机变量伴随着实数值。

> 我们通常用无格式字体来表示随机变量，用手写体来表示随机变量的取值。例如，$x_1$ 和 $x_2$ 都是随机变量 $\mathbb{x}$ 可能的取值。对于向量值变量，我们会将随机变量写成 $\textbf{x}$，它的一个可能取值为 $\boldsymbol{x}$。

## 概率分布

**概率分布** (probability distribution) 用来描述随机变量或一簇随机变量在每一个可能取到的状态的可能性大小。我们描述概率分布的方式取决于随机变量是离散的还是连续的。

### 离散型变量和概率质量函数

离散型变量的概率分布可以用**概率质量函数** (probability mass function, PMF) 来描述，通常用大写字母 $P$ 来表示。概率质量函数将随机变量能够取得的每个状态映射到对应的概率。$\mathbb{x} = x$ 的概率用 $P(x)$ 来表示，概率为 $1$ 表示 $\mathbb{x} = x$ 是确定的，概率为 $0$ 表示 $\mathbb{x} = x$ 是不可能发生的。

> 有时为了避免混淆，我们会明确写出随机变量的名称 $P (\mathbb{x} = x)$。有时我们会先定义一个随机变量，然后用 $\sim$ 符号来说明它遵循的分布 $\mathbb{x} \sim P(\mathbb{x})$。

概率质量函数可以同时作用于多个随机变量，这种多个变量的概率分布被称为**联合概率分布** (joint probability distribution)。$P(\mathbb{x} = x, \mathbb{y} = y)$ 表示 $\mathbb{x} = x$ 和 $\mathbb{y} = y$ 同时发生的概率，可以简写为 $P (x, y)$。

如果一个函数 $P$ 是随机变量 $\mathbb{x}$ 的 PMF，必须满足下面这几个条件：

- $P$ 的定义域必须是 $\mathbb{x}$ 所有可能状态的集合。
- $\forall x \in \mathbb{x}, 0 \le P (x) \le 1$. 不可能发生的事件概率为 $0$，不存在比这概率更低的状态；一定发生的事件概率为 $1$，不存在比这概率更高的状态。
- $\sum_{x\in\mathbb{x}} P(x) = 1.$ 我们把这条性质称之为**归一化的** (normalized)。

### 连续型变量和概率密度函数

对于连续型随机变量，我们用**概率密度函数** (probability density function, PDF) 而不是概率质量函数来描述它的概率分布。如果一个函数 $p$ 是概率密度函数，必须满足下面这几个条件：

- $p$ 的定义域必须是 $\mathbb{x}$ 所有可能状态的集合。
- $\forall x \in \mathbb{x}, p(x) \ge 0.$ 注意，我们并不要求 $p(x) \le 1$。
- $\int p(x) dx = 1.$

概率密度函数 $p(x)$ 并没有直接对特定的状态给出概率，相对的，它给出了落在面积为 $\delta x$ 的无限小的区域内的概率为 $p(x)\delta x$。我们可以对概率密度函数求积分来获得点集的真实概率质量。特别地，$x$ 落在集合 $\mathbb{S}$ 中的概率可以通过 $p(x)$ 对这个集合求积分来得到。

> 在单变量的例子中，$x$ 落在区间 $[a, b]$ 的概率是 $\int_{[a,b]}p(x)dx$。

## 边缘概率

有时候我们知道了一组变量的联合概率分布，但想要了解其中一个子集的概率分布。这种定义在子集上的概率分布被称为**边缘概率分布** (marginal probability distribution)。

例如，假设有离散型随机变量 $\mathbb{x}$ 和 $\mathbb{y}$，并且我们知道 $P(\mathbb{x},\mathbb{y})$。我们可以依据下面的**求和法则** (sum rule) 来计算 $P (\mathbb{x})$：

$$
\forall x \in \mathbb{x}, P(\mathbb{x} = x) = \sum_y P(\mathbb{x} = x, \mathbb{y} = y) \tag{2.1}
$$

对于连续型变量，我们需要用积分替代求和：

$$
P(x) = \int p(x,y) dy \tag{2.2}
$$

## 条件概率

很多情况下，我们感兴趣的是某个事件在给定其他事件发生时出现的概率，这种概率叫做条件概率。我们将给定 $\mathbb{x} = x$，$\mathbb{y} = y$ 发生的条件概率记为 $P (\mathbb{y} = y \mid \mathbb{x} = x)$，这个条件概率可以通过下面的公式计算：

$$
P(\mathbb{y} = y\mid \mathbb{x} = x) = \frac{P(\mathbb{y} = y, \mathbb{x} = x)}{P(\mathbb{x} = x)} \tag{3.1}
$$

> 注意，条件概率只在 $P (\mathbb{x} = x) \gt 0$ 时有定义。

任何多维随机变量的联合概率分布，都可以分解成只有一个变量的条件概率相乘的形式：

$$
P(\mathbb{x}^{(1)},...,\mathbb{x}^{(n)}) = P(\mathbb{x}^{(1)})\prod_{i=2}^n P(\mathbb{x}^{(i)}\mid\mathbb{x}^{(1)},...,\mathbb{x}^{(i-1)}) \tag{3.2}
$$

这个规则被称为概率的**链式法则** (chain rule)。 它可以直接从式 $(3.5)$ 条件概率的定义中得到。例如，使用两次定义可以得到

$$
\begin{align}
P(\mathbb{a},\mathbb{b},\mathbb{c}) &= P(\mathbb{a} \mid \mathbb{b},\mathbb{c}) P(\mathbb{b},\mathbb{c})\\
P(\mathbb{b},\mathbb{c}) &= P(\mathbb{b} \mid \mathbb{c}) P(\mathbb{c}) \\
P(\mathbb{a},\mathbb{b},\mathbb{c}) &= P(\mathbb{a} \mid \mathbb{b},\mathbb{c}) P(\mathbb{b}\mid \mathbb{c}) P(\mathbb{c})
\end{align}
$$

## 独立性和条件独立性

两个随机变量 $\mathbb{x}$ 和 $\mathbb{y}$，如果它们的概率分布可以表示成两个因子的乘积形式，并且一个因子只包含 $\mathbb{x}$ 另一个因子只包含 $\mathbb{y}$，我们就称这两个随机变量是**相互独立的** (independent)：

$$
\forall x \in \mathbb{x},y \in \mathbb{y}, p(\mathbb{x} = x, \mathbb{y} = y) = p(\mathbb{x} = x)p(\mathbb{y}= y) \tag{4.1}
$$

如果关于 $\mathbb{x}$ 和 $\mathbb{y}$ 的条件概率分布对于 $\mathbb{z}$ 的每一个值都可以写成乘积的形式，那么这两个随机变量 $\mathbb{x}$ 和 $\mathbb{y}$ 在给定随机变量 $\mathbb{z}$ 时是**条件独立的** (conditionally independent)：

$$
\forall x \in \mathbb{x},y\in\mathbb{y},z\in\mathbb{z}, p(\mathbb{x} = x, \mathbb{y} = y \mid \mathbb{z} = z) = p(\mathbb{x} = x \mid \mathbb{z} = z)p(\mathbb{y} = y \mid \mathbb{z} = z) \tag{4.2}
$$

## 期望、方差和协方差

### 期望

函数 $f(x)$ 关于某分布 $P(\mathbb{x})$ 的**期望** (expectation) 是指，当 $x$ 由 $P$ 产生，$f$ 作用于 $x$ 时，$f(x)$ 的平均值。对于离散型随机变量，这可以通过求和得到：

$$
\mathbb{E}_{\mathbb{x}\sim P}[f(x)] = \sum_x P(x)f(x) \tag{5.1}
$$

对于连续型随机变量可以通过求积分得到：

$$
\mathbb{E}_{\mathbb{x}\sim p}[f(x)] = \int p(x)f(x)dx \tag{5.2}
$$

> 当概率分布指明时，我们可以只写出期望作用的随机变量的名称，例如 $\mathbb{E}_\mathbb{x}[f(x)]$。如果期望作用的随机变量也很明确，我们可以完全不写脚标， 就像 $\mathbb{E}[f(x)]$，默认地，我们假设 $\mathbb{E}[\cdot]$ 表示对方括号内的所有随机变量的值求平均。 

期望是线性的，例如，

$$
\mathbb{E}_{\mathbb{x}}[\alpha f(x) + \beta g(x)] = \alpha \mathbb{E}_{\mathbb{x}} [f(x)] + \beta \mathbb{E}_{\mathbb{x}} [g(x)] \tag{5.3}
$$

其中 $\alpha$ 和 $\beta$ 不依赖于 $x$。

### 方差

**方差** (variance) 衡量的是当我们对 $x$ 依据它的概率分布进行采样时，随机变量 $\mathbb{x}$ 的函数值会呈现多大的差异：

$$
\text{Var}(f(x)) = \mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2] \tag{5.4}
$$

当方差很小时，$f(x)$ 的值形成的簇比较接近它们的期望值。方差的平方根被称为**标准差** (standard deviation)。

### 协方差

**协方差** (covariance) 在某种意义上给出了两个变量线性相关性的强度以及这些变量的尺度：

$$
\text{Cov}(f(x),g(x)) = \mathbb{E}[(f(x)-\mathbb{E[f(x)]})(g(y) - \mathbb{E}[g(y)])] \tag{5.5}
$$

协方差的绝对值如果很大则意味着变量值变化很大并且它们同时距离各自的均值很远。如果协方差是正的，那么两个变量都倾向于同时取得相对较大的值。如果协方差是负的，那么其中一个变量倾向于取得相对较大的值的同时，另一个变量倾向于取得相对较小的值，反之亦然。

> 其他的衡量指标如**相关系数** (correlation) 将每个变量的贡献归一化，为了只衡量变量的相关性而不受各个变量尺度大小的影响。两个变量如果相互独立那么它们的协方差为零，如果两个变量的协方差不为零那么它们一定是相关的。

随机向量 $\boldsymbol{x} \in \mathbb{R}^n$ 的**协方差矩阵** (covariance matrix) 是一个 $n \times n$ 的矩阵，并且满足

$$
\text{Cov}(\textbf{x})_{i,j} = \text{Cov}(\mathbb{x}_i,\mathbb{x}_j)
$$

协方差矩阵的对角元是方差：

$$
\text{Cov}(\mathbb{x}_i, \mathbb{x}_i) = \text{Var}(\mathbb{x}_i)
$$

> 本文为 [《Deep Learning》](http://www.deeplearningbook.org/)的学习笔记，感谢 [Exacity 小组](https://github.com/exacity)的翻译。