---
layout: article
title: "机器学习基础（五）：监督学习算法和随机梯度下降"
tags:
    - 机器学习
mathjax: true
sidebar:
  nav: machine-learning-note
---

> 本文内容摘取自 [《Deep Learning》](http://www.deeplearningbook.org/)，部分内容有修改。

## 监督学习算法

粗略地说，监督学习算法是给定一组输入 $\boldsymbol{x}$ 和输出 $\boldsymbol{y}$ 的训练集，学习如何关联输入和输出。

### 概率监督学习

大部分监督学习算法都是基于估计概率分布 $p(y \mid \boldsymbol{x})$ 的，我们可以使用最大似然估计找到对于有参分布族 $p(y \mid \boldsymbol{x}; \boldsymbol{\theta})$ 最好的参数向量 $\boldsymbol{\theta}$。

我们已经看到，线性回归对应于分布族

$$
p(y\mid\boldsymbol{x};\boldsymbol{\theta}) = \mathcal{N}(y;\boldsymbol{\theta}^\top\boldsymbol{x},\boldsymbol{I})
$$

通过定义一族不同的概率分布，我们可以将线性回归扩展到分类情况中。如果我们有两个类，类 $0$ 和类 $1$，那么我们只需要指定这两类之一的概率，因为这两个值加起来必须等于 $1$。

我们用于线性回归的实数正态分布是用均值参数化的。二元变量上的的分布稍微复杂些，因为它的均值必须始终在 $0$ 和 $1$ 之间。解决这个问题的一种方法是使用 logistic sigmoid 函数将线性函数的输出压缩进区间 $(0, 1)$。该值可以解释为概率：

$$
p(y = 1\mid\boldsymbol{x};\boldsymbol{\theta}) = \sigma(\boldsymbol{\theta}^\top\boldsymbol{x})
$$

这个方法被称为**逻辑回归** (logistic regression)，虽然名字叫回归但是该模型用于分类。

线性回归能够通过求解正规方程以找到最佳权重，而逻辑回归的最佳权重没有闭解，我们必须最大化对数似然来搜索最优解。我们可以通过梯度下降算法最小化负对数似然来搜索。

**通过确定正确的输入和输出变量上的有参条件概率分布族，相同的策略基本上可以用于任何监督学习问题。**

### 支持向量机

**支持向量机** (support vector machine, SVM) 是监督学习中最有影响力的方法之一。类似于逻辑回归，这个模型也是基于线性函数 $\boldsymbol{w}^\top\boldsymbol{x} + b$。不同于逻辑回归的是，支持向量机不输出概率，只输出类别。当 $\boldsymbol{w}^\top\boldsymbol{x} + b$ 为正时，支持向量机预测属于正类，为负时则预测属于负类。

支持向量机的一个重要创新是**核技巧** (kernel trick)。核技巧观察到许多机器学习算法都可以写成样本间点积的形式。例如，支持向量机中的线性函数可以重写为

$$
\boldsymbol{w}^\top\boldsymbol{x}+b = b +\sum_{i=1}^m \alpha_i\boldsymbol{x}^\top\boldsymbol{x}^{(i)}
$$

其中，$\boldsymbol{x}^{(i)}$ 是训练样本，$\boldsymbol{\alpha}$ 是系数向量。学习算法重写为这种形式允许我们将 $\boldsymbol{x}$ 替换为特征函数 $\phi(\boldsymbol{x})$ 的输出，点积替换为被称为**核函数** (kernel function) 的函数 $k(\boldsymbol{x}, \boldsymbol{x}^{(i)}) = \phi(\boldsymbol{x}) · \phi(\boldsymbol{x}^{(i)})$。运算符 $\cdot$ 表示类似于 $\phi(\boldsymbol{x})^\top\phi(\boldsymbol{x}^{(i)})$ 的点积。

> 对于某些特征空间，我们可能不会书面地使用向量内积。在某些无限维空间中，我们需要使用其他类型的内积，如基于积分而非加和的内积。

使用核估计替换点积之后，我们可以使用如下函数进行预测

$$
f(\boldsymbol{x}) = b + \sum_i \alpha_i k(\boldsymbol{x},\boldsymbol{x}^{(i)})
$$

这个函数关于 $\boldsymbol{x}$ 是非线性的，关于 $\phi(\boldsymbol{x})$ 是线性的。$\boldsymbol{\alpha}$ 和 $f(\boldsymbol{x})$ 之间的关系也是线性的。核函数完全等价于用 $\phi(\boldsymbol{x})$ 预处理所有的输入，然后在新的转换空间学习线性模型。在某些情况下，$\phi(\boldsymbol{x})$ 甚至可以是无限维的，对于普通的显式方法而言，这将是无限的计算代价。在很多情况下，即使 $\phi(\boldsymbol{x})$ 是难算的，$k(\boldsymbol{x}, \boldsymbol{x}')$ 却会是一个关于 $\boldsymbol{x}$ 非线性的、易算的函数。

> 核技巧十分强大有两个原因：
>
> 一、它使我们能够使用保证有效收敛的凸优化技术来学习非线性模型(关于 $\boldsymbol{x}$ 的函数)。我们可以认为 $\phi$ 是固定的，仅优化 $\boldsymbol{\alpha}$，即优化算法可以将决策函数视为不同空间中的线性函数。
>
> 二、核函数 $k$ 的实现方法通常有比直接构建 $\phi(\boldsymbol{x})$ 再算点积高效很多。

最常用的核函数是**高斯核** (Gaussian kernel)，

$$
k(\boldsymbol{u},\boldsymbol{v}) = \mathcal{N}(\boldsymbol{u}-\boldsymbol{v};0,\sigma^2\boldsymbol{I})
$$

其中 $\mathcal{N} (\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ 是标准正态密度，这个核也被称为**径向基函数** (radial basis function, RBF) 核。我们可以认为高斯核在执行一种**模板匹配** (template matching)。训练标签 $y$ 相关的训练样本 $\boldsymbol{x}$ 变成了类别 $y$ 的模版。当测试点 $\boldsymbol{x}'$ 到 $\boldsymbol{x}$ 的欧几里得距离很小，对应的高斯核响应很大时，表明 $\boldsymbol{x}'$ 和模版 $\boldsymbol{x}$ 非常相似。该模型进而会赋予相对应的训练标签 $y$ 较大的权重。

许多其他的线性模型也可以通过核技巧来增强，这些算法被称为**核机器** (kernel machine) 或**核方法** (kernel method)。核机器的一个主要缺点是计算决策函数的成本关于训练样本的数目是线性的（因为第 $i$ 个样本贡献 $\alpha_i k(\boldsymbol{x}, \boldsymbol{x}^{(i)})$ 到决策函数）。支持向量机能够通过学习主要包含零的向量 $\boldsymbol{\alpha}$ 以缓和这个缺点，那么判断新样本的类别仅需要计算非零 $\alpha_i$ 对应的训练样本的核函数，这些训练样本被称为**支持向量** (support vector)。

## 随机梯度下降

机器学习中反复出现的一个问题是好的泛化需要大的训练集，但大的训练集的计算代价也更大。机器学习算法中的代价函数通常可以分解成每个样本的代价函数的总和。例如，训练数据的负条件对数似然可以写成

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{x},y \sim \hat{p}_{\text{data}}}L(\boldsymbol{x},y,\boldsymbol{\theta})=\frac{1}{m}\sum_{i=1}^m L(\boldsymbol{x}^{(i)},y^{(i)},\boldsymbol{\theta})
$$

其中 $L$ 是每个样本的损失 $L(\boldsymbol{x}, y, \boldsymbol{\theta}) = − \log p(y \mid \boldsymbol{x}; \boldsymbol{\theta})$。

对于这些相加的代价函数，梯度下降需要计算

$$
\nabla_\boldsymbol{\theta} J(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^m \nabla_\boldsymbol{\theta} L(\boldsymbol{x}^{(i)},y^{(i)},\boldsymbol{\theta})
$$

这个运算的计算代价是 $O(m)$。随着训练集规模增长为数十亿的样本，计算一步梯度也会消耗相当长的时间。

随机梯度下降的核心是，梯度是期望，而期望可使用小规模的样本近似估计。我们在算法的每一步都从训练集中均匀抽出一**小批量** (minibatch) 样本 $\mathbb{B}=\{\boldsymbol{x}^{(1)},…,\boldsymbol{x}^{(m')}\}$。小批量的数目 $m'$ 通常是一个相对较小的数，而且当训练集大小 $m$ 增长时，$m'$ 通常是固定的。

梯度的估计可以表示成

$$
\boldsymbol{g} = \frac{1}{m'} \nabla_\boldsymbol{\theta} \sum_{i=1}^{m'} L(\boldsymbol{x}^{(i)},y^{(i)},\boldsymbol{\theta})
$$

使用来自小批量 $\mathbb{B}$ 的样本。然后，随机梯度下降算法使用如下的梯度下降估计：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \epsilon \boldsymbol{g}
$$

其中，$\epsilon$ 是学习率。

> 梯度下降往往被认为很慢或不可靠。以前，将梯度下降应用到非凸优化问题被认为很鲁莽或没有原则。现在，我们知道梯度下降用于机器学习训练时效果不错。优化算法不一定能保证在合理的时间内达到一个局部最小值，但它通常能及时地找到代价函数一个很小的值，并且是有用的。

随机梯度下降在深度学习之外有很多重要的应用，它是在大规模数据上训练大型线性模型的主要方法。对于固定大小的模型，每一步随机梯度下降更新的计算量不取决于训练集的大小 $m$。达到收敛所需的更新次数通常会随训练集规模增大而增加。然而，当 $m$ 趋向于无穷大时，该模型最终会在随机梯度下降抽样完训练集上的所有样本之前收敛到可能的最优测试误差。继续增加 $m$ 不会延长达到模型可能的最优测试误差的时间。

> 在深度学习兴起之前，学习非线性模型的主要方法是结合核技巧的线性模型。 很多核学习算法需要构建一个 $m \times m$ 的矩阵 $G_{i,j} = k(\boldsymbol{x}^{(i)}, \boldsymbol{x}^{(j)})$。构建这个矩阵的计算量是 $O(m^2)$。当数据集是几十亿个样本时，这个计算量是不能接受的。

## 构建机器学习算法

几乎所有的深度学习算法都可以被描述为一个相当简单的配方：特定的数据集、代价函数、优化过程和模型。

例如，线性回归算法由以下部分组成：$\boldsymbol{X}$ 和 $\boldsymbol{y}$ 构成的数据集，代价函数

$$
J(\boldsymbol{w},b) = -\mathbb{E}_{\boldsymbol{x},y\sim\hat{p}_{\text{data}}} \log p_{\text{model}}(y \mid \boldsymbol{x})
$$

模型是 $p_{\text{model}}(y \mid \boldsymbol{x}) = \mathcal{N} (y; \boldsymbol{x}^\top\boldsymbol{w} + b, 1)$，在大多数情况下，优化算法可以定义为求解代价函数梯度为零的正规方程。

通常代价函数至少含有一项使学习过程进行统计估计的成分。最常见的代价函数是负对数似然，最小化代价函数会导致最大似然估计。代价函数也可能含有附加项，如正则化项。例如，我们可以将权重衰减加到线性回归的代价函数中

$$
J(\boldsymbol{w},b) = \lambda \Vert\boldsymbol{w}\Vert^2_2 - \mathbb{E}_{\boldsymbol{x},y\sim\hat{p}_{\text{data}}} \log p_\text{model}(y\mid\boldsymbol{x})
$$

该优化仍然有闭解。

如果我们将该模型变成非线性的，那么大多数代价函数不再能通过闭解优化。 这就要求我们选择一个迭代数值优化过程，如梯度下降等。

在某些情况下，由于计算原因，我们不能实际计算代价函数。在这种情况下，只要我们有近似其梯度的方法，那么我们仍然可以使用迭代数值优化近似最小化目标。

尽管有时候不显然，但大多数学习算法都用到了上述配方。

> 本文内容摘取自 [《Deep Learning》](http://www.deeplearningbook.org/)，部分内容有修改。