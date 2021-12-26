---
layout: article
title: "机器学习基础（一）：学习算法"
tags:
    - 机器学习
mathjax: true
sidebar:
  nav: machine-learning-note
---

> 本文内容摘取自 [《Deep Learning》](http://www.deeplearningbook.org/)，部分内容有修改。

机器学习算法是一种能够从数据中学习的算法。Mitchell (1997) 提供了一个简洁的定义：“对于某类任务 $T$ 和性能度量 $P$，一个计算机程序被认为可以从经验 $E$ 中学习是指，通过经验 $E$ 改进后，它在任务 $T$ 上由性能度量 $P$ 衡量的性能有所提升。”

## 任务 $T$

通常机器学习任务定义为机器学习系统应该如何处理**样本** (example)。样本是指我们从某些希望机器学习系统处理的对象或事件中收集到的已经量化的**特征** (feature) 的集合。我们通常会将样本表示成一个向量 $\boldsymbol{x} \in \mathbb{R}^n$，其中向量的每一个元素 $x_i$ 是一个特征。

机器学习可以解决很多类型的任务。一些非常常见的机器学习任务列举如下:

- **分类：**在这类任务中，计算机程序需要指定某些输入属于 $k$ 类中的哪一类。为了完成这个任务，学习算法通常会返回一个函数 $f : \mathbb{R}^n \rightarrow \{1,…,k\}$。当 $y = f(\boldsymbol{x})$ 时，模型将向量 $\boldsymbol{x}$ 所代表的输入分类到 $y$ 所代表的类别。还有一些其他的分类问题，例如，$f$ 输出的是不同类别的概率分布。
- **回归：**在这类任务中，计算机程序需要对给定输入预测数值。为了解决这个任务，学习算法需要输出函数 $f : \mathbb{R}^n \rightarrow \mathbb{R}$。除了返回结果的形式不一样外，这类问题和分类问题是很像的。
- **结构化输出：**结构化输出任务的输出是向量或者其他包含多个值的数据结构，并且构成输出的这些不同元素间具有重要关系。这是一个很大的范畴，包括转录、机器翻译、语法分析在内的很多任务。这类任务被称为结构化输出任务是因为输出值之间内部紧密相关。例如，为图片添加标题的程序输出的单词必须组合成一个通顺的句子。

当然，还有很多其他类型的任务。这里我们列举的任务类型只是用来介绍机器学习可以做哪些任务，并非严格地定义机器学习任务分类。

## 性能度量 $P$

对于分类任务，我们通常度量模型的**准确率** (accuracy)，指该模型输出正确结果的样本比率，也可以通过**错误率** (error rate)得到相同的信息，指该模型输出错误结果的样本比率。我们通常把错误率称为 0−1 损失的期望。在一个特定的样本上，如果结果是对的，那么 0−1 损失是 0，否则是 1。但是对于密度估计这类任务而言，我们必须使用不同的性能度量，使模型对每个样本都输出一个连续数值的得分。最常用的方法是输出模型在一些样本上概率对数的平均值。

通常，我们会更加关注机器学习算法在未观测数据上的性能如何，因为这将决定其在实际应用中的性能。因此，我们使用**测试集** (test set) 数据来评估系统性能，将其与训练机器学习系统的训练集数据分开。

> 性能度量的选择或许看上去简单且客观，但是选择一个与系统理想表现对应的性能度量通常是很难的。在某些情况下，这是因为很难确定应该度量什么。还有一些情况，我们知道应该度量哪些数值，但是度量它们不太现实。

## 经验 $E$

根据学习过程中的不同经验，机器学习算法可以大致分类为**无监督** (unsupervised) 算法和**监督** (supervised) 算法。

- **无监督学习算法** (unsupervised learning algorithm)

  训练含有很多特征的数据集，然后学习出这个数据集上有用的结构性质。在深度学习中，我们通常要学习生成数据集的整个概率分布，显式地，比如密度估计，或是隐式地，比如合成或去噪。 还有一些其他类型的无监督学习任务，例如聚类，将数据集分成相似样本的集合。

- **监督学习算法** (supervised learning algorithm)

  训练含有很多特征的数据集，不过数据集中的样本都有一个**标签** (label) 或**目标** (target)。例如，Iris 数据集注明了每个鸢尾花卉样本属于什么品种，监督学习算法通过研究 Iris 数据集，学习如何根据测量结果将样本划分为三个不同品种。

大致说来，无监督学习涉及到观察随机向量 $\boldsymbol{x}$ 的好几个样本，试图显式或隐式地学习出概率分布 $p(\boldsymbol{x})$，或者是该分布一些有意思的性质；而监督学习包含观察随机向量 $\boldsymbol{x}$ 及其相关联的值或向量 $\boldsymbol{y}$，然后从 $\boldsymbol{x}$ 预测 $\boldsymbol{y}$，通常是估计 $p(\boldsymbol{y} \mid \boldsymbol{x})$。

无监督学习和监督学习不是严格定义的术语，它们之间界线通常是模糊的。例如，概率的链式法则表明对于向量 $\boldsymbol{x} \in \mathbb{R}^n$，联合分布可以分解成

$$
p(\boldsymbol{x}) = \prod_{i=1}^n p(x_i \mid x_1,...,x_{i-1}) \tag{1}
$$

该分解意味着我们可以将其拆分成 $n$ 个监督学习问题，来解决表面上的无监督学习 $p(\boldsymbol{x})$。另外，我们求解监督学习问题 $p(y \mid \boldsymbol{x})$ 时，也可以使用传统的无监督学习策略学习联合分布 $p(\boldsymbol{x}, y)$，然后推断

$$
p(y \mid \boldsymbol{x}) = \frac{p(\boldsymbol{x}, y)}{\sum_{y'} p(\boldsymbol{x},y')} \tag{2}
$$

传统地，人们将回归、分类或者结构化输出问题称为监督学习，支持其他任务的密度估计通常被称为无监督学习。

> 学习范式的其他变种也是有可能的。例如，半监督学习中，一些样本有监督目标，但其他样本没有。在多实例学习中，样本的整个集合被标记为含有或者不含有该类的样本，但是集合中单独的样本是没有标记的。有些机器学习算法并不是训练于一个固定的数据集上。例如，强化学习 (reinforcement learning) 算法会和环境进行交互，所以学习系统和它的训练过程会有反馈回路。

大部分机器学习算法简单地训练于一个数据集上，数据集是样本的集合，而样本是特征的集合。表示数据集的常用方法是**设计矩阵** (design matrix)，每一行包含一个不同的样本，每一列对应不同的特征。例如，Iris 数据集包含 150 个样本，每个样本有 4 个特征，我们可以将该数据集表示为设计矩阵 $\boldsymbol{X} \in \mathbb{R}^{150\times 4}$。有时样本具有不同数量的特征。例如，你有不同大小的照片集合，那么不同的照片将会包含不同数量的像素。这时我们会将数据集表示成 $m$ 个元素的结合：$\{\boldsymbol{x}^{(1)}, \boldsymbol{x}^{(2)}, . . . , \boldsymbol{x}^{(m)}\}$。这种表示方式意味着样本向量 $\boldsymbol{x}^{(i)}$ 和 $\boldsymbol{x}^{(j)}$ 可以有不同的大小。

在监督学习中，样本包含一个标签或目标和一组特征。通常在处理包含观测特征的设计矩阵 $\boldsymbol{X}$ 的数据集时，我们也会提供一个标签向量 $\boldsymbol{y}$，其中 $y_i$ 表示样本 $i$ 的标签。当然，有时标签可能不止一个数。例如，如果我们想要训练语音模型转录整个句子，那么每个句子样本的标签是一个单词序列。

## 示例：线性回归

线性回归的目标是建立一个系统，将向量 $\boldsymbol{x} \in \mathbb{R}^n$ 作为输入，预测标量 $y \in \mathbb{R}$ 作为输出。令 $\hat{y}$ 表示模型预测 $y$ 的值。我们定义输出为

$$
\hat{y} = \boldsymbol{w}^\top\boldsymbol{x} \tag{3}
$$

其中 $\boldsymbol{w} \in \mathbb{R}^n$ 是**参数** (parameter) 向量。

> 我们可以将 $\boldsymbol{w}$ 看作是一组决定每个特征如何影响预测的**权重** (weight)。如果特征 $x_i$ 对应的权重 $w_i$ 是正的，那么特征的值增加，我们的预测值 $\hat{y}$ 也会增加；如果特征 $x_i$ 对应的权重 $w_i$ 是负的，那么特征的值减少，我们的预测值 $\hat{y}$ 也会减少。如果特征权重的大小很大，那么它对预测有很大的影响；如果特征权重的大小是零，那么它对预测没有影响。

因此，我们可以定义任务 $T$：通过输出 $\hat{y} = \boldsymbol{w}^\top \boldsymbol{x}$ 从 $\boldsymbol{x}$ 预测 $y$。接下来我们定义性能度量 $P$ ，一种常见的方法是计算模型在测试集上的**均方误差** (mean squared error)。假设我们有 $m$ 个输入样本组成的设计矩阵，$\hat{\boldsymbol{y}}^{(\text{test})}$ 表示模型在测试集上的预测值，那么均方误差表示为：

$$
\text{MSE}_{\text{test}} = \frac{1}{m}\sum_i (\hat{\boldsymbol{y}}^{(\text{test})}-\boldsymbol{y}^{(\text{test})})^2_i \tag{4}
$$

直观上，当 $\hat{\boldsymbol{y}}^{(\text{test})} = \boldsymbol{y}^{(\text{test})}$ 时，我们会发现误差降为 $0$。我们也可以看到

$$
\text{MSE}_{\text{test}} = \frac{1}{m}\Big\Arrowvert\hat{\boldsymbol{y}}^{(\text{test})}-\boldsymbol{y}^{(\text{test})}\Big\Arrowvert^2_2 \tag{5}
$$

所以当预测值和目标值之间的欧几里得距离增加时，误差也会增加。

为了构建一个机器学习算法，我们需要设计一个算法，通过观察训练集 $(\boldsymbol{X}^{(\text{train})}, \boldsymbol{y}^{(\text{train})})$ 获得经验，减少 $\text{MSE}\_{\text{test}}$ 以改进权重 $\boldsymbol{w}$。一种直观方式是最小化训练集上的均方误差，即 $\text{MSE}\_{\text{train}}$。最小化 $\text{MSE}\_{\text{train}}$，我们可以简单地求解其导数为 $0$ 的情况：

$$
\begin{gather}
\nabla_\boldsymbol{w} \text{MSE}_{\text{train}} = 0 \\
\Rightarrow \nabla_\boldsymbol{w} \frac{1}{m}\Big\Arrowvert\hat{\boldsymbol{y}}^{(\text{train})}-\boldsymbol{y}^{(\text{train})}\Big\Arrowvert_2^2=0\\
\Rightarrow \frac{1}{m} \nabla_\boldsymbol{w} \Big\Arrowvert \boldsymbol{X}^{(\text{train})}\boldsymbol{w}-\boldsymbol{y}^{(\text{train})}\Big\Arrowvert_2^2 = 0\\
\Rightarrow\nabla_\boldsymbol{w}\Big(\boldsymbol{X}^{(\text{train})}\boldsymbol{w}-\boldsymbol{y}^{(\text{train})}\Big)^\top\Big(\boldsymbol{X}^{(\text{train})}\boldsymbol{w}-\boldsymbol{y}^{(\text{train})}\Big) = 0\\
\Rightarrow\nabla_\boldsymbol{w}\Big(\boldsymbol{w}^\top\boldsymbol{X}^{(\text{train})} {}^\top \boldsymbol{X}^{(\text{train})} \boldsymbol{w} -2\boldsymbol{w}^\top\boldsymbol{X}^{(\text{train})} {}^\top\boldsymbol{y}^{(\text{train})} + \boldsymbol{y}^{(\text{train})} {}^\top\boldsymbol{y}^{(\text{train})}\Big) = 0\\
\Rightarrow2 \boldsymbol{X}^{(\text{train})} {}^\top \boldsymbol{X}^{(\text{train})}\boldsymbol{w}-2\boldsymbol{X}^{(\text{train})} {}^\top\boldsymbol{y}^{(\text{train})} = 0\\
\Rightarrow \boldsymbol{w} = \Big(\boldsymbol{X}^{(\text{train})} {}^\top\boldsymbol{X}^{(\text{train})}\Big)^{-1}\boldsymbol{X}^{(\text{train})} {}^\top\boldsymbol{y}^{(\text{train})}
\end{gather} \tag{6}
$$

通过式 $(6)$ 给出解的系统方程被称为**正规方程(normal equation)**。计算式 $(6)$ 构成了一个简单的机器学习算法。

值得注意的是，术语**线性回归** (linear regression) 通常用来指稍微复杂一些，附加额外参数(截距项 $b$)的模型。在这个模型中，

$$
\hat{y} = \boldsymbol{w}^\top\boldsymbol{x} + b \tag{7}
$$

因此从参数到预测的映射仍是一个线性函数，而从特征到预测的映射是一个仿射函数。如此扩展到仿射函数意味着模型预测的曲线仍然看起来像是一条直线，只是这条直线没必要经过原点。除了通过添加偏置参数 $b$，我们还可以使用仅含权重的模型，但是 $\boldsymbol{x}$ 需要增加一项永远为 $1$ 的元素。对应于额外 $1$ 的权重起到了偏置参数的作用。

> 本文内容摘取自 [《Deep Learning》](http://www.deeplearningbook.org/)，部分内容有修改。