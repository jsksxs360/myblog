---
layout: article
title: "深度学习中的优化方法：梯度下降和约束优化"
tags:
    - 机器学习
mathjax: true
sidebar:
  nav: machine-learning-note
---

大多数深度学习算法都涉及某种形式的优化。优化指的是改变 $\boldsymbol{x}$ 以最小化或最大化某个函数 $f(\boldsymbol{x})$ 的任务。我们通常以最小化 $f(\boldsymbol{x})$ 指代大多数最优化问题，最大化可经由最小化算法最小化 $−f(\boldsymbol{x})$ 来实现。

我们把要最小化或最大化的函数称为**目标函数** (objective function)，当我们对其进行最小化时，我们也把它称为**代价函数**(cost function)、**损失函数** (loss function)或**误差函数** (error function)。

我们通常使用一个上标 $\*$ 表示最小化或最大化函数的 $\boldsymbol{x}$ 值。如我们记 $\boldsymbol{x}^* = \arg \min f (\boldsymbol{x})$。

## 基于梯度的优化方法

假设我们有一个函数 $y = f (x)$，其中 $x$ 和 $y$ 是实数。这个函数的**导数** (derivative) 记为 $f'(x)$ 或 $\frac{dy}{dx}$ 。导数 $f'(x)$ 代表 $f(x)$ 在点 $x$ 处的斜率。换句话说，它表明如何缩放输入的小变化才能在输出获得相应的变化：$f(x + \epsilon) \approx f(x) +\epsilon f'(x)$。

### 梯度下降和驻点

因此导数对于最小化一个函数很有用，因为它告诉我们如何更改 $x$ 来略微地改善 $y$。例如，我们知道对于足够小的 $\epsilon$ 来说，$f(x − \epsilon \text{sign}(f'(x)))$ 是比 $f(x)$ 小的。因此我们可以将 $x$ 往导数的反方向移动一小步来减小 $f(x)$。这种技术被称为**梯度下降** (gradient descent)。下图展示了一个例子。

<img src="/img/article/optimization-methods/gradient_descent.jpg" style="display:block;margin:auto;"/>

当 $f'(x) = 0$，导数无法提供往哪个方向移动的信息。$f'(x) = 0$ 的点称为**驻点** (stationary point)。一个**局部极小点** (local minimum) 意味着这个点的 $f(x)$ 小于所有邻近点，因此不可能通过移动无穷小的步长来减小 $f (x)$；一个**局部极大点** (local maximum) 意味着这个点的 $f (x)$ 大于所有邻近点，因此不可能通过移动无穷小的步长来增大 $f(x)$；一个**鞍点** (saddle point) 意味着这个点的 $f(x)$ 存在更高和更低的相邻点。下图是一维情况下，三种临界点的示例。

<img src="/img/article/optimization-methods/point.jpg" style="display:block;margin:auto;"/>

使 $f(x)$ 取得绝对的最小值（相对所有其他值）的点是**全局最小点** (global minimum)。函数可能只有一个全局最小点或存在多个全局最小点，还可能存在不是全局最优的局部极小点。

当存在多个局部极小点或平坦区域时，优化算法可能无法找到全局最小点。 在深度学习的背景下，即使找到的解不是真正最小的，但只要它们是对应于代价函数显著低的值，我们通常就能接受这样的解。

### 多维输入

我们经常最小化具有多维输入的函数：$f : \mathbb{R}^n \rightarrow \mathbb{R}$。为了使“最小化”的概念有意义，输出必须是一维的标量。

针对具有多维输入的函数，我们需要用到**偏导数** (partial derivative) 的概念。 偏导数 $\frac{\partial}{\partial x_i} f (\boldsymbol{x})$ 衡量点 $\boldsymbol{x}$ 处只有 $x_i$ 增加时 $f (\boldsymbol{x})$ 如何变化。**梯度** (gradient) 是相对一个向量求导的导数：$f$ 的导数是包含所有偏导数的向量，记为 $\nabla_\boldsymbol{x}f(\boldsymbol{x})$。梯度的第 $i$ 个元素是 $f$ 关于 $x_i$ 的偏导数。在多维情况下，临界点是梯度中所有元素都为零的点。

为了最小化 $f$，我们希望找到使 $f$ 下降得最快的方向。因为梯度向量指向上坡，负梯度向量指向下坡，所以我们在负梯度方向上移动可以最快地减小 $f$。这被称为**最速下降法** (method of steepest descent) 或**梯度下降** (gradient descent)。

最速下降建议新的点为

$$
\boldsymbol{x}' = \boldsymbol{x} - \epsilon \nabla_\boldsymbol{x}f(\boldsymbol{x}) \tag{1.1}
$$

其中 $\epsilon$ 为**学习率** (learning rate)，是一个确定步长大小的正标量，普遍的方式是选择一个小常数。最速下降在梯度的每一个元素为零时收敛 (或在实践中，很接近零时)。在某些情况下，我们也许能够避免运行该迭代算法，并通过解方程 $\nabla_\boldsymbol{x}f(\boldsymbol{x}) = 0$ 直接跳到临界点。

## 约束优化

有时候，在 $\boldsymbol{x}$ 的所有可能值下最大化或最小化一个函数 $f(\boldsymbol{x})$ 不是我们所希望的。相反，我们可能希望在 $\boldsymbol{x}$ 的某些集合 $\mathbb{S}$ 中找 $f(\boldsymbol{x})$ 的最大值或最小值，这被称为**约束优化** (constrained optimization)。在约束优化术语中，集合 $\mathbb{S}$ 内的点 $\boldsymbol{x}$ 被称为**可行** (feasible) 点。

我们常常希望找到在某种意义上小的解。针对这种情况下的常见方法是强加一个范数约束，如 $\Vert x \Vert ≤ 1$。

约束优化的一个简单方法是将约束考虑在内后简单地对梯度下降进行修改。一个更复杂的方法是设计一个不同的、无约束的优化问题，其解可以转化成原始约束优化问题的解。

$\textbf{Karush–Kuhn–Tucker}$ (KKT) 方法是针对约束优化非常通用的解决方案。 为介绍 $\text{KKT}$ 方法，我们引入一个称为**广义 Lagrangian** (generalized Lagrangian) 或**广义 Lagrange 函数** (generalized Lagrange function) 的新函数。

### 广义拉格朗日函数 

为了定义 Lagrangian，我们先要通过等式和不等式的形式描述 $\mathbb{S}$。我们希望通过 $m$ 个函数 $g^{(i)}$ 和 $n$ 个函数 $h^{(j)}$ 描述 $\mathbb{S}$，那么 $\mathbb{S}$ 可以表示为

$$
\mathbb{S} = \{\boldsymbol{x}\mid \forall i,g^{(i)}(\boldsymbol{x})= 0 \text{ and } \forall j, h^{(j)}(\boldsymbol{x}) \le 0\} \tag{2.1}
$$

其中涉及 $g^{(i)}$ 的等式称为**等式约束** (equality constraint)，涉及 $h^{(j)}$ 的不等式称为**不等式约束** (inequality constraint)。

我们为每个约束引入新的变量 $\lambda_i$ 和 $\alpha_j$，这些新变量被称为 $\text{KKT}$ 乘子。广义 Lagrangian 可以如下定义：

$$
L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\alpha}) = f(\boldsymbol{x}) + \sum_i \lambda_i g^{(i)}(\boldsymbol{x}) + \sum_j \alpha_j h^{(j)}(\boldsymbol{x}) \tag{2.2}
$$

现在，我们可以通过优化无约束的广义 Lagrangian 解决约束最小化问题。只要存在至少一个可行点且 $f(\boldsymbol{x})$ 不允许取 $\infty$，那么

$$
\min_\boldsymbol{x} \max_{\boldsymbol{\lambda}} \max_{\boldsymbol{\alpha},\boldsymbol{\alpha}\ge0} L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\alpha}) \tag{2.3}
$$

与如下函数有相同的最优目标函数值和最优点集 $\boldsymbol{x}$

$$
\min_{\boldsymbol{x}\in\mathbb{S}} f(\boldsymbol{x}) \tag{2.4}
$$

这是因为当约束满足时，

$$
\max_{\boldsymbol{\lambda}} \max_{\boldsymbol{\alpha},\boldsymbol{\alpha}\ge0} L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\alpha}) = f(\boldsymbol{x}) \tag{2.4}
$$

而违反任意约束时，

$$
\max_{\boldsymbol{\lambda}} \max_{\boldsymbol{\alpha},\boldsymbol{\alpha}\ge0} L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\alpha}) = \infty \tag{2.5}
$$

> 因为若某个 $i$ 使约束 $g^{(i)} \ne 0$，则可令 $\lambda_i$ 使 $\lambda_i g^{(i)}\rightarrow \infty$，若某个 $j$ 使约束 $h^{(j)}>0$，则可令 $\alpha_j \rightarrow \infty$，而将其余各 $\lambda_i,\alpha_j$ 均取为 0。

即

$$
\max_{\boldsymbol{\lambda}} \max_{\boldsymbol{\alpha},\boldsymbol{\alpha}\ge0} L(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\alpha}) = \begin{cases}f(\boldsymbol{x}),& \boldsymbol{x} \text{ 满足约束}\\\infty,&\text{其他}\end{cases} \tag{2.6}
$$

这些性质保证不可行点不会是最佳的，并且可行点范围内的最优点不变，即 $(2.3)$ 与原始优化问题等价，它们有相同的解。

与上面类似，如果我们要解决约束最大化问题，可以构造 $−f(\boldsymbol{x})$ 的广义 Lagrange 函数，从而导致以下优化问题：

$$
\min_{\boldsymbol{x}} \max_{\boldsymbol{\lambda}}\max_{\boldsymbol{\alpha},\boldsymbol{\alpha}\ge0} -f(\boldsymbol{x}) + \sum_i \lambda_ig^{(i)}(\boldsymbol{x}) + \sum_j \alpha_j h^{(j)}(\boldsymbol{x}) \tag{2.7}
$$

我们也可将其转换为在外层最大化的问题：

$$
\max_{\boldsymbol{x}}\min_{\boldsymbol{\lambda}}\min_{\boldsymbol{\alpha},\boldsymbol{\alpha}\ge 0} f(\boldsymbol{x}) + \sum_i\lambda_i g^{(i)}(\boldsymbol{x}) -\sum_j\alpha_jh^{(j)}(\boldsymbol{x}) \tag{2.8}
$$

等式约束对应项的符号并不重要，因为优化可以自由选择每个 $\lambda_i$ 的符号，我们可以随意将其定义为加法或减法。

### KKT

不等式约束特别有趣。如果 $h^{(i)}(\boldsymbol{x}^*) = 0$，我们就说说这个约束 $h^{(i)}(\boldsymbol{x})$ 是活跃 (active) 的。如果约束不是活跃的，则有该约束的问题的解与去掉该约束的问题的解至少存在一个相同的局部解。一个不活跃约束有可能排除其他解。例如，整个区域 (代价相等的宽平区域) 都是全局最优点的的凸问题可能因约束消去其中的某个子区域，或在非凸问题的情况下，收敛时不活跃的约束可能排除了较好的局部驻点。

然而，无论不活跃的约束是否被包括在内，收敛时找到的点仍然是一个驻点。因为一个不活跃的约束 $h^{(i)}$ 必有负值，那么 $\min_{\boldsymbol{x}} \max_{\boldsymbol{\lambda}} \max_{\boldsymbol{\alpha},\boldsymbol{\alpha \ge 0}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\alpha})$ 中的 $\alpha_i = 0$。因此，我们可以观察到在该解中 $\boldsymbol{\alpha} \odot \boldsymbol{h}(\boldsymbol{x}) = 0$。换句话说，对于所有的 $i$，$\alpha_i \ge 0$ 或 $h^{(j)}(\boldsymbol{x}) \le 0$ 在收敛时必有一个是活跃的。

为了获得关于这个想法的一些直观解释， 我们可以说这个解是由不等式强加的边界，我们必须通过对应的 KKT 乘子影响 $\boldsymbol{x}$ 的解，或者不等式对解没有影响，我们则归零 KKT 乘子。

我们可以使用一组简单的性质来描述约束优化问题的最优点。这些性质称为 $\textbf{Karush–Kuhn–Tucker}$ (KKT) 条件。这些是确定一个点是最优点的必要条件，但不一定是充分条件。这些条件是：

- 广义 Lagrangian 的梯度为零。
- 所有关于 $\boldsymbol{x}$ 和 KKT 乘子的约束都满足。
- 不等式约束显示的“互补松弛性”：$\boldsymbol{\alpha} \odot \boldsymbol{h}(\boldsymbol{x}) = 0$。

> 本文为 [《Deep Learning》](http://www.deeplearningbook.org/)的学习笔记，感谢 [Exacity 小组](https://github.com/exacity)的翻译。