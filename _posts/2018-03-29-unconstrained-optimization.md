---
layout: article
title: "无约束优化：梯度下降、牛顿法和拟牛顿法"
tags:
    - 机器学习
mathjax: true
sidebar:
  nav: machine-learning-note
---

许多机器学习模型的训练过程就是在求解无约束最优化问题，梯度下降法 (gradient descent)、牛顿法 (Newton method) 和拟牛顿法 (quasi Newton method) 都是求解这类问题的常用方法。其中梯度下降法实现简单，而牛顿法和拟牛顿法收敛速度快。

## 梯度下降法

假设 $f(\boldsymbol{x})$ 是 $\mathbb{R}^{n}$ 上具有一阶连续偏导数的函数。要求解的无约束最优化问题是

$$
\min_{\boldsymbol{x} \in \mathbb{R}^n} f(\boldsymbol{x}) \tag{1.1}
$$

$\boldsymbol{x}^*$ 表示目标函数 $f(\boldsymbol{x})$ 的极小点。

梯度下降法是一种迭代算法，选取适当的初值 $\boldsymbol{x}^{(0)}$，不断迭代，更新 $\boldsymbol{x}$ 的值，进行目标函数的极小化，直至收敛。由于负梯度方向是使函数值下降最快的方向，在迭代的每一步，以负梯度方向更新 $\boldsymbol{x}$ 的值，从而达到减少函数值的目的。

由于 $f(\boldsymbol{x})$ 具有一阶连续偏导数，若第 $k$ 次迭代值为 $\boldsymbol{x}^{(k)}$，则可将 $f(\boldsymbol{x})$ 在 $\boldsymbol{x}^{(k)}$ 附近进行一阶泰勒展开：

$$
f(\boldsymbol{x}) = f(\boldsymbol{x}^{(k)}) + \nabla f(\boldsymbol{x}^{(k)})^\top(\boldsymbol{x} - \boldsymbol{x}^{(k)}) \tag{1.2}
$$

为了满足 $f(\boldsymbol{x})<f(\boldsymbol{x}^{(k)})$，我们可以选择

$$
\boldsymbol{x} - \boldsymbol{x}^{(k)} = -\lambda_k \nabla f(\boldsymbol{x}^{(k)})
$$

即 $k+1$ 次迭代值 $\boldsymbol{x}^{(k+1)}$ 可以这样计算：

$$
\boldsymbol{x}^{(k+1)}\leftarrow \boldsymbol{x}^{(k)} - \lambda_k\nabla f(\boldsymbol{x}^{(k)}) \tag{1.3}
$$

其中 $\lambda_k$ 是步长，可以设置为固定常数，也可以由一维搜索确定，即 $\lambda_k$ 使得

$$
f(\boldsymbol{x}^{(k)} - \lambda_k\nabla f(\boldsymbol{x}^{(k)})) = \min_{\lambda\ge0}f(\boldsymbol{x}^{(k)} - \lambda\nabla f(\boldsymbol{x}^{(k)})) \tag{1.4}
$$

这就是梯度下降法。当目标函数为凸函数时，局部极小点就对应着函数的全局最小点，此时梯度下降法可确保收敛到全局最优解。

当目标函数 $f(\boldsymbol{x})$ 二阶连续可微时，可将 $(1.2)$ 替换为更精确的二阶泰勒展开式，这样就得到了牛顿法。但牛顿法使用了二阶导数 $\nabla^2f(\boldsymbol{x})$，其每轮迭代中涉及到海森矩阵的求逆，计算复杂度相当高，尤其在高维问题中几乎不可行，因此通过拟牛顿法以较低的计算代价寻找海森矩阵的近似逆矩阵。

## 牛顿法

考虑无约束最优化问题

$$
\min_{\boldsymbol{x} \in \mathbb{R}^n} f(\boldsymbol{x}) \tag{2.1}
$$

其中 $\boldsymbol{x}^*$ 为目标函数的极小点。

牛顿法的基本思想是：在现在极小点估计值附近对 $f(\boldsymbol{x})$ 做二阶泰勒展开，进而找到极小点的下一个估计值。假设 $f(\boldsymbol{x})$ 具有二阶连续偏导数，若第 $k$ 次迭代值为 $\boldsymbol{x}^{(k)}$，则可将 $f(\boldsymbol{x})$ 在 $\boldsymbol{x}^{(k)}$ 附近进行二阶泰勒展开：

$$
f(\boldsymbol{x}) = f(\boldsymbol{x}^{(k)}) + g_k^\top(\boldsymbol{x} - \boldsymbol{x}^{(k)})+\frac{1}{2}(\boldsymbol{x} - \boldsymbol{x}^{(k)})^\top H(\boldsymbol{x}^{(k)})(\boldsymbol{x} - \boldsymbol{x}^{(k)}) \tag{2.2}
$$

这里，$g_k = \nabla f(\boldsymbol{x}^{(k)})$ 是 $f(\boldsymbol{x})$ 的梯度向量在点 $\boldsymbol{x}^{(k)}$ 的值，$H(\boldsymbol{x}^{(k)})$ 是 $f(\boldsymbol{x})$ 的**海森矩阵** (Hesse matrix)

$$
H(\boldsymbol{x}) = \Bigg[\frac{\partial
^{2}f}{\partial x_i\partial x_j}\Bigg]_{n\times n} \tag{2.3}
$$

在点 $\boldsymbol{x}^{(k)}$ 的值。函数 $f(\boldsymbol{x})$ 有极值的必要条件是在极值点处一阶导数为 0，即梯度向量为 0。特别是当 $H(\boldsymbol{x}^{(k)})$ 是正定矩阵时，函数 $f(\boldsymbol{x})$ 的极值为极小值。

牛顿法利用极小点的必要条件

$$
\nabla f(\boldsymbol{x}) = 0 \tag{2.4}
$$

每次迭代中从点 $\boldsymbol{x}^{(k)}$ 开始，求目标函数的极小点，作为第 $k+1$ 次迭代值 $\boldsymbol{x}^{(k+1)}$。具体地，假设 $\boldsymbol{x}^{(k+1)}$ 满足：

$$
\nabla f(\boldsymbol{x}^{(k+1)}) = 0 \tag{2.5}
$$

由 $(2.2)$ 有

$$
\nabla f(\boldsymbol{x}) = g_k + H_{k}(\boldsymbol{x} - \boldsymbol{x}^{(k)}) \tag{2.6}
$$

其中 $H_k = H(\boldsymbol{x}^{(k)})$。这样 $(2.5)$ 可以化为

$$
g_k + H_k(\boldsymbol{x}^{(k+1)} - \boldsymbol{x}^{(k)}) = 0 \tag{2.7}
$$

因此

$$
\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)} - H^{-1}_kg_k \tag{2.8}
$$

使用 $(2.8)$ 作为迭代公式的算法就是牛顿法。

## 拟牛顿法

### 思路

但是在牛顿法的迭代中，需要计算海森矩阵的逆矩阵 $H^{-1}$，这一计算比较复杂，考虑用一个 $n$ 阶矩阵 $G$ 来近似代替 $H^{-1}$。这就是拟牛顿法的基本想法。

先看牛顿法迭代中海森矩阵 $H_k$ 满足的条件。类似于牛顿法，设经过 $K+1$ 次迭代后得到 $\boldsymbol{x}^{(k+1)}$，我们将 $f(\boldsymbol{x})$ 在 $\boldsymbol{x}^{(k+1)}$ 进行二阶泰勒展开：

$$
f(\boldsymbol{x}) = f(\boldsymbol{x}^{(k+1)}) + g_{k+1}^\top(\boldsymbol{x} - \boldsymbol{x}^{(k+1)})+\frac{1}{2}(\boldsymbol{x} - \boldsymbol{x}^{(k+1)})^\top H_{k+1}(\boldsymbol{x} - \boldsymbol{x}^{(k+1)}) \tag{3.1}
$$

于是可得

$$
\nabla f(\boldsymbol{x}) =g_{k+1} + H_{k+1}(\boldsymbol{x} - \boldsymbol{x}^{(k+1)}) \tag{3.2}
$$

在 $(3.2)$ 中取 $\boldsymbol{x} = \boldsymbol{x}^{(k)}$，即得

$$
g_{k+1} - g_{k} = H_{k+1}(\boldsymbol{x}^{(k+1)} - \boldsymbol{x}^{(k)}) \tag{3.3}
$$

记 $y_k = g_{k+1}-g_{k}$，$\delta_k = \boldsymbol{x}^{(k+1)} - \boldsymbol{x}^{(k)}$，则

$$
y_k = H_{k+1}\delta_k \tag{3.4}
$$

或

$$
H^{-1}_{k+1}y_k =\delta_k \tag{3.5}
$$

式 $(3.4)$ 或 $(3.5)$ 称为拟牛顿条件。

拟牛顿法将 $G_{k+1}$ 作为 $H^{-1}\_{k+1}$ 的近似，或者将 $B_{k+1}$ 作为 $H_{k+1}$，要求矩阵 $G_{k}$ 满足同样的条件。首先，每次迭代矩阵 $G_{k}$ 或者 $B_{k}$ 是正定的。同时类似 $(3.4)$ 和 $(3.5)$，$G_{k+1}$ 和 $B_{k+1}$ 满足下面的拟牛顿条件：

$$
G_{k+1}y_k =\delta_k \tag{3.6}
$$

$$
y_k = B_{k+1}\delta_k \tag{3.7}
$$

### DFP 算法  

DFP 是最早的拟牛顿法，该算法的核心是通过迭代的方法，对 $H^{-1}_{k+1}$ 做近似。迭代格式为：

$$
G_{k+1} = G_{k} + P_{k} + Q_{k} \tag{3.8}
$$

其中 $P_{k},Q_{k}$ 是待定矩阵。这时，

$$
G_{k+1} y_k= G_{k} y_k+ P_{k} y_k + Q_{k}y_k \tag{3.9}
$$

为了使 $G_{k+1}$ 满足拟牛顿条件 $(3.6)$，可使 $P_{k}$ 和 $Q_{k}$ 满足：

$$
P_{k} y_k =\delta_k \tag{3.10}
$$

$$
Q_{k} y_k = -G_{k}y_k \tag{3.11}
$$

事实上，不难找出这样的 $P_{k}$ 和 $Q_{k}$，例如取

$$
P_k = \frac{\delta_k \delta_k^\top}{\delta_k^\top y_k} \tag{3.12}
$$

$$
Q_k = -\frac{G_ky_ky_k^\top G_k}{y_k^\top G_k y_k} \tag{3.13}
$$

这样就可得到矩阵 $G_{k+1}$ 的迭代公式：

$$
G_{k+1} = G_k + \frac{\delta_k\delta_k^\top}{\delta_k^\top y_k}-\frac{G_ky_ky_k^\top G_k}{y_k^\top G_k y_k} \tag{3.14}
$$

称为 DFP 算法。可以证明，如果初始矩阵 $G_0$ 是正定的，则迭代过程中的每个矩阵 $G_k$ 都是正定的。

>**DFP 算法**
>
>输入：目标函数 $f(\boldsymbol{x})$，梯度 $g(\boldsymbol{x}) = \nabla f(\boldsymbol{x})$，精度要求 $\varepsilon$：
>
>输出：$f(\boldsymbol{x})$ 的极小点 $\boldsymbol{x}^*$
>
>(1) 选定初始点 $\boldsymbol{x}^{(0)}$，取 $G_0$ 为正定对称矩阵，置 $k=0$
>
>(2) 计算 $g_k = g(\boldsymbol{x}^{(k)}$。若 $\Vert g_k \Vert < \varepsilon$，则停止计算，得近似解 $\boldsymbol{x}^*=\boldsymbol{x}^{(k)}$；否则转 (3)
>
>(3) 置 $\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)}-G_kg_k$
>
>(4) 计算 $g_{k+1} = g(\boldsymbol{x}^{(k+1)})$，若 $\Vert g_{k+1}\Vert < \varepsilon$，则停止计算，得近似解 $\boldsymbol{x}^* = \boldsymbol{x}^{(k+1)}$；否则，按 $(3.14)$ 计算 $G_{k+1}$
>
>(5) 置 $k=k+1$，转 $(3)$

### BFGS 算法

BFGS 算法是最流行的拟牛顿算法。

可以考虑用 $G$ 逼近海森矩阵的逆矩阵 $H^{-1}$，也可以考虑用 $B$ 逼近海森矩阵 $H$。这时，相应的拟牛顿条件是 $(3.7)$。我们可以用同样地方法得到另一迭代公式。首先令

$$
B_{k+1} = B_k + P_k + Q_k \tag{3.15}
$$

$$
B_{k+1}\delta_k = B_k\delta_k + P_k\delta_k + Q_k\delta_k \tag{3.16}
$$

为了使 $B_{k+1}$ 满足拟牛顿条件 $(3.7)$，可使 $P_{k}$ 和 $Q_{k}$ 满足：

$$
P_k\delta_k = y_k \tag{3.17}
$$

$$
Q_k\delta_k = -B_k\delta_k \tag{3.18}
$$

找出适合条件的 $P_k$ 和 $Q_k$，得到 BFGS 算法矩阵 $B_{k+1}$ 的迭代公式：

$$
B_{k+1} = B_k + \frac{y_ky_k^\top}{y_k^\top\delta_k} - \frac{B_k\delta_k\delta_k^\top B_k}{\delta_k^\top B_k\delta_k} \tag{3.19}
$$

可以证明，如果初始矩阵 $B_0$ 是正定的，则迭代过程中的每个矩阵 $B_k$ 都是正定的。

> **BFGS 算法**
>
> 输入：目标函数 $f(\boldsymbol{x})$，梯度 $g(\boldsymbol{x}) = \nabla f(\boldsymbol{x})$，精度要求 $\varepsilon$：
>
> 输出：$f(\boldsymbol{x})$ 的极小点 $\boldsymbol{x}^*$
>
> (1) 选定初始点 $\boldsymbol{x}^{(0)}$，取 $B_0$ 为正定对称矩阵，置 $k=0$
>
> (2) 计算 $g_k = g(\boldsymbol{x}^{(k)}$。若 $\Vert g_k \Vert < \varepsilon$，则停止计算，得近似解 $\boldsymbol{x}^*=\boldsymbol{x}^{(k)}$；否则转 (3)
>
> (3) 置 $\boldsymbol{x}^{(k+1)} = \boldsymbol{x}^{(k)}-B^{-1}_kg_k$
>
> (4) 计算 $g_{k+1} = g(\boldsymbol{x}^{(k+1)})$，若 $\Vert g_{k+1}\Vert < \varepsilon$，则停止计算，得近似解 $\boldsymbol{x}^* = \boldsymbol{x}^{(k+1)}$；否则，按 $(3.19)$ 计算 $B_{k+1}$
>
> (5) 置 $k=k+1$，转 $(3)$

## 参考

peghoty[《牛顿法与拟牛顿法学习笔记》](https://blog.csdn.net/itplus/article/details/21896453)  
李航《统计学习方法》