---
layout: article
title: "浅谈泛化性：从随机噪声、对抗训练到梯度惩罚"
author: 苏剑林
tags:
    - NLP
    - 机器学习
mathjax: true
---

> 转载自[《对抗训练浅谈：意义、方法和思考（附Keras实现）》](https://kexue.fm/archives/7234) 和[《泛化性乱弹：从随机噪声、梯度惩罚到虚拟对抗训练》](https://kexue.fm/archives/7466)，作者：苏剑林。

提高模型的泛化性能是机器学习致力追求的目标之一。常见的提高泛化性的方法主要有两种：

- 第一种是添加噪声，比如往输入添加高斯噪声、中间层增加 Dropout 以及对抗训练等，对图像进行随机平移缩放等数据扩增手段某种意义上也属于此列；
- 第二种是往 loss 里边添加正则项，比如 $L_1, L_2$ 惩罚、梯度惩罚等。

本文试图探索几种常见的提高泛化性能的手段的关联。

## 添加随机噪声

我们记模型为 $f(x)$，$\mathcal{D}$ 为训练数据集合，$l(f(x), y)$ 为单个样本的 loss，那么我们的优化目标是

$$
\mathop{\arg\min}_{\theta} L(\theta)=\mathbb{E}_{(x,y)\sim \mathcal{D}}[l(f(x), y)] \tag{1}
$$

$\theta$ 是 $f(x)$ 里边的可训练参数。假如往模型输入添加噪声 $\varepsilon$，其分布为 $q(\varepsilon)$，那么优化目标就变为

$$
\mathop{\arg\min}_{\theta} L_{\varepsilon}(\theta)=\mathbb{E}_{(x,y)\sim \mathcal{D}, \varepsilon\sim q(\varepsilon)}[l(f(x + \varepsilon), y)] \tag{2}
$$

当然，可以添加噪声的地方不仅仅是输入，也可以是中间层，也可以是权重 $\theta$，甚至可以是输出 $y$（等价于标签平滑），噪声也不一定是加上去的，比如 Dropout 是乘上去的。对于加性噪声来说，$q(\varepsilon)$ 的常见选择是均值为 0、方差固定的高斯分布；而对于乘性噪声来说，常见选择是均匀分布 $U([0,1])$ 或者是伯努利分布。

添加随机噪声的目的很直观，就是希望模型能学会抵御一些随机扰动，从而降低对输入或者参数的敏感性，而降低了这种敏感性，通常意味着所得到的模型不再那么依赖训练集，所以有助于提高模型泛化性能。

### 提高效率

添加随机噪声的方式容易实现，而且在不少情况下确实也很有效，但它有一个明显的缺点：不够“特异性”。噪声 $\varepsilon$ 是随机的，而不是针对 $x$ 构建的，这意味着多数情况下 $x + \varepsilon$ 可能只是一个平凡样本，也就是没有对原模型造成比较明显的扰动，所以对泛化性能的提高帮助有限。

从理论上来看，加入随机噪声后，单个样本的 loss 变为

$$
\tilde{l}(x,y)=\mathbb{E}_{\varepsilon\sim q(\varepsilon)}[l(f(x+\varepsilon),y)]=\int q(\varepsilon) l(f(x+\varepsilon),y) d\varepsilon\label{eq:noisy-loss} \tag{3}
$$

但实践上，对于每个特定的样本 $(x,y)$，我们一般只采样一个噪声，所以并没有很好地近似上式。当然，我们可以采样多个噪声 $\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_k\sim q(\varepsilon)$，然后更好地近似

$$
\tilde{l}(x,y)\approx \frac{1}{k}\sum_{i=1}^k l(f(x+\varepsilon_i),y) \tag{4}
$$

但这样相当于 batch size 扩大为原来的 $k$ 倍，增大了计算成本，并不是那么友好。

一个直接的想法是，如果能事先把式 $(3)$ 中的积分算出来，那就用不着低效率地采样了（或者相当于一次性采样无限多的噪声）。我们就往这个方向走一下试试。当然，精确的显式积分基本上是做不到的，我们可以做一下近似展开：

$$
l(f(x+\varepsilon),y)\approx l(f(x),y)+(\varepsilon \cdot \nabla_x) l(f(x),y)+\frac{1}{2}(\varepsilon \cdot \nabla_x)^2 l(f(x),y) \tag{5}
$$

然后两端乘以 $q(\varepsilon)$ 积分，这里假设 $\varepsilon$ 的各个分量是独立同分布的，并且均值为 0、方差为 $\sigma^2$，那么积分结果就是

$$
\int q(\varepsilon)l(f(x+\varepsilon),y)d\varepsilon \approx l(f(x),y)+\frac{1}{2}\sigma^2 \Delta l(f(x),y) \tag{6}
$$

这里的 $\Delta$ 是拉普拉斯算子，即 $\Delta f = \sum\limits_i \frac{\partial^2}{\partial x_i^2} f$。这个结果在形式上很简单，就是相当于往 loss 里边加入正则项 $\frac{1}{2}\sigma^2 \Delta l(f(x),y)$，然而实践上却相当困难，因为这意味着要算 $l$ 的二阶导数，再加上梯度下降，那么就一共要算三阶导数，这是现有深度学习框架难以实现的。

### 转移目标

直接化简 $l(f(x+\varepsilon),y)$ 的积分是行不通了，但我们还可以试试将优化目标换成

$$
l(f(x+\varepsilon),f(x)) + l(f(x),y)\label{eq:loss-2} \tag{7}
$$

也就是变成同时缩小 $f(x),y$、$f(x+\varepsilon),f(x)$ 的差距，两者双管齐下，一定程度上也能达到缩小 $f(x+\varepsilon),y$ 差距的目标。

> 用数学的话来讲，如果 $l$ 是某种形式的距离度量，那么根据三角不等式就有
>
> $$
> l(f(x+\varepsilon),y) \leq l(f(x+\varepsilon),f(x)) + l(f(x),y)
> $$
>
> 如果 $l$ 不是度量，那么通常根据詹森不等式也能得到一个类似的结果，比如 $l(f(x+\varepsilon),y)=\Vert f(x+\varepsilon) - y\Vert^2$，那么我们有
>
> $$
> \begin{aligned} 
> \Vert f(x+\varepsilon) - f(x) + f(x) - y\Vert^2 =& \left\Vert \frac{1}{2}\times 2[f(x+\varepsilon) - f(x)] + \frac{1}{2}\times 2[f(x) - y]\right\Vert^2\\ 
> \leq& \frac{1}{2} \Vert 2[f(x+\varepsilon) - f(x)]\Vert^2 + \frac{1}{2} \Vert 2[f(x) - y]\Vert^2\\ 
> =& 2\big(\Vert f(x+\varepsilon) - f(x)\Vert^2 + \Vert f(x) - y\Vert^2\big) 
> \end{aligned}
> $$
>
> 这也就是说，目标 $(7)$（的若干倍）可以认为是 $l(f(x+\varepsilon),y)$ 的上界，原始目标不大好优化，所以我们改为优化它的上界。

注意到，目标 $(7)$ 的两项之中，$l(f(x+\varepsilon),f(x))$ 衡量了模型本身的平滑程度，跟标签没关系，用无标签数据也可以对它进行优化，这意味着它可以跟带标签的数据一起，构成一个**半监督学习**流程。

对于目标 $(7)$ 来说，它的积分结果是：

$$
\begin{aligned}
\int q(\varepsilon) \big[l(f(x+\varepsilon),f(x)) + l(f(x),y)\big]d\varepsilon & = l(f(x),y) + \int q(\varepsilon) l(f(x+\varepsilon),f(x)) d\varepsilon\\
& \approx l(f(x),y) + \frac{1}{2}\sigma^2 \sum_i \lambda_i(x)\Vert \nabla_x f_i(x)\Vert^2
\end{aligned} \tag{8}
$$

这其实就是对每个 $f(x)$ 的每个分量都算一个梯度惩罚项 $\Vert \nabla_x f_i(x)\Vert^2$，然后按 $\lambda_i(x)$ 加权求和。

对于 MSE 来说，$l(f(x),y)=\Vert f(x) - y\Vert^2$，这时候可以算得 $\lambda_i(x)\equiv 2$，所以对应的正则项为 $\sum\limits_i\Vert \nabla_x f_i(x)\Vert^2$；对于 KL 散度来说，$l(f(x),y)=\sum\limits_i y_i \log \frac{y_i}{f_i(x)}$，这时候 $\frac{1}{f_i(x)}$，那么对应的正则项为 $\sum\limits_i f_i(x) \Vert \nabla_x \log f_i(x)\Vert^2$。这些结果大家多多少少可以从著名的“花书”[《深度学习》](https://book.douban.com/subject/27087503/)中找到类似的，所以并不是什么新的结果。类似的推导还可以参考文献[《Training with noise is equivalent to Tikhonov regularization》](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf)。

当然，虽然能求出只带有一阶梯度的正则项 $\sum\limits_i \lambda_i(x)\Vert \nabla_x f_i(x)\Vert^2$，但事实上这个计算量也不低，因为需要对每个 $f_i(x)$ 都要求梯度，如果输出的分量数太大，这个计算量依然难以承受。

这时候可以考虑的方案是通过采样近似计算：假设 $q(\eta)$ 是均值为 0、方差为 1 的分布，那么我们有

$$
\sum\limits_i \Vert \nabla_x f_i(x)\Vert^2=\mathbb{E}_{\eta_i\sim q(\eta)}\left[\left\Vert\sum_i \eta_i \nabla_x f_i(x)\right\Vert^2\right] \tag{9}
$$

这样一来，每步我们只需要算 $\sum\limits_i \eta_i f_i(x)$ 的梯度，不需要算多次梯度。$q(\eta)$ 的一个最简单的取法是空间为 $\\{-1,1\\}$ 的均匀分布，也就是 $\eta_i$ 等概率地从 $\\{-1,1\\}$ 中选取一个。

## 对抗训练

前面我们提到，添加随机加噪声可能太没特异性，所以想着先把积分算出来，才有了后面推导的关于近似展开与梯度惩罚的一些结果。换个角度想，如果我们能想办法更特异性地构造噪声信号，那么也能提高训练效率，增强泛化性能。

### Min-Max

总的来说，对抗训练可以统一写成如下格式

$$
\min_{\theta}\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\max_{\varepsilon\in\Omega}l(f(x+\varepsilon), y)\right]\label{eq:min-max} \tag{10}
$$

其中 $\mathcal{D}$ 代表训练集，$x$ 代表输入，$y$ 代表标签，$\theta$ 是模型参数，$l(f(x),y)$ 是单个样本的 loss，$\varepsilon$ 是对抗扰动，$\Omega$ 是扰动空间。

> 这个统一的格式首先由论文[《Towards Deep Learning Models Resistant to Adversarial Attacks》](https://arxiv.org/abs/1706.06083)提出。

这个式子可以分步理解如下：

1. 往属于 $x$ 里边注入扰动 $\varepsilon$，$\varepsilon$ 的目标是让 $l(f(x+\varepsilon), y)$ 越大越好，也就是说尽可能让现有模型的预测出错；
2. 当然 $\varepsilon$ 也不是无约束的，它不能太大，否则达不到“看起来几乎一样”的效果，所以 $\varepsilon$ 要满足一定的约束，常规的约束是 $\Vert\varepsilon\Vert\leq \epsilon$，其中 $\epsilon$ 是一个常数；
3. 每个样本都构造出对抗样本 $x + \varepsilon$ 之后，用 $(x + \varepsilon, y)$ 作为数据对去最小化 loss 来更新参数 $\theta$（梯度下降）；
4. 反复交替执行 1、2、3 步。

由此观之，整个优化过程是 $\max$ 和 $\min$ 交替执行，这确实跟 GAN 很相似，不同的是，GAN 所 $\max$ 的自变量也是模型的参数，而这里 $\max$ 的自变量则是输入（的扰动量），也就是说要对每一个输入都定制一步 $\max$。

现在的问题是如何计算 $\varepsilon$，它的目标是增大 $l(f(x+\varepsilon), y)$，而我们知道让 loss 减少的方法是梯度下降，那反过来，让 loss 增大的方法自然就是梯度上升，因此可以简单地取

$$
\varepsilon = \epsilon \nabla_x l(f(x), y) \tag{11}
$$

当然，为了防止 $\varepsilon$ 过大，通常要对 $\nabla_x l(f(x), y)$ 做些标准化，比较常见的方式是

$$
\varepsilon = \epsilon \frac{\nabla_x l(f(x), y)}{\Vert \nabla_x l(f(x), y)\Vert}\quad\text{或}\quad \varepsilon = \epsilon \text{sign}(\nabla_x l(f(x), y)) \tag{12}
$$

有了 $\varepsilon$ 之后，就可以代回式 $(10)$ 进行优化

$$
\min_{\theta}\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[l(f(x+\varepsilon), y)\right] \tag{13}
$$

这就构成了一种对抗训练方法，被称为 **Fast Gradient Method（FGM）**，它由 GAN 之父 Goodfellow 在论文[《Explaining and Harnessing Adversarial Examples》](https://arxiv.org/abs/1412.6572)首先提出。

> 对抗训练还有一种方法，叫做 **Projected Gradient Descent（PGD）**，其实就是通过多迭代几步来达到让 $l(f(x+\varepsilon),y)$ 更大的 $\varepsilon$。如果迭代过程中模长超过了 $\epsilon$，就缩放回去，细节请参考[《Towards Deep Learning Models Resistant to Adversarial Attacks》](https://arxiv.org/abs/1706.06083)。

对于 CV 领域的任务，上述对抗训练的流程可以顺利执行下来，因为图像可以视为普通的连续实数向量，$\varepsilon$ 也是一个实数向量，因此 $x + \varepsilon$ 依然可以是有意义的图像。但 NLP 不一样，NLP 的输入是文本，它本质上是 one hot 向量，而两个不同的 one hot 向量，其欧氏距离恒为 $\sqrt{2}$，因此对于理论上不存在什么“小扰动”。

一个自然的想法是像论文[《Adversarial Training Methods for Semi-Supervised Text Classification》](https://arxiv.org/abs/1605.07725)一样，将扰动加到 Embedding 层。这个思路在操作上没有问题，但问题是，扰动后的 Embedding 向量不一定能匹配上原来的 Embedding 向量表，这样一来对 Embedding 层的扰动就无法对应上真实的文本输入，这就不是真正意义上的对抗样本了，因为对抗样本依然能对应一个合理的原始输入。

那么，在 Embedding 层做对抗扰动还有没有意义呢？有！实验结果显示，在很多任务中，在 Embedding 层进行对抗扰动能有效提高模型的性能。

### 实验结果

对于 CV 任务来说，一般输入张量的 shape 是 $(b,h,w,c)$，这时候我们需要固定模型的 batch size（即 $b$），然后给原始输入加上一个 shape 同样为 $(b,h,w,c)$、全零初始化的 `Variable`，比如就叫做 $\varepsilon$，那么我们可以直接求 loss 对 $x$ 的梯度，然后根据梯度给 $\varepsilon$ 赋值，来实现对输入的干扰，完成干扰之后再执行常规的梯度下降。

对于 NLP 任务来说，原则上也要对 Embedding 层的输出进行同样的操作，Embedding 层的输出 shape 为 $(b,n,d)$，所以也要在 Embedding 层的输出加上一个 shape 为 $(b,n,d)$ 的 `Variable`，然后进行上述步骤。但这样一来，我们需要拆解、重构模型，对使用者不够友好。

不过，我们可以退而求其次。Embedding 层的输出是直接取自于 Embedding 参数矩阵的，因此我们可以直接对 Embedding 参数矩阵进行扰动。这样得到的对抗样本的多样性会少一些（因为不同样本的同一个 token 共用了相同的扰动），但仍然能起到正则化的作用，而且这样实现起来容易得多。

基于上述思路，这里给出 Keras 下基于 FGM 方式对 Embedding 层进行对抗训练的参考实现：

<center><a href="https://github.com/bojone/keras_adversarial_training" target="_blank">https://github.com/bojone/keras_adversarial_training</a></center>

核心代码如下：

```python
def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数
```

定义好上述函数后，给 Keras 模型增加对抗训练就只需要一行代码了：

```python
# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', 0.5)
```

需要指出的是，由于每一步算对抗扰动也需要计算梯度，因此每一步训练一共算了两次梯度，因此每步的训练时间会翻倍。

为了测试实际效果，我们选了[中文 CLUE 榜](https://www.cluebenchmarks.com/)的两个分类任务：IFLYTEK 和 TNEWS，模型选择了中文 BERT base。在 CLUE 榜单上，BERT base 模型在这两个数据上的成绩分别是 60.29% 和 56.58%，经过对抗训练后，成绩为 62.46%、57.66%，分别提升了 2% 和 1%！

$$
\begin{array}{c|cc} 
\hline 
& \text{IFLYTEK} & \text{TNEWS} \\ 
\hline 
\text{无对抗训练} & 60.29\% & 56.58\% \\ 
\text{加对抗训练} & 62.46\% & 57.66\% \\ 
\hline 
\end{array}
$$

训练脚本请参考：[task_iflytek_adversarial_training.py](https://github.com/bojone/bert4keras/blob/master/examples/task_iflytek_adversarial_training.py)。

当然，同所有正则化手段一样，对抗训练也不能保证每一个任务都能有提升，但从目前大多数“战果”来看，它是一种非常值得尝试的技术手段。

> BERT 的 finetune 本身就是一个非常玄学的过程，论文[《Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping》](https://arxiv.org/abs/2002.06305)换用不同的随机种子跑了数百次 finetune 实验，发现最好的结果能高出好几个点，所以如果你跑了一次发现没提升，不妨多跑几次再下结论。

### 延伸思考

假设已经得到对抗扰动 $\varepsilon$，那么我们在更新 $\theta$ 时，考虑对 $l(f(x+\varepsilon), y)$ 的展开：

$$
\begin{aligned}&\min_{\theta}\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[l(f(x+\varepsilon), y)\right]\\ 
\approx&\, \min_{\theta}\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[l(f(x), y)+\langle\nabla_x l(f(x), y), \varepsilon\rangle\right] 
\end{aligned} \tag{14}
$$

对应的 $\theta$ 的梯度为

$$
\begin{equation}\nabla_{\theta}l(f(x), y)+\langle\nabla_{\theta}\nabla_x l(f(x), y), \varepsilon\rangle\end{equation}
$$

代入 $\varepsilon=\epsilon \nabla_x l(f(x), y)$，得到

$$
\begin{aligned}&\nabla_{\theta}l(f(x), y)+\epsilon\langle\nabla_{\theta}\nabla_x l(f(x), y), \nabla_x l(f(x), y)\rangle\\ 
=&\,\nabla_{\theta}\left(l(f(x), y)+\frac{1}{2}\epsilon\left\Vert\nabla_x l(f(x), y)\right\Vert^2\right) 
\end{aligned} \tag{15}
$$

这个结果表示，对输入样本施加 $\epsilon \nabla_x l(f(x), y)$ 的对抗扰动，一定程度上等价于往 loss 里边加入“梯度惩罚”

$$
\frac{1}{2}\epsilon\left\Vert\nabla_x l(f(x), y)\right\Vert^2\label{eq:gp}\tag{16}
$$

如果对抗扰动是 $\epsilon \nabla_x l(f(x), y)/\Vert \nabla_x l(f(x), y)\Vert$，那么对应的梯度惩罚项则是 $\epsilon\left\Vert\nabla_x l(f(x), y)\right\Vert$。

这又跟前一节的关于噪声积分的结果类似，表明梯度惩罚应该是通用的能提高模型性能的手段之一。

> 这个结果首先出现在论文[《Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients》](https://arxiv.org/abs/1711.09404)里。

## 梯度惩罚

### 几何图像

事实上，关于梯度惩罚，我们有一个非常直观的几何图像。以常规的分类问题为例，假设有 $n$ 个类别，那么模型相当于挖了 $n$ 个坑，然后让同类的样本放到同一个坑里边去：

<img src="/img/article/talking-about-generalization/classification.png" width="550px" style="display: block; margin: auto;"/>

<center>分类问题就是挖坑，然后将同类样本放在同一个坑内</center>

梯度惩罚则说“**同类样本不仅要放在同一个坑内，还要放在坑底**”，这就要求每个坑的内部要长这样：

<img src="/img/article/talking-about-generalization/adversarial_training.png" width="320px" style="display: block; margin: auto;"/>

<center>对抗训练希望每个样本都在一个“坑中坑”的坑底</center>

为什么要在坑底呢？因为物理学告诉我们，坑底最稳定呀，所以就越不容易受干扰呀，这不就是对抗训练的目的么？

<img src="/img/article/talking-about-generalization/adversarial_training.gif" width="250px" style="display: block; margin: auto;"/>

<center>“坑底”最稳定。受到干扰后依然在坑底附近徘徊，不容易挑出坑</center>

那坑底意味着什么呢？极小值点呀，导数（梯度）为零呀，所以不就是希望 $\Vert\nabla_x l(f(x),y)\Vert$ 越小越好么？这便是梯度惩罚 $(16)$ 的几何意义了。

### 代码实现

相比前面的 FGM 式对抗训练，其实梯度惩罚实现起来还容易一些，因为它就是在 loss 里边多加一项罢了，而且实现方式是通用的，不用区分 CV 还是 NLP。

Keras 参考实现如下：

```python
def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp


model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)
```

可以看到，定义带梯度惩罚的 loss 非常简单，就两行代码而已。需要指出的是，梯度惩罚意味着参数更新的时候需要算二阶导数，但是 Tensorflow 和 Keras 自带的 loss 函数不一定支持算二阶导数，比如 `K.categorical_crossentropy` 支持而 `K.sparse_categorical_crossentropy` 不支持，遇到这种情况时，需要自定重新定义 loss。

### 效果比较

还是前面两个任务，结果如下表。可以看到，梯度惩罚能取得跟 FGM 基本一致的结果。

$$
\begin{array}{c|cc} 
\hline 
& \text{IFLYTEK} & \text{TNEWS} \\ 
\hline 
\text{无对抗训练} & 60.29\% & 56.58\% \\ 
\text{加对抗训练} & 62.46\% & 57.66\% \\ 
\text{加梯度惩罚} & 62.31\% & 57.81\% \\ 
\hline 
\end{array}
$$

完整的代码请参考：[task_iflytek_gradient_penalty.py](https://github.com/bojone/bert4keras/blob/master/examples/task_iflytek_gradient_penalty.py)。

> 转载自[《对抗训练浅谈：意义、方法和思考（附Keras实现）》](https://kexue.fm/archives/7234) 和[《泛化性乱弹：从随机噪声、梯度惩罚到虚拟对抗训练》](https://kexue.fm/archives/7466)，作者：苏剑林。

