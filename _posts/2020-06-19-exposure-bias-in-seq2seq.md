---
layout: article
title: "Seq2Seq 中 Exposure Bias 现象的浅析与对策"
author: 苏剑林
tags:
    - NLP
    - 机器学习
mathjax: true
---

> 转载自[《Seq2Seq中Exposure Bias现象的浅析与对策》](https://kexue.fm/archives/7259)，作者：苏剑林，部分内容有修改。

Seq2Seq 模型的典型训练方案 Teacher Forcing 是一个局部归一化模型，它存在着局部归一化所带来的毛病——也就是我们经常说的“Exposure Bias”。

<img src="/img/article/exposure-bias-in-seq2seq/seq2seq.png" width="550px" style="display: block; margin: auto;"/>

<center>经典的 Seq2Seq 模型图示</center>

本文算是一篇进阶文章，适合对 Seq2Seq 模型已经有一定的了解、希望进一步提升模型的理解或表现的读者。关于 Seq2Seq 的入门文章，可以阅读旧作[《Seq2Seq 模型入门》](/2019/09/08/introduction-to-seq2seq.html)。

本文的内容大致为：

> 1. Exposure Bias 的成因分析及例子；
> 2. 简单可行的缓解 Exposure Bias 问题的策略。

## Exposure Bias 问题

### Softmax

首先，我们来回顾 Softmax 相关内容。大家都知道，对于向量 $(x_1,x_2,\dots,x_n)$，它的 Softmax 为

$$
(p_1,p_2,\dots,p_n)=\frac{1}{\sum\limits_{i=1}^n e^{x_i}}\left(e^{x_1},e^{x_2},\dots,e^{x_n}\right) \tag{1}
$$

由于 $e^t$ 是关于 $t$ 的严格单调递增函数，所以如果 $x_k$ 是 $x_1,x_2,…,x_n$ 中的最大者，那么 $p_k$ 也是 $p_1,p_2,\dots,p_n$ 中的最大者。

对于分类问题，我们所用的 loss 一般是交叉熵，也就是

$$
-\log p_t = \log\left(\sum\limits_{i=1}^n e^{x_i}\right) - x_t \tag{2}
$$

其中 $t$ 是目标类。如文章[《寻求一个光滑的最大值函数》](https://kexue.fm/archives/3290)所述，上式第一项实际上是 $\max\left(x_1,x_2,\dots,x_n\right)$ 的光滑近似，所以为了形象理解交叉熵，我们可以写出

$$
-\log p_t \approx \max\left(x_1,x_2,\dots,x_n\right) - x_t \tag{3}
$$

也就是说，交叉熵实际上在缩小目标类得分 $x_t$ 与全局最大值的差距，显然这个差距最小只能为 0，并且此时目标类得分就是最大值者。所以，Softmax 加交叉熵的效果就是“希望目标类的得分成为最大值”。

### Teacher Forcing

现在，我们来看 Seq2Seq，它通过条件分解来建模联合概率分布：

$$
\begin{aligned}p(\boldsymbol{y}\mid\boldsymbol{x})=&\,p(y_1,y_2,\dots,y_n\mid\boldsymbol{x})\\ 
=&\,p(y_1\mid\boldsymbol{x})p(y_2\mid\boldsymbol{x},y_1)\dots p(y_n\mid\boldsymbol{x},y_1,\dots,y_{n-1}) 
\end{aligned} \tag{4}
$$

每一项自然也就用 Softmax 来建模的，即

$$
\begin{aligned}&p(y_1\mid\boldsymbol{x})=\frac{e^{f(y_1;\boldsymbol{x})}}{\sum\limits_{y_1}e^{f(y_1;\boldsymbol{x})}},\\ 
&p(y_2\mid\boldsymbol{x},y_1)=\frac{e^{f(y_1,y_2;\boldsymbol{x})}}{\sum\limits_{y_2}e^{f(y_1,y_2;\boldsymbol{x})}},\\ 
&\dots,\\ 
&p(y_n\mid\boldsymbol{x},y_1,\dots,y_{n-1})=\frac{e^{f(y_1,y_2,\dots,y_n;\boldsymbol{x})}}{\sum\limits_{y_n}e^{f(y_1,y_2,\dots,y_n;\boldsymbol{x})}} 
\end{aligned} \tag{5}
$$

乘起来就是

$$
p(\boldsymbol{y}\mid\boldsymbol{x})=\frac{e^{f(y_1;\boldsymbol{x})+f(y_1,y_2;\boldsymbol{x})+\dots+f(y_1,y_2,\dots,y_n;\boldsymbol{x})}}{\left(\sum\limits_{y_1}e^{f(y_1;\boldsymbol{x})}\right)\left(\sum\limits_{y_2}e^{f(y_1,y_2;\boldsymbol{x})}\right)\dots\left(\sum\limits_{y_n}e^{f(y_1,y_2,\dots,y_n;\boldsymbol{x})}\right)} \tag{6}
$$

而训练目标就是

$$
\begin{aligned}-\log p(\boldsymbol{y}\mid\boldsymbol{x})=-\log p(y_1\mid\boldsymbol{x})-\log p(y_2\mid\boldsymbol{x},y_1)-\\\dots -\log p(y_n\mid\boldsymbol{x},y_1,\dots,y_{n-1})\end{aligned}\tag{7}
$$

这个直接的训练目标就叫做 Teacher Forcing，因为在算 $-\log p(y_2\mid\boldsymbol{x},y_1)$ 的时候我们要知道真实的 $y_1$，在算 $-\log p(y_3\mid\boldsymbol{x},y_1,y_2)$ 我们需要知道真实的 $y_1,y_2$，依此类推，这就好像有一个经验丰富的老师预先给我们铺好了大部分的路，让我们只需要求下一步即可。这种方法训练起来简单，而且结合 CNN 或 Transformer 那样的模型就可以实现并行的训练，但它可能会带来 Exposure Bias 问题。

### Exposure Bias

其实 Teacher Forcing 这个名称本身就意味着它本身会存在 Exposure Bias 问题。回想一下老师教学生解题的过程，一般的步骤为：

1. 第一步应该怎么思考；

2. 第一步想出来后，第二步我们有哪些选择；

3. 确定了第二步后，第三步我们可以怎么做；

   ...

4. 有了这 $n-1$ 步后，最后一步就不难想到了。

这个过程其实跟 Seq2Seq 的 Teacher Forcing 方案的假设是一样的。有过教学经验的读者就知道，通常来说学生们都能听得频频点头，感觉全都懂了，然后让学生课后自己做题，多数还是一脸懵比。为什么会这样呢？其中一个原因就是 Exposure Bias。说白了，问题就在于，老师总是假设学生能想到前面若干步后，然后教学生下一步，但如果前面有一步想错了或者想不出来呢？这时候这个过程就无法进行下去了，也就是没法得到正确答案了，这就是 Exposure Bias 问题。

### Beam Search

事实上，我们真正做题的时候并不总是这样子，假如我们卡在某步无法确定时，我们就遍历几种选择，然后继续推下去，看后面的结果反过来辅助我们确定前面无法确定的那步。对应到 Seq2Seq 来说，这其实就相当于基于 Beam Search 的解码过程。

对于 Beam Search，我们应该能发现，beam size 并不是越大越好，有些情况甚至是 beam size 等于 1 时最好，这看起来有点不合理，因为 beam size 越大，理论上找到的序列就越接近最优序列，所以应该越有可能正确才对。事实上这也算是 Exposure Bias 的现象之一。

从式 $(6)$ 我们可以看出，Seq2Seq 对目标序列 $y_1,y_2,\dots,y_n$ 的打分函数为：

$$
f(y_1;\boldsymbol{x})+f(y_1,y_2;\boldsymbol{x})+\dots+f(y_1,y_2,\dots,y_n;\boldsymbol{x}) \tag{8}
$$

正常来说，我们希望目标序列是所有候选序列之中分数最高的，根据本文开头介绍的 Softmax 方法，我们建立的概率分布应该是

$$
p(\boldsymbol{y}\mid\boldsymbol{x})=\frac{e^{f(y_1;\boldsymbol{x})+f(y_1,y_2;\boldsymbol{x})+\dots+f(y_1,y_2,\dots,y_n;\boldsymbol{x})}}{\sum\limits_{y_1,y_2,\dots,y_n}e^{f(y_1;\boldsymbol{x})+f(y_1,y_2;\boldsymbol{x})+\dots+f(y_1,y_2,\dots,y_n;\boldsymbol{x})}} \tag{9}
$$

但上式的分母需要遍历所有路径求和，难以实现，而式 $(6)$ 就作为一种折衷的选择得到了广泛应用。但式 $(6)$ 跟式并不 $(9)$ 等价，因此哪怕模型已经成功优化，也可能出现“最优序列并不是目标序列”的现象。

我们来举一个简单例子。设序列长度只有 2，候选序列是 $(a,b)$ 和 $(c,d)$，而目标序列是 $(a,b)$，训练完成后，模型的概率分布情况为

$$
\begin{array}{c|c} 
\hline 
p(a) & p(c)\\ 
\hline 
0.6 & 0.4 \\ 
\hline 
\end{array}\qquad \begin{array}{c|c|c|c} 
\hline 
p(b\mid a) & p(d\mid a) & p(b\mid c) & p(d\mid c)\\ 
\hline 
0.55 & 0.45 & 0.1 & 0.9\\ 
\hline 
\end{array}
$$

如果 beam size 为 1，那么因为 $p(a) > p(c)$，所以第一步只能输出 $a$，接着因为 $p(b\mid a) > p(d\mid a)$，所以第二步只能输出 $b$，成功输出了正确序列 $(a,b)$。但如果 beam size 为 2，那么第一步输出 $(a,0.6),(c,0.4)$，而第二步遍历所有组合，我们得到

$$
\begin{array}{c|c|c|c} 
\hline 
(a, b) & (a, d) & (c, b) & (c, d)\\ 
\hline 
0.33 & 0.27 & 0.04 & 0.36\\ 
\hline 
\end{array}
$$

所以输出了错误的序列 $(c,d)$。

那是因为模型没训练好吗？并不是，前面说过 Softmax 加交叉熵的目的就是让目标的得分最大，对于第一步我们有 $p(a) > p(c)$，所以第一步的训练目标已经达到了，而第二步在 $a$ 已经预先知道的前提下我们有 $p(b\mid a) > p(d\mid a)$，这说明第二步的训练目标也达到了。因此，模型已经算是训练好了，只不过可能因为模型表达能力限制等原因，得分并没有特别高，但“让目标的得分最大”这个目标已经完成了。

## 思考对策

从上述例子中读者或许可以看出问题所在了：主要是 $p(d\mid c)$ 太高了，而 $p(d\mid c)$ 是没有经过训练的，没有任何显式的机制去抑制 $p(d\mid c)$ 变大，因此就出现了“最优序列并不是目标序列”的现象。

看到这里，读者可能就能想到一个朴素的对策了：添加额外的优化目标，降低那些 Beam Search 出来的非目标序列不就行了？事实上，这的确是一个有效的解决方法，相关结果发表在 2016 年的论文[《Sequence-to-Sequence Learning as Beam-Search Optimization》](https://arxiv.org/abs/1606.02960)。但这样一来几乎要求每步训练前的每个样本都要进行一次 Beam Search，计算成本太大。还有一些更新的结果，比如 ACL2019 的最佳长论文[《Bridging the Gap between Training and Inference for Neural Machine Translation》](https://arxiv.org/abs/1906.02448)就是聚焦于解决 Exposure Bias 问题。此外，通过强化学习直接优化 BLEU 等方法，也能一定程度上缓解 Exposure Bias。

然而，据笔者所了解，这些致力于解决 Exposure Bias 的方法，大部分都是大刀阔斧地改动了训练过程，甚至会牺牲原来模型的训练并行性（需要递归地采样负样本，如果模型本身是 RNN 那倒无妨，但如果本身是 CNN 或 Transformer，那伤害就很大了），成本的提升幅度比效果的提升幅度大得多。

### 构建负样本

纵观大部分解决 Exposure Bias 的论文，以及结合我们前面的例子和体会，不难想到，其主要思想就是构造有代表性的负样本，然后在训练过程中降低这些负样本的概率，所以问题就是如何构造“有代表性”的负样本了。这里给出笔者构思的一种简单策略，实验证明它能一定程度上缓解 Exposure Bias，提升文本生成的表现，重要的是，这种策略比较简单，基本能做到即插即用，几乎不损失训练性能。

方法很简单，就是随机替换一下 Decoder 的输入词（Decoder 的输入词有个专门的名字，叫做 oracle words），如下图所示：

<img src="/img/article/exposure-bias-in-seq2seq/strategies_to_mitigate_exposure_bias.png" width="560px" style="display: block; margin: auto;"/>

<center>一种缓解 Exposure Bias 的简单策略：直接将 Decoder 的部分输入词随机替换为别的词。</center>

其中紫色的 `[R]` 代表被随机替换的词。其实不少 Exposure Bias 的论文也是这个思路，只不过随机选词的方案不一样。笔者提出的方案很简单：

1. 50% 的概率不做改变；
2. 50% 的概率把输入序列中 30% 的词替换掉，替换对象为原目标序列的任意一个词。

也就是说，随机替换发生概率是 50%，随机替换的比例是 30%，随机抽取空间就是目标序列的词集。这个策略的灵感在于：尽管 Seq2Seq 不一定能完全生成目标序列，但它通常能生成大部分目标序列的词（但顺序可能不对，或者重复出现同一些词），因此这样替换后的输入序列通常可以作为有代表性的负样本。对了，说明一下，50% 和 30% 这两个比例纯粹是拍脑袋的，没仔细调参，因为生成模型调一次实在是太累了。

效果如何呢？笔者做了两个标题（摘要）生成的实验（就是 [CLGE](https://github.com/CLUEbenchmark/CLGE) 的前两个），其中 baseline 是 [task_seq2seq_autotitle_csl.py](https://github.com/bojone/bert4keras/blob/master/examples/task_seq2seq_autotitle_csl.py)，代码开源于：

> **Github地址：**[https://github.com/bojone/exposure_bias](https://github.com/bojone/exposure_bias)

结果如下表：

$$
\begin{array}{c} 
\text{CSL标题生成实验结果}\\ 
{\begin{array}{c|c|cccc} 
\hline 
& \text{beam size} & \text{Rouge-L} & \text{Rouge-1} & \text{Rouge-2} & \text{BLEU} \\ 
\hline 
\text{baseline} & 1 & 59.27 & 63.45 & 51.12 & 41.31 \\ 
\text{随机替换} & 1 & \textbf{60.02} & \textbf{64.35} & \textbf{51.87} & \textbf{41.99} \\ 
\hline 
\text{baseline} & 2 & 60.06 & 64.19 & 52.02 & 42.3 \\ 
\text{随机替换} & 2 & \textbf{60.99} & \textbf{65.15} & \textbf{52.89} & \textbf{43.24} \\ 
\hline 
\text{baseline} & 3 & 60.31 & 64.47 & 52.21 & 42.46 \\ 
\text{随机替换} & 3 & \textbf{61.08} & \textbf{65.19} & \textbf{53.03} & \textbf{43.45} \\ 
\hline 
\end{array}}\\ 
\\ 
\text{LCSTS摘要生成实验结果}\\ 
{\begin{array}{c|c|cccc} 
\hline 
& \text{beam size} & \text{Rouge-L} & \text{Rouge-1} & \text{Rouge-2} & \text{BLEU} \\ 
\hline 
\text{baseline} & 1 & 26.97 & 30.79 & \textbf{18.14} & \textbf{11.41} \\ 
\text{随机替换} & 1 & \textbf{27.4} & \textbf{31.48} & 17.89 & 10.9 \\ 
\hline 
\text{baseline} & 2 & 27.7 & 31.62 & \textbf{18.93} & \textbf{11.97} \\ 
\text{随机替换} & 2 & \textbf{28.21} & \textbf{32.28} & 18.88 & 11.6 \\ 
\hline 
\text{baseline} & 3 & 27.87 & 31.73 & \textbf{19.22} & \textbf{12.28} \\ 
\text{随机替换} & 3 & \textbf{28.34} & \textbf{32.42} & 19.06 & 11.77 \\ 
\hline 
\end{array}} 
\end{array}
$$

可以发现，在 CSL 任务中，基于随机替换的策略稳定提升了文本生成的所有指标，而 LCSTS 任务的各个指标则各有优劣，考虑到 LCSTS 本身比较难，各项指标本来就低，所以应该说 CSL 的结果更有说服力一些。这表明，笔者提出的上述策略确实是一种值得尝试的方案。（注：所有实验都重复了两次然后取平均，所以实验结果应该是比较可靠的了）

### 对抗训练

思考到这里，我们不妨再“天马行空”一下：既然解决 Exposure Bias 的思路之一就是要构造有代表性的负样本输入，说白了就是让模型在扰动下依然能预测正确，而[对抗训练](/2020/06/16/talking-about-generalization.html)正是一种生成扰动样本的方法。如果直接往 baseline 模型里边加入对抗训练，能不能提升模型的性能呢？简单起见，笔者做了往 baseline 模型里边梯度惩罚（也算是对抗训练的一种）的实验，结果对比如下：

$$
\begin{array}{c} 
\text{CSL标题生成实验结果}\\ 
{\begin{array}{c|c|cccc} 
\hline 
& \text{beam size} & \text{Rouge-L} & \text{Rouge-1} & \text{Rouge-2} & \text{BLEU} \\ 
\hline 
\text{baseline} & 1 & 59.27 & 63.45 & 51.12 & 41.31 \\ 
\text{随机替换} & 1 & 60.02 & 64.35 & 51.87 & 41.99 \\ 
\text{梯度惩罚} & 1 & \textbf{60.79} & \textbf{64.91} & \textbf{52.54} & \textbf{42.81} \\ 
\hline 
\text{baseline} & 2 & 60.06 & 64.19 & 52.02 & 42.3 \\ 
\text{随机替换} & 2 & 60.99 & 65.15 & 52.89 & 43.24 \\ 
\text{梯度惩罚} & 2 & \textbf{61.37} & \textbf{65.53} & \textbf{53.29} & \textbf{43.69} \\ 
\hline 
\text{baseline} & 3 & 60.31 & 64.47 & 52.21 & 42.46 \\ 
\text{随机替换} & 3 & 61.08 & 65.19 & 53.03 & 43.45 \\ 
\text{梯度惩罚} & 3 & \textbf{61.47} & \textbf{65.6} & \textbf{53.42} & \textbf{43.82} \\ 
\hline 
\end{array}}\\ 
\\ 
\text{LCSTS摘要生成实验结果}\\ 
{\begin{array}{c|c|cccc} 
\hline 
& \text{beam size} & \text{Rouge-L} & \text{Rouge-1} & \text{Rouge-2} & \text{BLEU} \\ 
\hline 
\text{baseline} & 1 & 26.97 & 30.79 & 18.14 & \textbf{11.41} \\ 
\text{随机替换} & 1 & 27.4 & 31.48 & 17.89 & 10.9 \\ 
\text{梯度惩罚} & 1 & \textbf{28.02} & \textbf{32.29} & \textbf{18.66} & 10.86 \\ 
\hline 
\text{baseline} & 2 & 27.7 & 31.62 & 18.93 & \textbf{11.97} \\ 
\text{随机替换} & 2 & \textbf{28.21} & 32.28 & 18.88 & 11.6 \\ 
\text{梯度惩罚} & 2 & 28.12 & \textbf{32.31} & \textbf{19.27} & 11.56 \\ 
\hline 
\text{baseline} & 3 & 27.87 & 31.73 & \textbf{19.22} & \textbf{12.28} \\ 
\text{随机替换} & 3 & \textbf{28.34} & \textbf{32.42} & 19.06 & 11.77 \\ 
\text{梯度惩罚} & 3 & 27.79 & 31.98 & 19.18 & 11.44 \\ 
\hline 
\end{array}} 
\end{array}
$$

可以看到，对抗训练（梯度惩罚）进一步提升了 CSL 生成的所有指标，而 LCSTS 上则同样比较“随缘”。因此，对抗训练也可以列入“提升文本生成模型的潜力技巧”名单之中。

> 转载自[《Seq2Seq中Exposure Bias现象的浅析与对策》](https://kexue.fm/archives/7259)，作者：苏剑林，部分内容有修改。

