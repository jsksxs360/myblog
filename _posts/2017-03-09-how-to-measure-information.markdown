---
layout: article
title: 信息的度量和作用：信息论基本概念
author: 吴军
tags:
    - NLP
mathjax: true
---

> 本文摘自吴军《数学之美》，部分内容有修改。

我们在生活中一直谈论信息，但是信息这个概念依然有些抽象。我们经常说信息很多，或者信息较少，但却很难说清楚信息到底有多少。比如，一本 50 多万字的《史记》到底有多少的信息量？我们也常说信息有用，那么它的作用又是如何客观、定量地体现出来的呢？

对于这两个问题，几千年来都没有人给出很好的解答。直到 1948 年，香农(Claude Elwood Shannon)在他著名的论文《通信的数学原理》(A Mathematic Theory of Communication)中提出了“信息熵”的概念，才解决了信息的度量问题，并且量化出信息的作用。

## 信息熵

一条信息的信息量与其不确定性有着直接的关系。比如说，我们要搞清楚一件非常非常不确定的事，或是我们一无所知的事情，就需要了解大量的信息。相反，如果已对某件事了解较多，则不需要太多的信息就能把它搞清楚。所以从这个角度来看，可以认为，信息量就等于不确定性的多少。

那么如何量化信息量的度量呢？来看一个例子：2014 年举行了世界杯足球赛，大家都很关心谁会是冠军。假如我错过了看世界杯，赛后我问一个知道比赛结果的观众“哪只球队是冠军”？他不愿意直接告诉我，而让我猜，并且我每猜一次，他要收我一元钱才肯告诉我是否猜对了，那么我要掏多少钱才能知道谁是冠军呢？我可以把球队编上号，从 1 到 32，然后提问：“冠军在 1~16 号中吗？”假如他告诉我猜对了，我会接着提问：“冠军在 1~8 号中吗？”假如他告诉我猜错了，我自然知道冠军在 9~16 号中。这样我只需要 5 次，就能知道哪只球队是冠军。所以，谁是世界杯冠军这条消息的信息量只值 5 块钱。

当然，香农不是用钱，而是用比特(Bit)这个概念来度量信息量。一个比特是一位二进制数，在计算机中，一个字节就是 8 比特。在上面的例子中，这条消息的信息量是 5 比特。（如果有朝一日有 64 支球队进入决赛，那么“谁是世界杯冠军”的信息量就是 6 比特，因为要多猜一次）。我们发现，信息量的比特数和所有可能情况的对数函数 $\log$ 有关。（$\log32=5$，$\log64=5$）。

但是实际上可能不需要猜 5 次就能猜出谁是冠军，因为像西班牙、巴西、德国、意大利这样的球队夺冠的可能性比日本、南非、韩国等球队大得多。因此，第一次猜测是不需要把 32 支球队等分成两个组，而可以把少数几只最可能的球队分成一组，把其他球队分成另一组。然后猜冠军是否在那几支热门队中。重复这样的过程，根据夺冠概率对余下候选球队分组，直至找到冠军。这样也许 3 次或 4 次就能猜出结果。因此当每只球队夺冠的可能性（概率）不等时，“谁是世界杯冠军”的信息量比 5 比特少。香农指出，它的准确信息量应该是：

$$
H=-(p_1\cdot\log p_1+p_2\cdot\log p_2+\cdots+p_{32}\cdot\log p_{32})
$$

其中 $p_1,p_2,\cdots,p_{32}$ 分别是这 32 支球队夺冠的概率。香农把它称为信息熵(Entropy)，一般用符号 $H$ 表示，单位比特。很容易计算出，当 32 支球队夺冠概率相同时，对应的信息熵等于 5 比特。

### 信息熵的定义

对于任意一个离散型随机变量 $X$，它的熵 $H(X)$ 定义为：

$$
H(X)=-\sum_{x\in X}p(x)\log p(x)
$$

其中，约定 $0\log0=0$。$H(X)$ 可以写为 $H(p)$。

变量的不确定性越大，熵也就越大，要把它搞清楚，所需信息量也就越大。信息熵的物理含义是对一个信息系统不确定性的度量，这一点上和热力学中熵的概念有相似之处，后者就是一个系统无序的度量，从另一角度讲也是对一种不确定性的度量。

> 常用汉字大约有 7000 字，假如每个字等概率，那么大约需要 13 比特表示一个汉字。但汉字的使用频率不是均等的。实际上，前 10% 的汉字占常用文本的 95% 以上。因此，即使不考虑上下文的相关性，而只考虑每个汉字的独立概率，那么每个汉字的信息熵大约也只有 8~9 比特，如果再考虑上下文相关性，每个汉字的信息熵大约就只有 5 比特左右了。
>
> 所以，一本 50 万字的中文书，信息量大约是 250 万比特。采用较好的算法进行压缩，整本书可以存成一个 320 KB 的文件。如果直接用两字节的国际编码存储这本书，大约需要 1 MB 大小，是压缩文件的三倍。
>
> 这两个数量的差距，在信息论中称作“冗余度”(Redundancy)。注意，上面讲的 250 万比特是个平均数，同样长度的书，所含的信息量可以相差很多。如果一本书重复的内容很多，它的信息量就小，冗余度就大。
>
> 不同语言的冗余度差别很大，而汉语在所有语言中冗余度是相对小的。一本英文书，翻译成汉语，如果字体大小相同，那么中译本一般都会薄很多。这和人们普遍的认识——汉语是最简洁的语言——是一致的。

### 最大熵原理

“最大熵”这个名词听起来很深奥，但是它的原理很简单，我们每天都在用。说白了，就是要保留全部的不确定性，将风险降到最小。

比如我们拿到了一个骰子，现在要问“每个面朝上的概率分别是多少”。在没有其他信息的情况下，我们会假设每种点数的概率都是 1/6。这种猜测当然是对的，因为对于一个我们“一无所知”的骰子，假定它每一面朝上的概率均等是最安全的做法。从投资角度看，这就是风险最小的做法。从信息论角度来讲，这就是保留了最大的不确定性，也就是让熵达到最大。现在假设我们已知四点朝上的概率是 1/3，这种情况下，每个面朝上的概率是多少？通常我们会认为其余所有面朝上的概率都是 2/15，也就是认为所有未知的面朝上的概率相等。

可以看到，在猜测这两种不同情况下的概率分布时，我们都没有添加任何主观的假设，这种基于直觉的猜测之所以准确，是因为它恰好符合了最大熵原理。

最大熵原理指出，对一个随机事件的概率分布进行预测时，我们的预测应当满足全部已知的条件，而对未知的情况不要做任何主观假设。在这种情况下，概率分布最均匀，预测的风险最小。因为这时概率分布的信息熵最大，所以人们把这种模型称作“最大熵模型”。我们常说的，不要把所有的鸡蛋放在一个篮子里，其实就是最大熵原理的一个朴素的说法，因为当我们遇到不确定时，就要保留各种可能性。

最大熵原理被广泛地应用于自然语言处理中，通常的做法是，根据已知样本设计特征函数，假设存在 $k$ 个特征函数 $f_i(i=1,2,\cdots,k)$，它们都在建模过程中对输出有影响，那么，所建立的模型应满足所有这些特征的约束，即所建立的模型 $p$ 应该属于这 $k$ 个特征函数约束下所产生的所有模型的集合 $C$。使熵 $H(p)$ 值最大的模型用来推断某种语言现象存在的可能性，或者作为进行某种处理操作的可靠性依据，即：

$$
\hat{p}=\arg\max_{p\in C}H(p)
$$

## 条件熵

自古以来，信息和消除不确定性是相联系的。在英语里，信息和情报是同一个词（Infomation），而我们知道情报的作用就是排除不确定性。

一个事物内部会存有随机性，也就是不确定性，假定为 $U$，而从外部消除这个不确定性的唯一的方法是引入信息 $I$，而引入的信息量取决于这个不确定性的大小，即 $I\gt U$ 才行。当 $I \lt U$ 时，这些信息可以消除一部分不确定性，也就是说新的不确定性：

$$
U'=U-I
$$

反之，如果没有信息，任何公式或者数学的游戏都无法排除不确定性。几乎所有的自然语言处理、信息与信号处理的应用都是一个消除不确定性的过程。

自然语言的统计模型，其中的一元模型就是通过某个词本身的概率分布，来消除不确定性；二元及更高阶的语言模型则还使用了上下文的信息，那就能准确预测一个句子中当前的词汇了。在数学上可以严格地证明为什么这些“相关的”信息也能够消除不确定性。为此，需要引入条件熵(Conditional Entropy)的概念。

### 条件熵的定义

假定 $X$ 和 $Y$ 是两个随机变量，$X$ 是我们需要了解的。假定我们现在知道了 $X$ 的随机分布 $P(X)$，那么也就知道了 $X$ 的熵：

$$
H(X)=-\sum_{x\in X}P(x)\cdot\log P(x)
$$

那么它的不确定性就是这么大。现在假定我们还知道 $Y$ 的一些情况，包括它和 $X$ 一起出现的概率，在数学上称为联合概率分布(Joint Probability)，以及在 $Y$ 取不同值的前提下 $X$ 的概率分布，在数学上称为条件概率分布(Conditional Probability)。定义在 $Y$ 条件下 $X$ 的条件熵为：

$$
\begin{align}H(X \mid Y) &=  \sum_{y\in Y}P(y)H(X \mid Y=y) \\&=\sum_{y\in Y}P(y)\big[-\sum_{x\in X}P(x\mid y)\log P(x\mid y)\big]\\&=-\sum_{x\in X,y \in Y}P(x,y)\log P(x\mid y)\end{align}
$$

可以证明 $H(X)\ge H(X\mid Y)$，也就是说，多了 $Y$ 的信息之后，关于 $X$ 的不确定性下降了！在统计语言模型中，如果把 $Y$ 看成是前一个字，那么在数学上就证明了二元模型的不确定性小于一元模型。同理，可以定义有两个条件的条件熵：

$$
H(X\mid Y,Z)=-\sum_{x\in X,y\in Y,z\in Z}P(x,y,z)\log P(x\mid y,z)
$$

还可以证明 $H(X\mid Y)\ge H(X\mid Y,Z)$。也就说，三元模型应该比二元的好。

### 一个有意思的问题

思考一下，上述式子中的等号什么时候成立？等号成立说明增加了信息，不确定性却没有降低，这可能吗？

答案是肯定的，如果我们获取的信息与要研究的事物毫无关系，等号就成立。那么如何判断事物之间是否存在关系，或者说我们应该如何去度量两个事物之间的关联性呢？这自然而然就引出了互信息(Mutual Infomation)的概念。

## 互信息

我们在上一节中提到，当获取的信息和要研究的事物“有关系”时，这些信息才能帮助我们消除不确定性。当然“有关系”这种说法太模糊，太不科学，最好能够量化地度量“相关性”。为此，香农在信息论中提出了一个互信息(Mutual Infomation)的概念作为两个随机事件“相关性”的量化度量。

### 定义

假定有两个随机事件 $X$ 和 $Y$，它们之间的互信息定义如下：

$$
I(X;Y)=\sum_{x\in X,y\in Y}P(x,y)\log\frac{P(x,y)}{P(x)P(y)}
$$

可以证明，互信息 $I(X;Y)$ 就是随机事件 $X$ 的不确定性（熵 $H(X)$）与在知道随机事件 $Y$ 条件下 $X$ 的不确定性（条件熵 $H(X\mid Y)$）之间的差异，即：

$$
I(X;Y)=H(X)-H(X\mid Y)
$$

所谓两个事件相关性的量化度量，就是在了解了其中一个 $Y$ 的前提下，对消除另一个 $X$ 不确定性所提供的信息量。互信息是一个取值在 0 到 $\min(H(X),H(Y))$ 之间的函数，当 $X$ 和 $Y$ 完全相关时，它的取值是 1；当二者完全无关时，它的取值是 0。

### 词义二义性问题

在自然语言中，两个随机事件，或者语言特征的互信息是很容易计算的。只要有足够的语料，就不难估计出互信息公式中的 $P(X,Y)$、$P(X)$ 和 $P(Y)$，进而算出互信息。因此互信息被广泛用于度量一些语言现象的相关性。

机器翻译中，最难的两个问题之一是词义的二义性（又称歧义性，Ambiguation）问题。比如 Bush 一词可以是美国总统布什的名字，也可以是灌木丛。那么如何正确地翻译这些词呢？人们很容易想到要用语法，分析语句，等等。

其实，迄今为止，没有一种语法能很好地解决这个问题，因为 Bush 不论翻译成人名还是灌木丛，都是名词，在语法上没有太大问题。当然有人可能会想到，可以加一条规则“总统做宾语时，主语得是一个人”，要是这样，语法规则就多得数不清了，而且还有很多例外，比如一个国家在国际组织中也可以做主席（总统）的轮值国。

其实，真正简单却非常实用的方法是使用互信息。具体的解决方法大致如下：首先从大量文本中找出和总统布什一起出现的互信息最大的词，比如总统、美国、国会、华盛顿等等；再用同样的方法找出和灌木丛一起出现的互信息最大的词，比如土壤、植物、野生等等。有了这两组词，在翻译 Bush 时，看看上下文中哪些相关的词多就可以了。这种方法最初是由吉尔(William Gale)、丘奇(Kenneth Church)和雅让斯基(David Yarowsky)提出的。

> 20 世纪 90 年代初，雅让斯基是宾夕法尼亚大学自然语言处理大师马库斯(Mitch Marcus)教授的博士生，他很多时候都泡在贝尔实验室丘奇等人的研究室里。
>
> 也许是急于毕业，他在吉尔等人的帮助下想出了一个最快也是最好地解决翻译中的二义性的方法，就是上面这个看似简单的方法，效果却好得让同行们大吃一惊。雅让斯基因而只用了三年就从马库斯那里拿到了博士，而他的师兄弟们平均要花六年时间。

## 相对熵

前面已经介绍了信息熵和互信息，它们是信息论的基础，而信息论则在自然语言处理中扮演着指导性的角色。下面我们将介绍信息论中的另一个重要的概念——相对熵(Relative Entropy，或 Kullback-Leibler Divergence)。

相对熵也用来衡量相关性，但和变量的互信息不同，它用来衡量两个取值为正数的函数的相似性。它的定义如下：

$$
KL(f(x) || g(x))=\sum_{x\in X}f(x)\cdot\log \frac{f(x)}{g(x)}
$$

大家不必关心公式本身，只需记住下面三条结论就好：

1. 对于两个完全相同的函数，它们的相对熵等于零。
2. 相对熵越大，两个函数差异越大；反之，相对熵越小，两个函数差异越小。
3. 对于概率分布或者概率密度函数，如果取值均大于零，相对熵可以度量两个随机分布的差异性。

注意，相对熵是不对称的，即：

$$
KL(f(x)||g(x))\ne KL(g(x)||f(x))
$$

这样使用起来有时不是很方便，为了让它对称，詹森和香农提出一种新的相对熵的计算方法，将上面的不等式两边取平均，即

$$
JS(f(x)||g(x))=\frac{1}{2}\big[KL(f(x)||g(x))+KL(g(x)||f(x))\big]
$$

相对熵最早是用在信号处理上。如果两个随机信号，它们的相对熵越小，说明这两个信号越接近，否则信号的差异越大。后来研究信息处理的学者们也用它来衡量两段信息的相似程度，比如如果一篇文章是照抄或者改写另一篇，那么这两篇文章中词频分布的相对熵就非常小，接近于零。

相对熵在自然语言处理中还有很多应用，比如用来衡量两个常用词（在语法和语义上）在不同文本中的概率分布，看它们是否同义。另外，利用相对熵，还可以得到信息检索中最重要的一个概念：词频-逆向文档频率(TF-IDF)。

## 参考

吴军《数学之美》  
宗成庆《统计自然语言处理》
