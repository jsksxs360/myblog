---
layout: article
title: "LSTM 网络原理：通过图解，一步一步“走过”LSTM"
author: colah
tags:
    - NLP
mathjax: true
---

> 译自[《Understanding LSTM Networks》](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，作者 [colah](http://colah.github.io)。部分内容有删改。

## 循环神经网络

人类不是在每一个时刻重新开始思考的。当你阅读这篇文章时，你会根据你对以前的话语的理解来理解每个单词。你不可能把所有东西都扔掉，再从头开始思考。即你的思想具有持续性。

但是传统的神经网络就不能做到这一点，这似乎是一个很大的缺陷。例如，假设你想要判断一部电影中每个时间点发生的事件是什么类型。传统的神经网络是无法根据电影中前面出现的事件来推理后面出现的事件的。

循环神经网络解决了这个问题。它们是具有循环的网络，允许信息的持续存在。

![1](/img/article/what-is-lstm/rnn.png)

在上图中，一组神经网络 $A$，输入一些 $x_t$，并且输出一个值 $h_t$。循环的结构允许信息从网络的一个步骤传递到下一个。

这些循环使得循环神经网络看上去有些神秘。但是，如果你再深入的思考一下，就会发现它们并非完全与普通的神经网络不同。一个循环神经网络可以被看作是很多个同一个网络的副本，并且每一个副本都会向下一个传递信息。考虑一下，如果我们展开循环会发生什么：

![2](/img/article/what-is-lstm/rnn_2.png)

这种链状的特征展示了循环神经网络与序列和列表的密切关系。它们是用于分析此类数据的自然的神经网络结构。

而且它们肯定是用的！在过去的几年里，RNN 在各种问题上取得了巨大的成功：语音识别、语言建模、翻译、图像字幕……我将不会讨论 Andrej Karpathy 优秀的博客文章[《循环神经网络的没有理由的有效性》](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)中描述的 RNN 可以实现惊人的壮举，但它们真的很神奇。

这些成功的基础是使用“LSTM”，这是一种非常特殊的循环神经网络，对于许多任务来说，它比标准的神经网络要有用多了。几乎所有基于循环神经网络的令人振奋的结果都是通过它们实现的。这篇文章将探讨这些 LSTM。

## 长程依赖的问题

RNN 吸引人的特点之一，就是认为它们可以将先前的信息连接到当前的任务，例如使用先前的视频帧或许可以帮助理解当前视频帧的意义。如果 RNN 可以做到这一点，它们将非常有用。但它们真的可以吗？这就要视情况而定了。

有时，我们只需要查看最近的信息来执行当前的任务。例如，考虑一种语言模型，尝试基于前面的单词来预测下一个单词。例如我们试图预测“the clouds are in the *sky*”中的最后一个词，我们就不需要任何其他的上下文信息——很明显下一个词将是 sky。在这种情况下，如果相关信息与需要信息的位置之间的距离很小，RNN 就可以学习使用过去的信息。

![3](/img/article/what-is-lstm/rnn_3.png)

但有时我们需要更多的上下文信息。考虑尝试预测文本“I grew up in France… I speak fluent *French*.”中的最后一个单词。邻近的信息表明，下一个词语可能是一种语言，但是如果我们想确定是哪种语言，那么我们就需要往后看很远距离，以获得上下文信息“法国”。相关信息和需要信息的位置之间的距离可能非常大。

不幸的是，随着距离的扩大，RNN 无法学会去连接这些信息。

![4](/img/article/what-is-lstm/rnn_4.png)

理论上来说，RNN 绝对有能力处理这样的“长程依赖”。人们可以仔细地为 RNN 挑选参数来解决这种形式的小问题。不幸的是，在实践中，RNN 似乎没有办法学习到这些。[Hochreiter (1991) German](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) 和 [Bengio, et al. (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf) 深入探讨了这个问题，并且发现了一些造成困难的基本原因。

幸运的是，LSTM 没有这个问题！

## LSTM 网络

长短期记忆网络(Long Short Term Memory networks)——通常简称为“LSTM”——是一种特殊的 RNN，它能够学习长程的依赖。LSTM 最早由 [Hochreiter & Schmidhuber (1997)](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) 提出，并且很多人在接下来的工作中对它们进行了改进和普及[1]。LSTM 在各种各样的问题上都工作得很好，现在被广泛使用。

LSTM 就是专门为避免长程依赖问题而设计的。记住长时间的信息实际上是它们的默认行为，而不是它们难以学习的东西！

所有循环神经网络都具有重复神经网络模块链的形式。在标准 RNN 中，该重复模块具有非常简单的结构，例如单个 tanh 层。

![5](/img/article/what-is-lstm/lstm.png)

LSTM 也具有这样的链结构，但是重复模块具有不同的结构。它们不是只有一个单一的神经网络层，而是有四个，并且层之间以非常特殊的方式进行交互。

![6](/img/article/what-is-lstm/lstm_2.png)

不要担心发生了什么的细节。我们将逐步介绍 LSTM 的图示。现在，让我们试着熟悉一下我们将要使用的符号。

![7](/img/article/what-is-lstm/symbol.png)

在上图中，每一行都携带了从一个节点的输出到其他节点的输入的整个向量。粉色圆圈表示点运算，如向量加法，而黄色框是学习到的神经网络层。行合并表示连接(concatenation)，而行分叉表示其内容正在复制，副本将传到不同的位置。

## LSTM 背后的核心思想

LSTM 的关键是单元状态(cell state)，即穿过图顶部的水平线。单元状态类似于一个输送带。它直接在整个链上运行，只有一些小的线性相互作用。信息能够很容易保持不变地流动下去。

![8](/img/article/what-is-lstm/lstm_3.png)

LSTM 确实具有删除或添加信息到单元状态的能力，由被称为门的结构仔细调节。门是一种可选择地让信息通过的方式。它们由一个 sigmoid 神经网络层和点乘运算组成。

![9](/img/article/what-is-lstm/gate.png)

sigmoid 层输出零和一之间的数字，描述每个组件能够通过的程度。值为零表示“不让任何东西通过”，值为一意味着“让一切通过！”

LSTM 有三个门，用于保护和控制单元状态。

## 一步一步“走过”LSTM

LSTM 的第一步就是决定我们要从单元状态中丢弃什么信息。这个决定由一个 sigmoid 层做出，称为“遗忘门层(forget gate layer)”。它查看 $h_{t-1}$ 和 $x_t$，并且为单元状态 $C_{t-1}$ 中的每个数字输出一个 $0$ 和 $1$ 之间的数字。$1$ 代表“完全保持这个数字”，而 $0$ 表示“完全遗忘”。

回到我们语言模型的例子，试图根据所有以前的单词来预测下一个单词。在这样的问题中，单元状态可能包括当前主题(subject)的性质，从而可以使用正确的代词。当我们看到一个新主题时，我们想要忘记老主题的性质。

![10](/img/article/what-is-lstm/forget_gate.png)

下一步是决定我们要在单元状态下存储的新信息。这包含两部分。首先，一个称为“输入门层(input gate layer)”的 sigmoid 层决定我们将更新哪些值。然后，一个 tanh 层创建一个可以被添加到状态中的新候选值向量 $\tilde{C}_t$。在下一步中，结合这两个门来创建对状态的更新。

![11](/img/article/what-is-lstm/input_gate.png)

现在是将旧的单元状态 $C_{t-1}$ 更新为新的单元状态 $C_t$ 的时候了。以前的步骤已经决定了要做什么，我们只需要实际去做就行了。

我们将旧的状态乘以 $f_t$，去忘记那些我们之前决定忘记的东西。然后我们向状态中添加 $i_t * \tilde{C}_t$。这是新的候选值，并且按照我们决定更新每个状态值的程度来缩放。

对于语言模型来说，这就是我们实际删除旧主题性质的信息，并添加新信息的地方，正如我们在之前的步骤中决定的那样。

![12](/img/article/what-is-lstm/cell.png)

最后，我们需要决定输出的信息。此输出将基于我们的单元状态，但将是一个过滤后的版本。首先，我们运行一个 sigmoid 层，它决定了我们要输出单元状态的哪些部分。然后，我们让单元状态通过 tanh（将值变为 $-1$ 和 $1$ 之间），并将其乘以 sigmoid 门的输出，以便我们只输出我们决定输出的部分。

以语言模型为例，由于它只看到一个主题，如果后续是一个动词，它可能需要输出与动词相关的信息。例如，它可能会输出主题是单数还是复数，以便我们知道接下来的动词应该是什么形式。

![13](/img/article/what-is-lstm/lstm_output.png)

## LSTM 的变种

到目前为止我所描述的是一个很普通的 LSTM。但并不是所有的 LSTM 都与上面描述的相同。事实上，似乎几乎每一篇涉及 LSTM 的论文都使用了一个略有不同的版本。虽然差异很小，但它们中的一些还是值得一提的。

一个流行的由 [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf) 提出的 LSTM 变种，添加了“窥视孔连接(peephole connections)”。这意味着我们让门层可以看到单元状态。

![14](/img/article/what-is-lstm/lstm_4.png)

上图为所有的门添加了窥视，但许多论文只会给其中的一些门添加窥视。

另一个变种是使用耦合的忘记和输入门。它不是分开决定要忘记和添加的信息，而是一起做出这些决定。只有我们要向某个位置输入一些东西的时候我们才会忘记它。我们只会向某个忘记了旧东西的状态输入新的值。

![15](/img/article/what-is-lstm/lstm_5.png)

LSTM 的一个稍微更显着的变种是由 [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf) 提出的门控循环单元或者称为 GRU。它将忘记门和输入门组合成一个单一的“更新门”，它还合并了单元状态和隐藏状态，并进行了一些其他更改。所得到的模型比标准 LSTM 模型更简单，并且越来越受欢迎。

![16](/img/article/what-is-lstm/lstm_6.png)

这些只是最显着的 LSTM 变种中很少的一部分。还有很多其他的，像 [Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf) 提出的深度门控 RNN (Depth Gated RNNs)。还有一些完全不同的处理长期依赖的方法，例如 [Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf) 提出的发条装置 RNN (Clockwork RNNs)。

这些变种中的哪一个最好？变种的不同之处是否有用？[Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf) 对流行的变种做了一个很好的比较，发现他们都是一样的。[Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) 测试了一万多种 RNN 结构，发现某些结构在某些任务上比 LSTM 更好。

## 结论

此前，我提到人们通过 RNN 实现了显着的成果。基本上所有这些都是使用 LSTM 实现的。而且它们的确在大多数任务上做得更好！

写成一组方程式，LSTM 看起来很吓人。希望在这篇文章中一步步走过 LSTM，使他们变得更加平易近人。

LSTM 是我们可以用 RNN 完成的一大步。我们自然而然就会想：还有另一大步吗？研究人员的共同观点是：“是的！有一个下一步，它就是注意力机制(attention)！”这个想法是让 RNN 的每个步骤从一些较大的信息集合中选择信息。例如，如果您使用 RNN 创建描述图像的标题，则可能会选择图像的一部分来查看其输出的每个词。事实上，[Xu, *et al.*(2015)](http://arxiv.org/pdf/1502.03044v2.pdf) 就是这么做的——如果你想要去探索注意力机制，这可能是一个有趣的起点！逐步开始有一些使用注意力机制的真正令人兴奋的结果，但似乎更多的依然在角落...

注意力机制不是 RNN 研究中唯一令人兴奋的线索。例如，[Kalchbrenner, *et al.* (2015)](http://arxiv.org/pdf/1507.01526v1.pdf) 提出的网格 LSTM (Grid LSTMs) 看上去非常有希望。在生成模型中使用 RNN 进行工作——例如 [Gregor, *et al.* (2015)](http://arxiv.org/pdf/1502.04623.pdf)、[Chung, *et al.* (2015)](http://arxiv.org/pdf/1506.02216v3.pdf) 或 [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf)——也似乎很有趣。最近的几年对于循环神经网络来说已经是一个激动人心的时刻了，而且后来的这些神经网络只会带来更多的惊喜！

[1] 除了原作者，很多人对现代 LSTM 做出了贡献。一个不全面的列表是：Felix Gers, Fred Cummins, Santiago Fernandez, Justin Bayer, Daan Wierstra, Julian Togelius, Faustino Gomez, Matteo Gagliolo, and [Alex Graves](https://scholar.google.com/citations?user=DaFHynwAAAAJ&hl=en)。

> 译自[《Understanding LSTM Networks》](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，作者 [colah](http://colah.github.io)。部分内容有删改。