---
layout: article
title: Transformer 模型通俗小传
tags:
    - NLP
mathjax: true
---

近年来，在自然语言处理 (NLP) 领域，Transformer 已经替代了循环神经网络 (RNN)、卷积神经网络 (CNN) 等模型，成为了深度学习模型的标配。

如题目所示，本文并非是对 Transformer 模型原理进行介绍的专业文章，而是专注于概述 Transformer 模型的定义及发展。你可以将本文看作是一份地图，在阅读后再根据自己的实际需要继续深入了解。

## 起源与发展

2017 年，Google 的研究者在[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)中提出了一个名为 Transformer 的神经网络模型用于序列标注，该模型在翻译任务上超过了之前最优秀的循环神经网络模型，不仅翻译质量更高，而且训练成本更低。与此同时，Fast AI 的研究者在[《Universal Language Model Fine-tuning for Text Classification》](https://arxiv.org/abs/1801.06146)中提出了一种名为 ULMFiT 的迁移学习方法，它首先在一个巨大且内容丰富的数据集上训练长短时记忆网络 (LSTM)，然后将模型迁移用于文本分类任务，最后证明只需要很少的标注数据就能达到最佳的分类性能。

这些具有开创性的工作促成了两个著名 Transformer 模型的出现：[GPT](https://openai.com/blog/language-unsupervised/) (the Generative Pretrained Transformer) 和 [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers)。通过将 Transformer 结构与无监督学习相结合，模型不再需要对每一个任务都从头训练特定的模型，并且几乎在所有 NLP 任务中都远远超过先前的最强基准。

GPT 和 BERT 被提出之后，NLP 领域出现了越来越多基于 Transformer 结构的模型，其中比较有名有：

![transformers_chrono](/img/article/transformers-biography/transformers_chrono.svg)

虽然新的 Transformer 模型层出不穷，它们采用不同的预训练目标在不同的数据集上进行训练，但是大致上依然可以按模型结构将它们分为三类：

- **纯 Encoder 模型**（例如 BERT），又被称为自编码 (auto-encoding) Transformer 模型；
- **纯 Decoder 模型**（例如 GPT），又被称为自回归 (auto-regressive) Transformer 模型；
- **Encoder-Decoder 模型**（例如 BART、T5），又被称为 Seq2Seq (sequence-to-sequence) Transformer 模型。

后面我们会对这三种模型框架进行更详细的介绍。

## Transformer 是什么

### 语言模型

上面所有提到的 Transformer 模型本质上都是预训练语言模型，它们都采用自监督的方式在大量的生语料 (raw text) 上进行训练。自监督学习是一种训练目标可以根据模型的输入自动计算的训练方法，也就是说，训练这些 Transformer 模型完全不需要人工标注数据。

例如下面两个常用的预训练任务：

- 基于句子中的前 $n$ 个词来预测下一个词，因为输出依赖于过去和当前（而不是未来）的输入，因此该任务被称为**因果语言建模** (causal language modeling)；

  <img src="/img/article/transformers-biography/causal_modeling.svg" alt="causal_modeling" style="display: block; margin: auto; width: 700px"/>

- 基于上下文（周围的词语）来预测句子中被遮盖掉的词语 (masked word)，因此该任务被称为**遮盖语言建模** (masked language modeling)。

  <img src="/img/article/transformers-biography/masked_modeling.svg" alt="masked_modeling" style="display: block; margin: auto; width: 700px"/>


这些语言模型虽然可以对训练过的语言产生统计意义上的理解，例如可以根据上下文预测被遮盖掉的词语，但是如果直接拿来完成特定任务，效果往往并不好。因此我们通常还会采用迁移学习 (transfer learning) 的方法，使用特定任务的标注语料，以有监督学习的方式对模型参数进行微调 (fine-tune)，以取得更好的任务性能。

### 大模型与碳排放

除了 DistilBERT 等少数例外，大部分 Transformer 模型都为了取得更好的性能而不断地增加模型大小（模型参数量）以及增加预训练使用的数据量。下图展示了近年来模型大小的变化趋势：

<img src="/img/article/transformers-biography/nlp_models_size.png" alt="nlp_models_size" style="display: block; margin: auto; width: 700px"/>

但是，从头训练一个模型，尤其是大模型，需要海量的数据，这造成训练模型的时间和计算成本都非常高，并且对环境的影响也很大：

![carbon_footprint](/img/article/transformers-biography/carbon_footprint.svg)

可以想象，如果每一次研究者或是公司想要训练一个语言模型，都需要基于海量的数据从头开始训练，将耗费巨大且不必要的全球成本，因此共享语言模型非常重要。只要我们共享并且在预训练好的模型权重上构建自己的模型，就可以大幅地降低计算成本和碳排放。

> 现在也有一些工作致力于在（尽可能）保持模型性能的情况下大幅减少参数量，达到用“小模型”获得媲美“大模型”的效果。

### 迁移学习

前面已经讲过，预训练是一种从头开始训练模型的方式：所有的模型权重都被随机初始化，然后在没有任何先验知识的情况下开始训练：

<img src="/img/article/transformers-biography/pretraining.svg" alt="pretraining" style="display: block; margin: auto; width: 700px"/>

这个过程不仅需要海量的训练数据，而且时间和经济成本都非常高。

因此，在大部分情况下，我们都不会从头训练模型，而是将别人预训练好的模型通过迁移学习应用到自己的任务中，即使用任务对应的语料对模型进行“二次训练”，通过微调模型参数使模型适应于新的任务。

这种迁移学习的好处是：

- 模型在预训练时很可能已经见过与我们任务类似的数据集，因此通过微调就可以激发出模型在预训练过程中获得的知识，将模型基于海量数据获得的统计理解能力应用于我们的任务；
- 由于预训练模型已经在大量数据上进行过训练，因此微调时只需要很少的数据量就可以达到不错的性能；
- 出于同样的原因，在自定义任务中获得优秀性能所需的时间和计算成本都可以很小。

例如，我们可以选择一个在大规模英文语料上预训练好的模型，然后使用 arXiv 语料进行微调，以生成一个面向学术/研究领域的模型。这个微调的过程只需要有限数量的数据：我们相当于将预训练模型已经获得的知识“迁移”到了新的领域，因此被称为**迁移学习**。

<img src="/img/article/transformers-biography/finetuning.svg" alt="finetuning" style="display: block; margin: auto; width: 700px"/>

与从头训练相比，微调模型所需的时间、数据、经济和环境成本都要低得多，并且与完整的预训练相比，微调训练的约束更少，因此迭代尝试不同的微调方案也更快、更容易。实践证明，即使是对于自定义任务，除非你有大量的语料，否则相比训练一个专门的模型，基于预训练模型进行微调会是一个更好的选择。

因此，**在绝大部分情况下，我们都应该尝试找到一个尽可能接近我们任务的预训练模型，然后微调它**，也就是所谓的“站在巨人的肩膀上”。

## Transformer 的结构

标准的 Transformer 模型主要由两个模块构成：

- **Encoder（左边）：**负责理解输入文本，为每个输入构造对应的语义表示（语义特征），；
- **Decoder（右边）：**负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列。

<img src="/img/article/transformers-biography/transformers_blocks.svg" alt="transformers_blocks" style="display: block; margin: auto; width: 700px"/>

这两个模块可以根据任务的需求而单独使用：

- **纯 Encoder 模型：**适用于只需要理解输入语义的任务，例如句子分类和命名实体识别；
- **纯 Decoder 模型：**适用于生成式任务，例如文本生成；
- **Encoder-Decoder 模型**或 **Seq2Seq 模型：**适用于需要基于输入的生成式任务，例如翻译和摘要。

我们后面会具体地介绍每一种框架。

### 注意力层

Transformer 模型的标志性特征就是它采用了一种称为**注意力层** (Attention Layers) 的结构，前面也说过，提出 Transformer 结构的论文名字就叫[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)。顾名思义，注意力层的作用就是让模型在处理文本时，更多地关注句子中的某些词语，而在一定程度上忽略其他词语，也就是将注意力只放在某些词语上。

注意力机制非常有用，例如我们要将英文句子“You like this course”翻译为法语，由于法语中动词“like”的变位方式因主语而异，因此模型要为词语“like”生成合适的翻译就需要同时关注相邻的词语“You”，而其他的词语则对翻译该词没什么帮助。同样地，在翻译“this”时，模型还需要注意“course”这个词，因为“this”的法语翻译会根据相关名词的极性而有所不同，而其他词则对翻译“this”没什么用。对于更加复杂的句子，很多时候如果要正确地翻译某个词语，甚至需要关注在句子中离这个词很远的单词。

同样的概念适用于任何 NLP 任务：虽然词语本身就有语义，但是词语的语义同样深受上下文的影响，同一个词语出现在不同上下文中可能具有完全不同的语义（例如“我买了一个苹果”和“我买了一个苹果手机”中的“苹果”），这里上下文可以是该词前面或后面的任何词语（或多个词）。

### 原始结构

Transformer 模型本来是为了翻译任务而设计的。在训练过程中，Encoder 接受源语言的句子作为输入，而 Decoder 则接受目标语言的翻译作为输入。在 Encoder 中，由于翻译一个词语需要依赖于上下文，因此注意力层可以访问句子中的所有词语；而 Decoder 是顺序地进行解码，在生成某个词语时，注意力层只能访问前面已经生成的单词。例如，假设翻译模型当前已经预测出了三个词语，我们会把这三个词语作为输入送入 Decoder，然后 Decoder 结合 Encoder 所有的输入来预测第四个词语。

> 实际训练中为了加快速度，会将整个目标序列都送入 Decoder，然后在注意力层中通过 mask 遮盖掉未来的词语来防止信息泄露。例如我们在预测第三个词语时，应该只能访问到已生成的前两个词语，如果 Decoder 能够访问到序列中的第三个（甚至后面的）词语，就相当于作弊了。

原始的 Transformer 模型结构如下图所示，Encoder 在左，Decoder 在右：

![transformers](/img/article/transformers-biography/transformers.svg)

其中，Decoder 中的第一个注意力层关注 Decoder 过去所有的输入，而第二个注意力层则是使用 Encoder 的输出，因此 Decoder 可以基于整个输入句子来预测当前词语。这对于翻译任务非常有用，因为同一句话在不同语言下的词语顺序可能并不一致（不能逐词翻译），所以出现在源语言句子后部的词语反而可能对目标语言句子前部词语的预测非常重要。

> 在 Encoder/Decoder 的注意力层中，我们还会使用 Attention Mask 遮盖掉某些词语来防止模型关注它们，例如为了将数据处理为相同长度而向序列中添加的填充 (padding) 字符。

## Transformer 家族

随着时间的推移，新的 Transformer 模型层出不穷，但是它们依然可以被归类到三种主要架构下：

<img src="/img/article/transformers-biography/main_transformer_architectures.png" alt="main_transformer_architectures" style="display: block; margin: auto; width: 400px" />

### Encoder 分支

纯 Encoder 模型只使用 Transformer 模型中的 Encoder 模块，也被称为自编码 (auto-encoding) 模型。在每个阶段，注意力层都可以访问到原始输入句子中的所有词语，即具有“双向 (Bi-directional)”注意力。纯 Encoder 模型通常通过破坏给定的句子（例如随机遮盖其中的单词），然后让模型进行重构来进行预训练，最适合处理那些需要理解整个句子语义的任务，例如句子分类、命名实体识别（词语分类）和抽取式问答。

BERT 是第一个基于 Transformer 结构的纯 Encoder 模型，它在提出时横扫了整个 NLP 界，在流行的 [GLUE](https://arxiv.org/abs/1804.07461) 基准（通过多个任务度量模型的自然语言理解能力）上超过了当时所有的最强模型。随后的一系列工作对 BERT 的预训练目标和架构进行调整以进一步提高性能。时至今日，纯 Encoder 模型依然在 NLP 行业中占据主导地位。

下面我们简略地介绍一下 BERT 模型以及各种变体：

- [BERT](https://arxiv.org/abs/1810.04805)：通过预测文本中被掩码的词语和判断一个文本是否跟随着另一个来进行预训练，前一个任务被称为**遮盖语言建模** (Masked Language Modeling, MLM)，后一个任务被称为**下一句预测** (Next Sentence Prediction, NSP)；
- [DistilBERT](https://arxiv.org/abs/1910.01108)：尽管 BERT 模型性能优异，但它的模型大小使其难以部署在低延迟需求的环境中。 通过在预训练期间使用知识蒸馏 (knowledge distillation) 技术，DistilBERT 在内存占用减少 40%、计算速度提高 60% 的情况下，依然可以保持 BERT 模型 97% 的性能；
- [RoBERTa](https://arxiv.org/abs/1907.11692)：BERT 之后的一项研究表明，通过修改预训练方案可以进一步提高性能。 RoBERTa 在更多的训练数据上，以更大的批次训练了更长的时间，并且放弃了 NSP 任务。与 BERT 模型相比，这些改变显著地提高了模型的性能；
- [XLM](https://arxiv.org/abs/1901.07291)：跨语言语言模型 (XLM) 探索了构建多语言模型的数个预训练目标，包括来自 GPT 模型的自回归语言建模和来自 BERT 的 MLM。此外，研究者还通过将 MLM 任务拓展到多语言输入，提出了翻译语言建模 (Translation Language Modeling, TLM)。XLM 模型基于这些任务进行预训练后，在数个多语言 NLU 基准和翻译任务上取得了最好的性能；
- [XLM-RoBERTa](https://arxiv.org/abs/1911.02116)：跟随 XLM 和 RoBERTa 的工作，XLM-RoBERTa (XLM-R) 模型通过升级训练数据使得多语言预训练更进一步。具体地，首先基于 Common Crawl 语料库创建了一个包含 2.5 TB 文本数据的语料，然后在该数据集上运用 MLM 训练了一个编码器。由于数据集没有包含平行对照文本，因此移除了 XLM 的 TLM 目标。最终，该模型大幅超越了 XLM 和多语言 BERT 变体；
- [ALBERT](https://arxiv.org/abs/1909.11942)：ALBERT 模型通过三处变化使得 Encoder 架构更高效：首先，它将词嵌入维度与隐藏维度解耦，使得嵌入维度很小以减少模型参数；其次，所有模型层共享参数，这进一步减少了模型的实际参数量；最后，将 NSP 任务替换为句子排序预测，即预测两个连续句子的顺序是否被交换。这些变化使得可以用更少的参数训练更大的模型，并在 NLU 任务上取得了优异的性能；
- [ELECTRA](https://arxiv.org/abs/2003.10555)：标准 MLM 预训练的一个缺陷是，在每个训练步骤中，只有被遮盖掉词语的表示会得到更新。ELECTRA 使用了一种双模型方法来解决这个问题：第一个模型（通常很小）继续按标准的遮盖语言模型一样工作，预测被遮盖的词语；第二个模型（称为鉴别器）则预测第一个模型的输出中哪些词语是被遮盖的，因此判别器需要对每个词语进行二分类，这使得训练效率提高了 30 倍。对于下游任务，鉴别器也像标准 BERT 模型一样进行微调；
- [DeBERTa](https://arxiv.org/abs/2006.03654)：DeBERTa 模型引入了两处架构变化。首先，每个词语都被表示为两个向量，一个用于记录内容，另一个用于记录相对位置。通过将词语的内容与相对位置分离，自注意力层 (Self-Attention) 层就可以更好地建模邻近词语对的依赖关系。另一方面，词语的绝对位置也很重要（尤其对于解码），因此 DeBERTa 在词语解码头的 softmax 层之前添加了一个绝对位置嵌入。DeBERTa 是第一个在 [SuperGLUE](https://arxiv.org/abs/1905.00537) 基准（更困难的 GLUE 版本）上击败人类基准的模型。

### Decoder 分支

纯 Decoder 模型只使用 Transformer 模型中的 Decoder 模块。在每个阶段，对于某个给定的词语，注意力层只能访问句子中位于它之前的词语，即只能迭代地基于已经生成的词语来逐个预测后面的词语，因此也被称为自回归 (auto-regressive) 模型。纯 Decoder 模型的预训练通常围绕着预测句子中下一个单词展开。纯 Decoder 模型最适合处理那些只涉及文本生成的任务。

对 Transformer Decoder 模型的探索在在很大程度上是由 [OpenAI](https://openai.com/) 带头进行的，通过使用更大的数据集进行预训练，以及将模型的规模扩大，纯 Decoder 模型的性能也在不断提高。

下面我们就来简要介绍一下一些生成模型的演变：

- [GPT](https://openai.com/blog/language-unsupervised)：GPT 结合了新颖高效的 Transformer Decoder 架构和迁移学习，通过根据前面单词预测下一个单词的预训练任务，在 BookCorpus 数据集上进行了训练。GPT 模型在分类等下游任务上取得了很好的效果。
- [GPT-2](https://openai.com/blog/better-language-models/)：受简单且可扩展的预训练方法的启发，OpenAI 通过扩大原始模型和训练集创造了 GPT-2，它能够生成篇幅较长且语义连贯的文本。由于担心被误用，该模型分阶段进行发布，首先公布了较小的模型，然后发布了完整的模型。
- [CTRL](https://arxiv.org/abs/1909.05858)：像 GPT-2 这样的模型虽然可以续写文本（也称为 prompt），但是用户几乎无法控制生成序列的风格。因此，条件 Transformer 语言模型 (Conditional Transformer Language, CTRL) 通过在序列开头添加特殊的“控制符”使得用户可以控制生成文本的风格，并且只需要调整控制符就可以生成多样化的文本。
- [GPT-3](https://arxiv.org/abs/2005.14165)：在成功将 GPT 扩展到 GPT-2 之后，通过对不同规模语言模型行为的分析表明，存在[幂律](https://arxiv.org/abs/2001.08361)来约束计算、数据集大小、模型大小和语言模型性能之间的关系。因此，GPT-2 被进一步放大 100 倍，产生了具有 1750 亿个参数的 GPT-3。除了能够生成令人印象深刻的真实篇章之外，该模型还展示了小样本学习 (few-shot learning) 的能力：只需要给出很少新任务的样本（例如将文本转换为代码），GPT-3 就能够在未见过的新样本上完成任务。但是 OpenAI 并没有开源这个模型，而是通过 OpenAI API 提供了调用接口；
- [GPT-Neo](https://zenodo.org/record/5297715) / [GPT-J-6B](https://github.com/kingoflolz/mesh-transformer-jax)：由于 GPT-3 没有开源，因此一些旨在重新创建和发布 GPT-3 规模模型的研究人员组成了 EleutherAI，GPT-Neo 和 GPT-J-6B 就是由 EleutherAI 训练的类似 GPT 的模型。当前公布的模型具有 1.3、2.7 和 60 亿个参数，在性能上可以媲美 OpenAI 提供的较小版本的 GPT-3 模型。

### Encoder-Decoder 分支

Encoder-Decoder 模型（又称 Seq2Seq 模型）同时使用 Transformer 架构的两个模块。在每个阶段，Encoder 的注意力层都可以访问初始输入句子中的所有单词，而 Decoder 的注意力层则只能访问输入中给定词语之前的词语（已经解码生成的词语）。这些模型可以使用编码器或解码器模型的目标来完成预训练，但通常会包含一些更复杂的任务。例如，T5 通过使用 mask 字符随机遮盖掉输入中的文本片段（包含多个词）进行预训练，训练目标则是预测出被遮盖掉的文本。Encoder-Decoder 模型最适合处理那些需要根据给定输入来生成新句子的任务，例如自动摘要、翻译或生成式问答。

下面我们简单介绍一些在自然语言理解 (NLU) 和自然语言生成 (NLG) 领域的 Encoder-Decoder 模型：

- [T5](https://arxiv.org/abs/1910.10683)：T5 模型将所有 NLU 和 NLG 任务都转换为文本到文本任务来统一解决，这样就可以运用 Encoder-Decoder 框架来完成任务。例如，对于文本分类问题，将文本作为输入送入 Encoder，然后 Decoder 生成文本形式的标签。T5 采用原始的 Transformer 架构，在 C4 大型爬取数据集上，通过遮盖语言建模以及将所有 SuperGLUE 任务转换为文本到文本任务来进行预训练。最终，具有 110 亿个参数的最大版本 T5 模型在多个基准上取得了最优性能。
- [BART](https://arxiv.org/abs/1910.13461)：BART 在 Encoder-Decoder 架构中结合了 BERT 和 GPT 的预训练过程。首先将输入句子通过遮盖词语、打乱句子顺序、删除词语、文档旋转等方式进行破坏，然后通过 Encoder 编码后传递给 Decoder，并且要求 Decoder 能够重构出原始的文本。这使得模型可以灵活地用于 NLU 或 NLG 任务，并且在两者上都实现了最优性能。
- [M2M-100](https://arxiv.org/abs/2010.11125)：一般翻译模型是按照特定的语言对和翻译方向来构建的，因此无法用于处理多语言。事实上，语言对之间可能存在共享知识，这可以用来处理稀有语言之间的翻译。M2M-100 是第一个可以在 100 种语言之间进行翻译的模型，并且对小众的语言也能生成高质量的翻译。该模型使用特殊的前缀标记（类似于 BERT 的 `[CLS]`）来指示源语言和目标语言。
- [BigBird](https://arxiv.org/abs/2007.14062)：由于注意力机制 $\mathcal{O}(n^2)$ 的内存要求，Transformer 模型只能处理一定长度范围内的文本。 BigBird 通过使用线性扩展的稀疏注意力形式解决了这个问题，从而将可处理的文本长度从大多数 BERT 模型的 512 大幅扩展到 4096，这对于处理文本摘要等需要保存长距离依赖关系的任务特别有用。

## 小结

至此，我们对 Transformer 模型的参观旅程就全部结束了，相信你已经对 Transformer 模型的定义和发展有了大概的了解，接下来就可以针对自己的兴趣对某一个部分进行更深入地探索。

幸运的是，[Hugging Face](https://huggingface.co/) 专门为使用 Transformer 模型编写了一个 [Transformers 库](https://huggingface.co/docs/transformers/index)，我们前面介绍的所有 Transformer 模型都可以在 [Hugging Face Hub](https://huggingface.co/models) 中找到并且加载使用。如果你对如何使用 Transformers 库感兴趣，可以参考我写的 [Transformers 库快速入门教程](/2021/12/08/transformers-note-1.html)。

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://transformersbook.com/) Lewis Tunstall 等人. 《Natural Language Processing with Transformers》