---
layout: article
title: "运用 BERT 的 MLM 模型进行小样本学习"
author: 苏剑林
tags:
    - NLP
mathjax: true
---

> 转载自[《必须要GPT3吗？不，BERT的MLM模型也能小样本学习》](https://kexue.fm/archives/7764)和[《P-tuning：自动构建模版，释放语言模型潜能》](https://kexue.fm/archives/8295)，作者：苏剑林，部分内容有修改。

大家都知道现在 GPT3 风头正盛，然而，到处都是 GPT3、GPT3 地推，读者是否记得 GPT3 论文的名字呢？事实上，GPT3 的论文叫做[《Language Models are Few-Shot Learners》](https://arxiv.org/abs/2005.14165)，标题里边已经没有 G、P、T 几个单词了，只不过它跟开始的 GPT 是一脉相承的，因此还是以 GPT 称呼它。顾名思义，GPT3 主打的是 Few-Shot Learning，也就是小样本学习。此外，GPT3 的另一个特点就是大，最大的版本多达 1750 亿参数，是 BERT Base的一千多倍。

正因如此，前些天 Arxiv 上的一篇论文[《It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners》](https://arxiv.org/abs/2009.07118)便引起了笔者的注意，意译过来就是**“谁说一定要大的？小模型也可以做小样本学习”**。显然，这标题对标的就是 GPT3，于是笔者饶有兴趣地点进去看看是谁这么有勇气挑战 GPT3，又是怎样的小模型能挑战 GPT3？经过阅读，原来作者提出通过适当的构造，用 **BERT 的 MLM 模型**也可以做小样本学习，看完之后颇有一种“原来还可以这样做”的恍然大悟感～在此与大家分享一下。

## 冉冉升起的 MLM

MLM，全称“Masked Language Model”，可以翻译为“掩码语言模型”，实际上就是一个完形填空任务，随机 Mask 掉文本中的某些字词，然后要模型去预测被 Mask 的字词，示意图如下：

<img src="/img/article/from-pet-to-p-tuning/bert_mlm.png" width="500px" style="display:block; margin:auto;"/>

<center>BERT 的 MLM 模型简单示意图</center>

其中被 Mask 掉的部分，可以是直接随机选择的 Token，也可以是随机选择连续的能组成一整个词的 Token，后者称为 Whole Word Masking (WWM)。

开始，MLM 仅被视为 BERT 的一个预训练任务，训练完了就可以扔掉的那种，因此有一些开源的模型干脆没保留 MLM 部分的权重，比如 [brightmart版](https://github.com/brightmart/roberta_zh) 和 [clue版](https://github.com/CLUEbenchmark/CLUEPretrainedModels) 的 RoBERTa，而哈工大开源的 [RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm) 则不知道出于什么原因随机初始化了 MLM 部分的权重，因此如果要复现本文后面的结果，这些版本是不可取的。

然而，随着研究的深入，研究人员发现不止 BERT 的 Encoder 很有用，预训练用的 MLM 本身也很有用。比如论文[《BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model》](https://arxiv.org/abs/1902.04094)指出 MLM 可以作为一般的生成模型用，论文[《Spelling Error Correction with Soft-Masked BERT》](https://kexue.fm/archives/7661)则将 MLM 用于文本纠错，笔者之前在[《从语言模型到Seq2Seq：Transformer如戏，全靠Mask》](https://kexue.fm/archives/6933)的实验也表明 MLM 的预训练权重也可以当作 UniLM 来用做 Seq2Seq 任务，还有[《无监督分词和句法分析！原来BERT还可以这样用》](https://kexue.fm/archives/7476)一文将 MLM 的思想用于无监督分词和句法分析了。可以说 MLM 已经是大放异彩了。

## 将任务转成完形填空

在本文里，我们再学习 MLM 的一个精彩应用：用于小样本学习或半监督学习，某些场景下甚至能做到零样本学习。

怎么将我们要做的任务跟 MLM 结合起来呢？很简单，**给任务一个文本描述，然后转换为完形填空问题**即可。举个例子，假如给定句子“这趟北京之旅我感觉很不错。”，那么我们补充个描述，构建如下的完形填空：

<center>______满意。这趟北京之旅我感觉很不错。</center>

进一步地，我们限制空位处只能填一个“很”或“不”，问题就很清晰了，就是要我们根据上下文一致性判断是否满意，如果“很”的概率大于“不”的概率，说明是正面情感倾向，否则就是负面的，这样我们就将情感分类问题转换为一个完形填空问题了，它可以用 MLM 模型给出预测结果，而 MLM 模型的训练可以不需要监督数据，因此理论上这能够实现零样本学习了。

多分类问题也可以做类似转换，比如新闻主题分类，输入句子为“八个月了，终于又能在赛场上看到女排姑娘们了。”，那么就可以构建

<center>下面报导一则______新闻。八个月了，终于又能在赛场上看到女排姑娘们了。</center>

这样我们就将新闻主题分类也转换为完形填空问题了，一个好的 MLM 模型应当能预测出“体育”二字来。

还有一些简单的推理任务也可以做这样的转换，常见的是给定两个句子，判断这两个句子是否相容，比如“我去了北京”跟“我去了上海”就是矛盾的，“我去了北京”跟“我在天安门广场”是相容的，常见的做法就是将两个句子拼接起来输入到模型做，作为一个二分类任务。如果要转换为完形填空，那该怎么构造呢？一种比较自然的构建方式是：

<center>我去了北京？______，我去了上海。<br />我去了北京？______，我在天安门广场。</center>

其中空位之处的候选词为 $\{\text{是的}, \text{不是}\}$。

## Pattern-Exploiting

读到这里，读者应该不难发现其中的规律了，就是给输入的文本增加一个前缀或者后缀描述，并且 Mask 掉某些 Token，转换为完形填空问题，这样的转换在原论文中称为 Pattern，这个转换要尽可能与原来的句子组成一句自然的话，不能过于生硬，因为预训练的 MLM 模型就是在自然语言上进行的。显然同一个问题可以有很多不同的 Pattern，比如情感分类的例子，描述可以放最后，变成“这趟北京之旅我感觉很不错。____满意。”；也可以多加几个字，比如“觉得如何？____满意。这趟北京之旅我感觉很不错。”。

然后，我们需要构建预测 Token 的候选空间，并且建立 Token 到实际类别的映射，这在原论文中称为 Verbalizer，比如情感分类的例子，我们的候选空间是 $\{\text{很}, \text{不}\}$，映射关系是 $\text{很}\to\text{正面},\text{不}\to\text{负面}$，候选空间与实际类别之间不一定是一一映射，比如我们还可以加入“挺”、“太”、“难”字，并且认为 $\{\text{很},\text{挺},\text{太}\}\to\text{正面}$ 以及 $\{\text{不},\text{难}\}\to\text{负面}$，等等。不难理解，不少 NLP 任务都有可能进行这种转换，但显然这种转换一般只适用于候选空间有限的任务，说白了就是只用来做选择题，常见任务的就是文本分类。

刚才说了，同一个任务可以有多种不同的Pattern，原论文是这样处理的：

```
1、对于每种 Pattern，单独用训练集 Finetune 一个 MLM 模型出来；
2、然后将不同 Pattern 对应的模型进行集成，得到融合模型；
3、用融合模型预测未标注数据的伪标签；
4、用伪标签数据 Finetune 一个常规的（非 MLM 的）模型。
```

具体的集成方式大家自己看论文就行，这不是重点。这种训练模式被称为 Pattern-Exploiting Training (PET)，它首先出现在论文[《Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference》](https://arxiv.org/abs/2001.07676)，本文要介绍的这篇论文则进一步肯定和完善了 Pattern-Exploiting Training 的价值和结果，并整合了多任务学习，使得它在 SuperGLUE 榜单上的小样本学习效果超过了 GPT3。两篇论文的作者是相同的，是一脉相承的作品。

<img src="/img/article/from-pet-to-p-tuning/pet_results_on_superglue.png" width="700px" style="display:block; margin:auto;"/>

<center>PET 在 SuperGLUE 上的小样本学习的结果</center>

不过要吐槽一个点是，上图中 PET 的 223M 参数，所用的模型是 ALBERT-xxlarge-v2，事实上称 ALBERT 为“小模型”是一种很耍流氓的行为，因为它前向计算的速度并没有得到任何提升。ALBERT-xxlarge 共有 12 层，层与层之间参数是共享的，就前向计算而言，它应该等价于约 2700M（12 倍）参数的 GPT 才对。

## PET 中文实践，检验效果

要真正确认一个方法或模型的价值，看论文的实验表格是不够的，论文给出的实验结果谁都不好说能否复现，其次就算英文上能复现也不代表中文上有s价值，因此最实际的还是亲自动手做实验验证。下面是笔者的实验代码，供读者参考：

<center><strong>Github地址：</strong><a href="https://github.com/bojone/Pattern-Exploiting-Training" target="_blank">https://github.com/bojone/Pattern-Exploiting-Training</a></center>

我们将从以下几个角度来探讨 PET 的可行性：

```
1、直接利用现成的 MLM 模型效果如何？（零样本学习1）
2、用“大量无标签数据”微调现成的 MLM 模型效果如何？（零样本学习2）
3、用“小量标签数据”微调现成的 MLM 模型效果如何？（小样本学习）
4、用“小量标签数据+大量无标签数据”微调现成的 MLM 模型效果如何？（半监督学习）
```

下面主要给出情感二分类的实验结果。另外还有一个新闻主题的多分类，代码也放到 Github 了，其结果是类似的，就不重复陈述了。

### 零样本学习1

这里主要探索的是给输入文本补上对应的 Pattern 后，直接基于现成的 MLM 模型进行预测，预测的准确率。由于构建模型的整个过程都不涉及到标签数据监督训练，因此这算是一种“零样本学习”。我们需要比较的是不同 Pattern、不同 MLM 模型上的效果：

下面是实验的几个 Pattern，其中空位处候选词语都为“很”和“不”：

```
P1：____满意。这趟北京之旅我感觉很不错。
P2：这趟北京之旅我感觉很不错。____满意。
P3：____好。这趟北京之旅我感觉很不错。
P4：____理想。这趟北京之旅我感觉很不错。
P5：感觉如何？____满意。这趟北京之旅我感觉很不错。
```

至于 MLM 模型，则是下面几个：

> M1：Google 开源的中文版 BERT Base（[链接](https://github.com/google-research/bert)）；
>
> M2：哈工大开源的 RoBERTa-wwm-ext Base（[链接](https://github.com/ymcui/Chinese-BERT-wwm)）：
>
> M3：腾讯 UER 开源的 BERT Base（[链接](https://share.weiyun.com/5QOzPqq)）；
>
> M4：腾讯 UER 开源的BERT Large（[链接](https://share.weiyun.com/5G90sMJ)）。

实验结果如下表（验证集/测试集）：

$$
\begin{array}{c} 
\text{不同模型不同Pattern的零样本学习效果} 
\\ 
{\begin{array}{c|ccccc} 
\hline 
& \text{P1} & \text{P2} & \text{P3} & \text{P4} & \text{P5} \\ 
\hline 
\text{M1} & 66.94\,/\,67.60 & 57.56\,/\,56.13 & 58.83\,/\,59.69 & 83.70\,/\,83.33 & 75.98\,/\,76.13\\ 
\text{M2} & 85.17\,/\,84.27 & 70.63\,/\,68.69 & 58.55\,/\,59.12 & 81.81\,/\,82.28 & 80.25\,/\,81.62\\ 
\text{M3} & 66.75\,/\,68.64 & 50.45\,/\,50.97 & 68.97\,/\,70.11 & 81.95\,/\,81.48 & 61.49\,/\,62.58\\ 
\text{M4} & 83.56\,/\,85.08 & 72.52\,/\,72.10 & 76.46\,/\,77.03 & 88.25\,/\,87.45 & 82.43\,/\,83.56\\ 
\hline 
\end{array}} 
\end{array}
$$

最好的效果居然可以达到 88%！也就是说，加载现成的 MLM，配合适当的 Pattern，不需要任何标注数据，就可以正确识别大部分样本情感倾向了。这不得不让我们对 MLM 模型的潜力刮目相看了。

可以观察到，不同的 Pattern、不同的预训练模型之间还是有一定的差异的，整体而言 Large 版本的效果要明显好于 Base 版本的模型，说明像 GPT 到 GPT2 再到 GPT3 一样，还是把模型做得更大会更好。此外，这还有可能说明实际上 MLM 还没有被充分训练好，或许是因为 BERT 这种 Mask 掉一部分的训练方式过于低效了，可能用[《修改Transformer结构，设计一个更快更好的MLM模型》](https://kexue.fm/archives/7661)一文提到的改进版 MLM 会更好。

### 零样本学习2

看完上述结果，读者可能会想到：如果我用领域内的数据继续预训练 MLM 模型，那么能不能提升效果呢？答案是：能！下面是我们的实验结果，算力有限，我们只在 RoBERTa-wwm-ext（上述的 M2，继续预训练后的模型我们称为 $\text{M2}^{+\text{无监督}}$）的基础上做了比较：

$$
\begin{array}{c} 
\text{继续MLM预训练的零样本学习效果} 
\\ 
{\begin{array}{c|ccccc} 
\hline 
& \text{P1} & \text{P2} & \text{P3} & \text{P4} & \text{P5} \\ 
\hline 
\text{M2} & 85.17\,/\,84.27 & 70.63\,/\,68.69 & 58.55\,/\,59.12 & 81.81\,/\,82.28 & 80.25\,/\,81.62\\ 
\text{M2}^{+\text{无监督}} & 88.05\,/\,87.53 & 71.01\,/\,68.78 & 81.05\,/\,81.24 & 86.40\,/\,85.65 & 87.26\,/\,87.40\\ 
\hline 
\end{array}} 
\end{array}
$$

要注意的是，这里我们只是用领域内的数据继续做 MLM 训练，这个过程是无监督的，也不需要标注信号，因此也算是“零样本学习”。同时，从到目前为止的结果我们可以看出，给输入本文加入“前缀”的效果比“后缀”更有优势一些。

### 小样本学习

刚才我们讨论了无标签数据继续预训练 MLM 的提升，如果回到 PET 的目标场景，直接用小量的标签数据配合特定的 Pattern 训练 MLM 又如何呢？这也就是真正的“小样本学习”训练了，这里我们保留约 200 个标注样本，构造样本的时候，我们先给每个句子补上 Pattern，除了 Pattern 自带的 Mask 位置之外，我们还随机 Mask 其他一部分，以增强对模型的正则。最终实验结果如下：

$$
\begin{array}{c} 
\text{小样本学习效果} 
\\ 
{\begin{array}{c|ccccc} 
\hline 
& \text{P1} & \text{P2} & \text{P3} & \text{P4} & \text{P5} \\ 
\hline 
\text{M2} & 85.17\,/\,84.27 & 70.63\,/\,68.69 & 58.55\,/\,59.12 & 81.81\,/\,82.28 & 80.25\,/\,81.62\\ 
\text{M2}^{+\text{小样本}} & 89.29\,/\,89.18 & 84.71\,/\,82.76 & 88.91\,/\,89.05 & 89.31\,/\,89.13 & 89.07\,/\,88.75\\ 
\hline 
\end{array}} 
\end{array}
$$

结论就是除了“后缀式”的 P2 之外，其它结果都差不多，这进一步说明了“前缀式”的 Pattern 会比“后缀式”更有竞争力一些。在效果上，直接用同样的数据用常规的方法去微调一个 BERT 模型，大概的结果是 88.93 左右，所以基于“MLM+Pattern”的小样本学习方法可能带来轻微的性能提升。

### 半监督学习

无监督的零样本学习和有监督的小样本学习都说完了，自然就轮到把标注数据和非标注数据都结合起来的“半监督学习”了。还是同样的任务，标注数据和非标注数据的比例大约是 1:99，标注数据带 Pattern，非标注数据不带 Pattern，大家都 Mask 掉一部分 Token 进行 MLM 预训练，最终测出来的效果如下：

$$
\begin{array}{c} 
\text{半监督学习效果} 
\\ 
{\begin{array}{c|ccccc} 
\hline 
& \text{P1} & \text{P2} & \text{P3} & \text{P4} & \text{P5} \\ 
\hline 
\text{M2} & 85.17\,/\,84.27 & 70.63\,/\,68.69 & 58.55\,/\,59.12 & 81.81\,/\,82.28 & 80.25\,/\,81.62\\ 
\text{M2}^{+\text{半监督}} & 90.09\,/\,89.76 & 79.58\,/\,79.35 & 90.19\,/\,88.96 & 90.05\,/\,89.54 & 89.88\,/\,89.23\\ 
\hline 
\end{array}} 
\end{array}
$$

还是同样的，“后缀”明显比“前缀”差，“前缀”的效果差不多。具体效果上，则是肯定了额外的无标注数据也是有作用的。直觉上来看，“前缀”比“后缀”要好，大体上是因为“前缀”的 Mask 位置比较固定，微弱的监督信号得以叠加增强？但这也不能解释为什么零样本学习的情况下也是“前缀”更好，估计还跟模型的学习难度有关系，可能句子前面部分的规律更加明显，相对来说更加容易学一些，所以前面部分就学习得更加充分？这一切都还只是猜测。

### 汇总与结论

将上述结果汇总如下：

$$
\begin{array}{c} 
\text{结果汇总比较} 
\\ 
{\begin{array}{c|ccccc} 
\hline 
& \text{P1} & \text{P2} & \text{P3} & \text{P4} & \text{P5} \\ 
\hline 
\text{M2} & 85.17\,/\,84.27 & 70.63\,/\,68.69 & 58.55\,/\,59.12 & 81.81\,/\,82.28 & 80.25\,/\,81.62\\ 
\text{M2}^{+\text{无监督}} & 88.05\,/\,87.53 & 71.01\,/\,68.78 & 81.05\,/\,81.24 & 86.40\,/\,85.65 & 87.26\,/\,87.40\\ 
\text{M2}^{+\text{小样本}} & 89.29\,/\,89.18 & 84.71\,/\,82.76 & 88.91\,/\,89.05 & 89.31\,/\,89.13 & 89.07\,/\,88.75\\ 
\text{M2}^{+\text{半监督}} & 90.09\,/\,89.76 & 79.58\,/\,79.35 & 90.19\,/\,88.96 & 90.05\,/\,89.54 & 89.88\,/\,89.23\\ 
\hline 
\end{array}} 
\end{array}
$$

读者还可以对比我们之前在文章[《泛化性乱弹：从随机噪声、梯度惩罚到虚拟对抗训练》](https://kexue.fm/archives/7466#参考实现)中用虚拟对抗训练 (VAT) 做半监督学习的结果，可以看到不管是零样本学习、小样本学习还是半监督学习，基于 MLM 模型的方式都能媲美基于 VAT 的半监督学习的结果。我们在做短新闻多分类实验时的结果也是相似的。因此，这说明了 MLM 模型确实也可以作为一个优秀的零样本/小样本/半监督学习器来使用。

当然，基于 MLM 模型的缺点还是有的，比如 MLM 所使用的独立假设限制了它对更长文本的预测能力（说白了空位处的文字不能太长），以及无法预测不定长的答案也约束了它的场景（所以当前只能用于做选择题，不能做生成）。我们期待有更强的 MLM 模型出现，那时候就有可能在所有任务上都能与 GPT3 一较高下了。

## 什么是模版

前面介绍的 Pattern-Exploiting Training (PET) 方法，其主要的思想是借助由自然语言构成的模版（英文常称 Pattern 或 Prompt），将下游任务也转化为一个完形填空任务，这样就可以用 BERT 的 MLM 模型来进行预测了。比如下图中通过条件前缀来实现情感分类和主题分类的例子：

<img src="/img/article/from-pet-to-p-tuning/sentiment_task_to_mlm.png" width="500px" style="display:block; margin:auto;"/>

<center>通过特定模版将情感分类转换为 MLM 任务</center>

<img src="/img/article/from-pet-to-p-tuning/news_classification_to_mlm.png" width="500px" style="display:block; margin:auto;"/>

<center>通过特定模版将新闻分类转换为 MLM 任务</center>

当然，这种方案也不是只有 MLM 模型可行，用 GPT 这样的单向语言模型（LM）其实也很简单：

<img src="/img/article/from-pet-to-p-tuning/sentiment_task_to_lm.png" width="500px" style="display:block; margin:auto;"/>

<center>通过特定模版将情感分类转换为 LM 任务</center>

<img src="/img/article/from-pet-to-p-tuning/news_classification_to_lm.png" width="500px" style="display:block; margin:auto;"/>

<center>通过特定模版将新闻分类转换为 LM 任务</center>

不过由于语言模型是从左往右解码的，因此预测部分只能放在句末了（但还可以往补充前缀说明，只不过预测部分放在最后）。

某种意义上来说，这些模版属于语言模型的“探针”，我们可以通过模版来抽取语言模型的特定知识，从而做到不错的零样本效果，而配合少量标注样本，可以进一步提升效果。

然而，对于某些任务而言，人工构建模版并不是那么容易的事情，模型的优劣我们也不好把握，而不同模型之间的效果差别可能很大，在这种情况下，人工标注一些样本可能比构建模版还要轻松得多。所以，如何根据已有的标注样本来自动构建模版，便成了一个值得研究的问题了。

## P-tuning

最近 Arxiv 上的论文[《GPT Understands, Too》](https://arxiv.org/abs/2103.10385)提出了名为 P-tuning 的方法，成功地实现了模版的自动构建。不仅如此，借助 P-tuning，GPT 在 SuperGLUE 上的成绩首次超过了同等级别的 BERT 模型，这颠覆了一直以来“GPT 不擅长 NLU”的结论，也是该论文命名的缘由。

P-tuning 重新审视了关于模版的定义，放弃了“模版由自然语言构成”这一常规要求，从而将模版的构建转化为连续参数优化问题，虽然简单，但却有效。

### 模版的反思

首先，我们来想一下“什么是模版”。直观来看，模版就是由自然语言构成的前缀/后缀，通过这些模版我们使得下游任务跟预训练任务一致，这样才能更加充分地利用原始预训练模型，起到更好的零样本、小样本学习效果。

**等等，我们真的在乎模版是不是“自然语言”构成的吗？**

并不是。本质上来说，我们并不关心模版长什么样，**我们只需要知道模版由哪些 token 组成，该插入到哪里，插入后能不能完成我们的下游任务，输出的候选空间是什么。**模版是不是自然语言组成的，对我们根本没影响，“自然语言”的要求，只是为了更好地实现“一致性”，但不是必须的。于是，P-tuning 考虑了如下形式的模版：

<img src="/img/article/from-pet-to-p-tuning/p_tuning.png" width="500px" style="display:block; margin:auto;"/>

<center>P-tuning 直接使用 [unused*] 的 token 来构建模版，不关心模版的自然语言性</center>

这里的 [u1]～[u6]，代表 BERT 词表里边的 [unused1]～[unused6]，也就是用几个从未见过的 token 来构成模板，这里的 token 数目是一个超参数，放在前面还是后面也可以调整。接着，为了让“模版”发挥作用，我们用标注数据来求出这个模板。

### 如何去优化

这时候，根据标注数据量的多少，我们又分两种情况讨论。

**第一种，标注数据比较少。**这种情况下，我们固定整个模型的权重，只优化 [unused1]～[unused6] 这几个 token 的 Embedding，换句话说，其实我们就是要学 6 个新的 Embedding，使得它起到了模版的作用。这样一来，因为模型权重几乎都被固定住了，训练起来很快，而且因为要学习的参数很少，因此哪怕标注样本很少，也能把模版学出来，不容易过拟合。

**第二种，标注数据很充足。**这时候如果还按照第一种的方案来，就会出现欠拟合的情况，因为只有 6 个 token 的可优化参数实在是太少了。因此，我们可以放开所有权重微调，原论文在 SuperGLUE 上的实验就是这样做的。读者可能会想：这样跟直接加个全连接微调有什么区别？原论文的结果是这样做效果更好，可能还是因为跟预训练任务更一致了吧。

<img src="/img/article/from-pet-to-p-tuning/p_tuning_results_on_superglue.png" width="800px" style="display:block; margin:auto;"/>

<center>P-tuning 在 SuperGLUE 上的表现</center>

此外，在上面的例子中，目标 token 如“很”、“体育”是认为选定的，那么它们可不可以也用 [unused\*] 的 token 代替呢？答案是可以，但也分两种情况考虑：1、在标注数据比较少的时候，人工来选定适当的目标 token 效果往往更好些；2、在标注数据很充足的情况下，目标 token 用 [unused\*] 效果更好些，因为这时候模型的优化空间更大一些。

### 增强相关性

在原论文中，P-tuning 并不是随机初始化几个新 token 然后直接训练的，而是通过一个小型的 LSTM 模型把这几个 Embedding 算出来，并且将这个 LSTM 模型设为可学习的。这样多绕了一步有什么好处呢？原论文大概的意思是：LSTM 出现的 token 表示相关性更强，某种程度上来说更像“自然语言”（因为自然语言的 token 之间不是独立的），此外还能防止局部最优。我在 Github 上进一步向作者确认了一下（参考[这里](https://github.com/THUDM/P-tuning/issues/5)），效果上的差别是通过 LSTM 多绕一步的方法可以使得模型收敛更快、效果更优。

然而，这样多了一个LSTM，总感觉有些别扭，而且实现上也略微有点麻烦。按照作者的意思，LSTM 是为了帮助模版的几个 token（某种程度上）更贴近自然语言，但这并不一定要用 LSTM 生成，而且就算用 LSTM 生成也不一定达到这一点。笔者认为，更自然的方法是在训练下游任务的时候，不仅仅预测下游任务的目标 token（前面例子中的“很”、“新闻”），**还应该同时做其他 token 的预测**。

比如，如果是 MLM 模型，那么也随机 mask 掉其他的一些 token 来预测；如果是 LM 模型，则预测完整的序列，而不单单是目标词。这样做的理由是：因为我们的 MLM/LM 都是经过自然语言预训练的，所以我们（迷之自信地）认为能够很好完成重构的序列必然也是接近于自然语言的，因此这样增加训练目标，也能起到让模型更贴近自然语言的效果。经过笔者的测试，加上这样辅助目标，相比单纯优化下游任务的目标，确实提升了效果。

## P-tuning 实验与效果

所谓“talk is cheap, show me the code”，又到了喜闻乐见的实验时间了。这里分享一下 P-tuning 的实验结果，其中还包括笔者对 P-tuning 的实现思路，以及笔者在中文任务上的实验结果。

### 停止的梯度

怎么实现上述的 P-tuning 算法比较好呢？如果是放开所有权重训练，那自然是简单的，跟普通的 BERT 微调没有什么区别。关键是在小样本场景下，如何实现“只优化几个 token”呢？

当然，实现的方法也不少，比如为那几个要优化的 token 重新构建一个 Embedding 层，然后拼接到 BERT 的Embedding层中，然后训练的时候只放开新 Embedding 层的权重。但这样写对原来模型的改动还是蛮大的，最好的方法是尽可能少改动代码，让使用者几乎无感。为此，笔者构思了一种用 `stop_gradient` 简单修改 `Embedding` 层的方案，大体上是将 `Embedding` 层修改如下：

```python
class PtuningEmbedding(Embedding):
    """新定义Embedding层，只优化部分Token
    """
    def call(self, inputs, mode='embedding'):
        embeddings = self.embeddings
        embeddings_sg = K.stop_gradient(embeddings)
        mask = np.zeros((K.int_shape(embeddings)[0], 1))
        mask[1:9] += 1  # 只优化id为1～8的token
        self.embeddings = embeddings * mask + embeddings_sg * (1 - mask)
        return super(PtuningEmbedding, self).call(inputs, mode)
```

变量经过 `stop_gradient` 算子后，在反向传播的时候梯度为 0，但是前向传播不变，因此在上述代码中，前向传播的结果不会有变化，但是反向传播求梯度的时候，梯度不为 0 的 token 由 `mask` 变量控制，其余 token 的梯度都为零，因此就实现了只更新部分 token。

完整代码可见：

<center><strong>Github：</strong><a href="https://github.com/bojone/P-tuning" target="_blank">https://github.com/bojone/P-tuning</a></center>

对了，原论文也开源了代码：

<center><strong>Github：</strong><a href="https://github.com/THUDM/P-tuning" target="_blank">https://github.com/THUDM/P-tuning</a></center>

### 测试与效果

前面已经分享了原作者在 SuperGLUE 上的实验结果，显示出如果配合 P-tuning，那么：1、GPT、BERT 的效果相比直接 finetune 都有所提升；2、GPT 的效果还能超过了 BERT。这表明 GPT 不仅有 NLG 的能力，也有 NLU 能力，可谓是把 GPT 的潜能充分“压榨”出来了，当然 BERT 配合 P-tuning 也有提升，说明 P-tuning 对语言模型潜能的释放是较为通用的。

原论文的实验比较丰富，建议读者仔细阅读原论文，相信会收获颇多。特别指出的是原论文的 Table 2 最后一列，当预训练模型足够大的时候，我们的设备可能无法 finetune 整个模型，而 P-tuning 可以选择只优化几个 Token 的参数，因为优化所需要的显存和算力都会大大减少，所以 **P-tuning 实则上给了我们一种在有限算力下调用大型预训练模型的思路**。

<img src="/img/article/from-pet-to-p-tuning/p_tuning_under_ language_model.png" width="500px" style="display:block; margin:auto;"/>

<center>P-tuning 在各个体量的语言模型下的效果</center>

当然，笔者一直以来的观点是“**没有在中文上测试过的算法是没有灵魂的**”，因此笔者也在中文任务上简单测试了，测试任务跟前文一致，都是情感分类的小样本学习，测试模型包括 BERT 和 GPT，两者的候选模版分别如下图：

<img src="/img/article/from-pet-to-p-tuning/p_tuning_template_for_sentiment_task.png" width="500px" style="display:block; margin:auto;"/>

<center>笔者在中文情感分类上使用的“BERT+P-tuning”模版</center>

<img src="/img/article/from-pet-to-p-tuning/p_tuning_template_for_sentiment_task_2.png" width="500px" style="display:block; margin:auto;"/>

<center>笔者在中文情感分类上使用的“GPT+P-tuning”模版</center>

注意，对于 LM 模型，前缀的引入非常重要，只引入后缀时效果会明显变差；而对于 MLM 模型，前缀的效果通常也优于后缀。总的效果如下表：

$$
\begin{array}{c|cc} 
\hline 
& \text{验证集} & \text{测试集} \\ 
\hline 
\text{小样本直接微调} & 88.93\% & 89.34\% \\ 
\text{VAT半监督学习} & 89.83\% & 90.37\% \\ 
\hline 
\text{PET零样本} & 85.17\% & 84.27\% \\ 
\text{PET无监督} & 88.05\% & 87.53\% \\ 
\text{PET小样本} & 89.29\% & 89.18\% \\ 
\text{PET半监督} & 90.09\% & 89.76\% \\ 
\hline 
\text{BERT + P-tuning} & 89.81\% & 89.75\% \\ 
\text{GPT + P-tuning} & 89.30\% & 88.51\% \\ 
\hline 
\end{array}
$$

其中“小样本”只用到了“少量标注样本”，“无监督”则用到了“大量无标注样本”，“半监督”则用到了“少量标注样本+大量无标注样本”，“P-tuning” 都是小样本，PET 的几个任务报告的是最优的人工模版的结果，其实还有更差的人工模版。从小样本角度来看，P-tuning 确实取得了最优的小样本学习效果；从模版构建的角度来看，P-tuning 确实也比人工构建的模版要好得多；从模型角度看，P-tuning 确实可以将 GPT 的分类性能发挥到跟 BERT 相近，从而揭示了 GPT 也有很强的 NLU 能力的事实。

## 进一步理解 P-tuning

这一节将会介绍笔者对 P-tuning 的进一步思考，以求从多个维度来理解 P-tuning。

### 离散 vs 连续

在 P-tuning 之前，也已经有一些在做模版的自动构建，如[《How Can We Know What Language Models Know?》](https://arxiv.org/abs/1911.12543)、[《AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts》](https://arxiv.org/abs/2010.15980)等，但它们搜索的都是在离散空间下搜索的自然语言模版，所以效果有所限制，并没有取得特别突出的结果。

相反，P-tuning 放弃了“模版由自然语言构成”这一要求，从而将其变成了可以简单梯度下降求解的连续参数问题，效果还更好。同时，这一改动意味着 P-tuning 突出了模版的本质——即**模版的关键在于它是怎么用的，不在于它由什么构成**——给人一种去芜存菁、眼前一亮的感觉，确实值得点赞。

（注：经读者[@brotherb](https://kexue.fm/archives/8295/comment-page-1#comment-16015)提醒，年初有一篇论文[《Prefix-Tuning: Optimizing Continuous Prompts for Generation》](https://arxiv.org/abs/2101.00190)提出的 Prefix-Tuning 方法其实已经相当接近 P-tuning，两者都设计了非自然语言的模版，只不过 Prefix-Tuning 主要关心 NLG 的应用而 P-tuning 更加关心 NLU 的应用。）

### Adapter

我们还可以从 Adapter 的角度来理解 P-tuning。BERT 出来后不久，Google 在论文[《Parameter-Efﬁcient Transfer Learning for NLP》](https://arxiv.org/abs/1902.00751)中提出了一种名为 Adapter 的微调方式，它并不是直接微调整个模型，而是固定住 BERT 原始权重，然后在 BERT 的基础上添加一些残差模块，只优化这些残差模块，由于残差模块的参数更少，因此微调成本更低。Adapter 的思路实际上来源于 CV 的[《Learning multiple visual domains with residual adapters》](https://arxiv.org/abs/1705.08045)，不过这两年似乎很少看到了，也许是因为它虽然提高了训练速度，但是预测速度却降低了，精度往往还有所损失。

在 P-tuning 中，如果我们不将新插入的 token 视为“模版”，是将它视为模型的一部分，那么实际上 P-tuning 也是一种类似 Adapter 的做法，同样是固定原模型的权重，然后插入一些新的可优化参数，同样是只优化这些新参数，只不过这时候新参数插入的是 Embedding 层。因此，从这个角度看，P-tuning 与 Adapter 有颇多异曲同工之处。

### 为什么有效

然后，还有一个值得思考的问题：为什么 P-tuning 会更好？比如全量数据下，大家都是放开所有权重，P-tuning 的方法依然比直接 finetune 要好，为啥呢？

事实上，提出这个问题的读者，应该是对 BERT 加个全连接层的直接 finetune 做法“习以为常”了。很明显，不管是 PET 还是 P-tuning，它们其实都更接近预训练任务，而加个全连接层的做法，其实还没那么接近预训练任务，所以某种程度上来说，P-tuning 有效更加“显然”，反而是加个全连接层微调为什么会有效才是值得疑问的。

去年有篇论文[《A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks》](https://arxiv.org/abs/2010.03648)试图回答这个问题，大致的论证顺序是：

```
1、预训练模型是某种语言模型任务；
2、下游任务可以表示为该种语言模型的某个特殊情形；
3、当输出空间有限的时候，它又近似于加一个全连接层；
4、所以加一个全连接层微调是有效的。
```

可以看到，该论文的假设主要是第 2 点，其实就是直接假设了下游任务可以表达为类似 PET 的形式，然后才去证明的。所以这进一步说明了，PET、P-tuning 等才是更自然的使用预训练模型的方式，加全连接直接 finetune 的做法其实只是它们的推论罢了，也就是说，PET、P-tuning 才是返璞归真、回归本质的方案，所以它们更有效。

> 转载自[《必须要GPT3吗？不，BERT的MLM模型也能小样本学习》](https://kexue.fm/archives/7764)和[《P-tuning：自动构建模版，释放语言模型潜能》](https://kexue.fm/archives/8295)，作者：苏剑林，部分内容有修改。

