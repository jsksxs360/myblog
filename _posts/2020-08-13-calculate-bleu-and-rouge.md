---
layout: article
title: "用 Python 计算文本 BLEU 分数和 ROUGE 值"
tags:
    - NLP
mathjax: true
---

文本生成是自然语言处理 (NLP) 中常见的一类任务，例如机器翻译、自动摘要、图片标题生成等等。如何评估生成文本的质量，或者说衡量生成文本与参考文本之间的差异，是一个必须考虑的问题。目前比较常见的评估方法就是计算 $\text{BLEU}$ 分数和 $\text{ROUGE}$ 值。

## BLEU

$\text{BLEU}$ (Bilingual Evaluation Understudy, 双语评估替换) 一开始是为翻译工作而开发的，是一个比较候选文本翻译与其他一个或多个参考翻译的评价分数。完美匹配的得分为 $1.0$，完全不匹配则得分为 $0.0$。尽管它还没做到尽善尽美，但它具有计算速度快、容易理解、与具体语言无关等优点。

> 翻译系统的 $\text{BLEU}$ 得分不可能为 $1$，除非它们与参考翻译完全相同。通常一个人类翻译在四个参考翻译下的得分为 $0.3468$，在两个参考翻译下的得分为 $0.2571$。

$\text{BLEU}$ 评分由 Kishore Papineni 等人在 2002 年的论文[《BLEU: a Method for Automatic Evaluation of Machine Translation》](http://www.aclweb.org/anthology/P02-1040.pdf)中提出。这种评测方法通过对候选翻译与参考文本中的相匹配的 $n$ 元组进行计数，其中一元组 ($\text{1-gram}$ / $\text{unigram}$) 比较的是每一个单词，而二元组 ($\text{bigram}$) 比较的则是每个单词对，以此类推，并且这种比较是不管单词顺序的。匹配个数越多，表明候选翻译的质量就越好。

同时为了避免机器翻译系统为了追求高精度而生成过多的“合理”单词，从而导致翻译结果不恰当。在识别出匹配的候选单词之后，相应的参考单词就被视为用过了，以确保不会对产生大量合理词汇的候选翻译进行加分。在论文中这被称之为修正的 $n$ 元组精度。

BLEU 评分是用来比较语句的，但是又提出了一个能更好地对语句块进行评分的修订版本，这个修订版根据 n 元组出现的次数来使 n 元组评分正常化。首先逐句计算 n 元组匹配数目，然后为所有候选句子加上修剪过的 n 元组计数，并除以测试语料库中的候选 n 元组个数，以计算整个测试语料库修正后的精度分数 pn。

## ROUGE

$\text{ROUGE}$ 受到了 $\text{BLEU}$ 的启发，不同之处在于 $\text{ROUGE}$ 采用召回率来作为指标。基本思想是将模型生成的摘要与参考摘要的 $n$ 元组贡献统计量作为评判依据。

$\text{ROUGE}$ 由 Chin-Yew Lin 在 2004 年的论文[《ROUGE: A Package for Automatic Evaluation of Summaries》](https://www.aclweb.org/anthology/W04-1013.pdf)中提出。与 $\text{BLEU}$ 类似，通过统计生成的摘要与参考摘要集合之间重叠的基本单元（$n$ 元组）的数目来评估摘要的质量，该方法已成为自动文摘系统评价的主流方法。

常见的 $\text{ROUGE}$ 评价指标有 $\text{ROUGE-N}$ 和 $\text{ROUGE-L}$。其中，$\text{ROUGE-N}$ 计算生成的摘要与参考摘要的 $\text{n-gram}$ 召回率，通常用 $\text{ROUGE-1/2}$ 来评估，而 $\text{ROUGE-L}$ 则匹配两个文本单元之间的最长公共序列 (LCS, Longest Common Sub sequence)。

$\text{ROUGE-N}$ 的计算方法非常简单，就是计算 $n$ 元组的召回率：

$$
\text{ROUGE-N} = \frac{\sum_{S\in\{\text{Ref Summaries}\}}\sum_{\text{n-gram}\in S}Count_{match}(\text{n-gram})}{\sum_{S\in\{\text{Ref Summaries}\}}\sum_{\text{n-gram}\in S}Count(\text{n-gram})}
$$

其中 $\{\text{Ref Summaries}\}$ 表示参考摘要集合，$Count_{match}(\text{n-gram})$ 表示生成的摘要和参考摘要中同时出现的 $\text{n-gram}$ 的个数，$Count(\text{n-gram})$ 则表示参考摘要中出现的 $\text{n-gram}$ 个数。因此分子就是生成摘要与参考摘要相匹配的 $\text{n-gram}$ 个数，分母就是参考摘要中所有的 $\text{n-gram}$ 个数。

$\text{Rouge-L}$ 使用了最长公共子序列，L 即是 LCS (longest common subsequence, 最长公共子序列) 的首字母，其计算方式如下：

$$
\begin{align}
R_{lcs} &= \frac{LCS(X,Y)}{m} \\
P_{lcs} &= \frac{LCS(X,Y)}{n} \\
F_{lcs} &= \frac{(1+\beta^2)R_{lcs}P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}}
\end{align}
$$

其中 $LCS(X,Y)$ 是 $X$ 和 $Y$ 的最长公共子序列的长度，$m,n$ 分别表示参考摘要和生成摘要的长度（一般就是所含词的个数），$R_{lcs},P_{lcs}$ 分别表示召回率和准确率。最后的 $F_{lcs}$ 即是我们所说的 $\text{ROUGE-L}$。通常 $\beta$ 被设置为一个很大的数，所以 $\text{ROUGE-L}$ 几乎只考虑了 $R_{lcs}$，也就是召回率。

## 计算 BLEU 分数

NLTK 自然语言工具包库提供了 $\text{BLEU}$ 评分的实现，你可以使用它来评估生成的文本。

### 语句 BLEU 分数

NLTK 提供了 `sentence_bleu()` 函数，用于根据一个或多个参考语句来评估候选语句。参考语句必须为语句列表，其中每个语句是一个词语列表。候选语句则直接是一个词语列表。例如：

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)

# > 1.0
```

### 语料库 BLEU 分数

NLTK 还提供了一个称为 `corpus_bleu()` 的函数来计算多个句子的 $\text{BLEU}$ 分数。

参考文档必须为文档列表，其中每个文档是一个参考语句列表，每个参考语句是一个词语列表。候选文档必须被指定为句子列表，每个句子是一个词语列表。以下是一个候选文档、两个参考文档的例子。

```python
# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score)

# > 1.0
```

运行这个例子就像之前一样输出满分

### 设置 N-Gram 权重

NLTK 中提供的 $\text{BLEU}$ 评分方法还允许你在计算 $\text{BLEU}$ 分数时为不同的 $n$ 元组指定不同的权重。权重被指定为一个数组，其中每个索引对应相应次序的 $n$ 元组，然后通过计算加权几何平均值来对它们进行加权计算。默认情况下，`sentence_bleu()` 和 `corpus_bleu()` 分数计算累加的 4 元组 $\text{BLEU}$ 分数，也称为 $\text{BLEU-4}$ 分数。

例如 $\text{BLEU-4}$ 对 1 元组，2 元组，3 元组和 4 元组分数的权重都为 $\frac{1}{4}$：

```python
# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)

# > 0.707106781187
```

累加的 1 元组和单独的 1 元组 $\text{BLEU}$ 使用相同的权重，也就是 `(1,0,0,0)`。累加的 2 元组 $\text{BLEU}$ 分数为 1 元组和 2 元组分别赋 $50％$ 的权重，计算累加的 3 元组 BLEU 为 1 元组，2 元组和 3 元组分别为赋 $33％$ 的权重。

例如计算 $\text{BLEU-1},\text{BLEU-2},\text{BLEU-3},\text{BLEU-4}$ 的累加得分：

```python
# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

# > Cumulative 1-gram: 0.750000
# > Cumulative 2-gram: 0.500000
# > Cumulative 3-gram: 0.632878
# > Cumulative 4-gram: 0.707107
```

可以看到结果的差别很大，比单独的 $\text{n-gram}$ 分数更具有表达性。在描述文本生成系统的性能时，通常会报告从 $\text{BLEU-1}$ 到 $\text{BLEU-4}$ 的累加分数。

## 计算 ROUGE 值

考虑到实践中最经常计算的就是 $\text{ROUGE-1}$、$\text{ROUGE-2}$ 和 $\text{ROUGE-L}$ 这三个评估指标，因此简单地使用 rouge 库就可以了，它虽然只能计算 $\text{ROUGE 1/2/L}$，但是十分方便。

先通过 `pip install rouge` 安装，然后就可以直接调用 `Rouge.get_scores()` 计算了：

```python
from rouge import Rouge

candidate = ['i am a student from xx school']  # 预测摘要, 可以是列表也可以是句子
reference = ['i am a student from school on china'] #真实摘要

rouge = Rouge()
rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
print(rouge_score[0]["rouge-1"])
print(rouge_score[0]["rouge-2"])
print(rouge_score[0]["rouge-l"])

# > {'f': 0.7999999950222222, 'p': 0.8571428571428571, 'r': 0.75}
# > {'f': 0.6153846104142012, 'p': 0.6666666666666666, 'r': 0.5714285714285714}
# > {'f': 0.7929824561399953, 'p': 0.8571428571428571, 'r': 0.75}

candidates = ['i am a student from xx school', 'happy new year']
references = ['i am a student from school on china', 'happy birthday']

rouge_score = rouge.get_scores(hyps=candidates, refs=references, avg=True)
print(rouge_score["rouge-1"])
print(rouge_score["rouge-2"])
print(rouge_score["rouge-l"])

# > {'r': 0.625, 'p': 0.5952380952380952, 'f': 0.5999999951111111}
# > {'r': 0.2857142857142857, 'p': 0.3333333333333333, 'f': 0.3076923052071006}
# > {'r': 0.625, 'p': 0.5952380952380952, 'f': 0.5999999951111111}
```

结果会输出在参考摘要的 $\text{1-gram}$ 和 $\text{2-gram}$ 的准确率、召回率和 F1 值。$\text{ROUGE-L}$ 类似，最长公共子序列在生成的摘要中所占比例是准确率，在参考摘要中所占比例是召回率，然后可以计算出 F1 值。

**注意：**对于英文直接把句子输入即可，中文如果没有空格就无法识别词，所以得先分词之后再计算 $\text{ROUGE}$ 值。如果以字为单位也得每个两个字中间加一个空格。

## 参考

[[1]](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/) A Gentle Introduction to Calculating the BLEU Score for Text in Python  
[[2]](https://cloud.tencent.com/developer/article/1042161) 浅谈用Python计算文本BLEU分数  
[[3]](https://blog.csdn.net/lime1991/article/details/42521029) ROUGE评价方法详解(一）  
[[4]](https://blog.csdn.net/qq_25222361/article/details/78694617) 自动文摘评测方法：Rouge-1、Rouge-2、Rouge-L、Rouge-S  
[[5]](https://www.jianshu.com/p/2d7c3a1fcbe3) pyrouge和rouge，文本摘要评测方法库

