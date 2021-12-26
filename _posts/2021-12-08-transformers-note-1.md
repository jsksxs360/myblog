---
layout: article
title: Hugging Face 的 Transformers 库快速入门（一）：开箱即用的 pipelines
tags:
    - Transformers
    - NLP
    - Pytorch
    - 机器学习
mathjax: true
sidebar:
  nav: transformers-note
---

[Transformers 库](https://huggingface.co/docs/transformers/index)是由 [Hugging Face](https://huggingface.co/) 开发的一个 NLP 包，支持几乎所有主流的预训练模型。随着深度学习的兴起，越来越多的公司和研究者采用 Transformers 库来进行 NLP 应用开发和研究，因此了解 Transformers 库的使用方法很有必要。

> 注：本文只专注于纯文本模态，多模态的相关使用方法请查阅[相关文档](https://huggingface.co/docs/transformers/index)。

## 开箱即用的 pipelines

Transformers 库将目前的 NLP 任务归纳为几下几类：

- **文本分类：**例如情感分析、垃圾邮件识别、“句子对”关系判断等等；
- **对文本中的每一个词语进行分类：**例如语法组件识别（名词、动词、形容词）或者命名实体识别（人物、地点、组织）；
- **文本生成：**例如自动填充预设的模板 (prompt)、预测文本中被 mask 掉（遮掩掉）的词语；
- **从文本中抽取答案：**例如根据给定的问题从一段文本中抽取出对应的答案；
- **根据输入文本生成新的句子：**例如文本翻译、自动摘要。

Transformers 库最基础的对象就是 `pipeline()` 函数，它封装了模型和对应的前处理和后处理环节，这样我们只需输入文本，就能得到预期的答案。目前常用的 [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) 有：

- `feature-extraction` (get the vector representation of a text)
- `fill-mask`
- `ner` (named entity recognition)
- `question-answering`
- `sentiment-analysis`
- `summarization`
- `text-generation`
- `translation`
- `zero-shot-classification`

下面我们以几个常见的 NLP 任务为例，演示一下如何调用 pipeline 模型。

### 情感分析

我们只需输入一段文本（或者多段），就能得到其情感极性（积极/消极）和对应的概率：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
results = classifier(
  ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(results)
```

```
[{'label': 'POSITIVE', 'score': 0.9598050713539124}]
[{'label': 'POSITIVE', 'score': 0.9598050713539124}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
```

pipeline 模型会自动完成以下三个步骤：

1. 将文本预处理为模型可以理解的格式；
2. 将预处理好的文本送入模型；
3. 对模型的预测值进行后处理，输出人类容易理解的格式。

> pipeline 会自动选择合适的预训练模型来完成指定的任务，例如对于情感分析，默认就会选择一个在情感分析任务上微调好的英文预训练模型。Transformers 库会在创建对象时下载并且缓存模型，只有在首次加载模型时才会下载，后续会直接调用缓存好的模型。

### 零样本分类

`zero-shot-classification` pipeline 允许我们自定义分类标签，而不用依赖预训练模型的标签。

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
"This is a course about the Transformers library",
candidate_labels=["education", "politics", "business"],
)
print(result)
```

```
{'sequence': 'This is a course about the Transformers library', 
 'labels': ['education', 'business', 'politics'], 
 'scores': [0.8445990085601807, 0.11197411268949509, 0.04342690855264664]}
```

> 这个 pipeline 之所以被称为零样本 (Zero-shot)，因为它不需要在我们自己的数据上进行微调，就能够直接返回我们指定标签的概率。

### 文本生成

我们提供一个模板 (prompt)，然后模型通过生成剩余的文本来填充模板。文本生成具有随机性，因此每次运行都会得到不同的结果。

```python
from transformers import pipeline

generator = pipeline("text-generation")
results = generator("In this course, we will teach you how to")
print(results)
results = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=50
) 
print(results)
```

```
[{'generated_text': 'In this course, we will teach you how to use our codebase and how to get on Stack Overflow with a little learning.\n\nWhy choose this course?\n\nStack Overflow is a place that many people come to for advice,'}]
[{'generated_text': 'In this course, we will teach you how to work with the latest technologies to create stunning visual effects. When you are ready to test your ideas to the world and become even more sophisticated, we will be holding a series of workshops where you take time'}, 
 {'generated_text': 'In this course, we will teach you how to make your own custom app.\n\nRequirements\n\nThere are three main platforms (Windows, Mac & Linux) for building applications:\n\nBuilding iOS App\n\nWe will make our own App'}]
```

上面的这些例子中 pipeline 都使用了默认的模型，我们也可以手工指定要使用的模型。对于文本生成任务，我们可以在 [Model Hub](https://huggingface.co/models) 页面左边选择 [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) tag 查询支持的模型。例如，我们在相同的 pipeline 中加载 [distilgpt2](https://huggingface.co/distilgpt2) 模型：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
results = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(results)
```

```
[{'generated_text': 'In this course, we will teach you how to get started with this course – learn how to get started in order to have a successful career as an'}, 
 {'generated_text': 'In this course, we will teach you how to apply the principles of B-C and B6 for more advanced applications, and will be sure to'}]
```

还可以通过左边的语言 tag 选择其他语言的用于文本生成的预训练模型。例如，也可以加载专门用于中文古诗的 [gpt2-chinese-poem](https://huggingface.co/uer/gpt2-chinese-poem) 模型：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
results = generator(
    "[CLS] 万 叠 春 山 积 雨 晴 ，",
    max_length=40,
    num_return_sequences=2,
)
print(results)
```

```
[{'generated_text': '[CLS] 万 叠 春 山 积 雨 晴 ， 孤 舟 遥 送 子 陵 行 。 别 情 共 叹 孤 帆 远 ， 交 谊 深 怜 一 座 倾 。 白 日 风 波 身 外 幻'}, 
 {'generated_text': '[CLS] 万 叠 春 山 积 雨 晴 ， 满 川 烟 草 踏 青 行 。 何 人 唤 起 伤 春 思 ， 江 畔 画 船 双 橹 声 。 桃 花 带 雨 弄 晴 光'}]
```

### 遮盖词填充

这个任务就是给定一段文本，其中一些词语被 Mask 掉（遮盖掉）了，需要填充这些词语。、

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
results = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(results)
```

```
[{'sequence': 'This course will teach you all about mathematical models.', 
  'score': 0.19619858264923096, 
  'token': 30412, 
  'token_str': ' mathematical'}, 
 {'sequence': 'This course will teach you all about computational models.', 
  'score': 0.04052719101309776, 
  'token': 38163, 
  'token_str': ' computational'}]
```

### 命名实体识别

命名实体识别 (NER) 任务负责找出文本中指定类型的实体，例如人物、地点、组织等等。

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(results)
```

```
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960186, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

可以看到模型正确地识别出了 Sylvain 是一个人物 (PER)，Hugging Face 是一个组织 (ORG)，Brooklyn 是一个地名。

> 通过设置参数 `grouped_entities=True`，使得 pipeline 自动合并属于同一个实体的多个子词，例如这里将“Hugging”和“Face”合并为一个组织实体，实际上 Sylvain 也进行了子词合并，因为分词器会将 Sylvain 切分为 `S`、`##yl` 、`##va` 和 `##in` 四个子词。

### 自动问答

自动问答 pipeline 根据给定的上下文回答问题，例如：

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
answer = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(answer)
```

```
{'score': 0.6949771046638489, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

> 注意这个 pipeline 是从给定的上下文中抽取信息，而不是生成答案。

### 自动摘要

自动摘要任务负责在尽可能保留主要信息的情况下，将长文本压缩成短文本，例如：

```python
from transformers import pipeline

summarizer = pipeline("summarization")
results = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
    """
)
print(results)
```

```
[{'summary_text': ' America has changed dramatically during recent years . The number of engineering graduates in the U.S. has declined in traditional engineering disciplines such as mechanical, civil,    electrical, chemical, and aeronautical engineering . Rapidly developing economies such as China and India continue to encourage and advance the teaching of engineering .'}]
```

与前面的文本生成类似，我们也可以通过 `max_length` 或 `min_length` 参数控制返回摘要的长度。

## 这些 pipeline 背后做了什么？

在介绍 Transformers 库的具体使用前，我们先来了解一下前一节中这些封装好的 pipeline 背后究竟进行了怎样的处理。以第一个情感分析 pipeline 为例，我们运行下面的代码

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
```

就会得到结果：

```
[{'label': 'POSITIVE', 'score': 0.9598048329353333}]
```

实际上这个 pipeline 在背后进行了三个步骤：预处理 (preprocessing)，将处理好的输入送入模型，对模型的输出进行后处理 (postprocessing)。

![full_nlp_pipeline](/img/article/transformers-note-1/full_nlp_pipeline.png)

### 使用分词器进行预处理

因为神经网络模型无法直接处理文本，因此首先需要通过预处理环节将文本转换为模型可以理解的数字。具体地，我们会使用一个分词器 (tokenizer)：

- 将输入切分为词语、子词或者符号（例如标点符号），这些统称为 tokens；
- 根据词表将每一个 token 映射到对应的数字；
- 添加一些模型需要的额外输入。

预处理环节需要与模型在预训练时对文本进行的操作完全一致，因此每一个模型的预处理环节都会有一些差异，可以通过 [Model Hub](https://huggingface.co/models) 查询每一个模型对应的信息。这里我们使用 `AutoTokenizer` 类和它的 `from_pretrained()`  函数，它可以自动根据模型 checkpoint 的名称来获取对应的分词器信息。

情感分析 pipeline 的默认 checkpoint 是 [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)，因此下载并调用其分词器如下所示：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

可以看到输出中包含两个键 `input_ids` 和 `attention_mask`，其中 `input_ids` 包含了句子分词之后 tokens 映射后的数字列表，而 `attention_mask` 则是用来标记哪些 tokens 是被填充的。

> 先不要关注 `padding`、`truncation` 这些参数，以及 `attention_mask`  项，后面我们会详细介绍。

### 将预处理好的输入送入模型

预训练模型的下载方式和分词器 (tokenizer) 类似，Transformers 包提供了一个 `AutoModel` 类和对应的 `from_pretrained()` 函数：

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

这个模型只包含基础的  Transformer 模块，对于给定的输入，它会输出一些神经元的值，称为 hidden states 或者特征 (features)，对于 NLP 任务来说，可以理解为文本的高维语义表示。这些 hidden states 通常会被输入到其他的模型部分，称为 head，以完成特定的任务。

> 上一节中举例的那些 pipelines 实际上具有相同的模型结构，只是模型的最后一部分会使用不同的 head 以完成对应的任务。
>
> ![transformer_and_head](/img/article/transformers-note-1/transformer_and_head.png)
>
> Transformers 库封装了很多不同的结构用于不同的任务，常见的有：
>
> - `*Model` （返回 hidden states）
> - `*ForCausalLM`
> - `*ForMaskedLM`
> - `*ForMultipleChoice`
> - `*ForQuestionAnswering`
> - `*ForSequenceClassification`
> - `*ForTokenClassification`

Transformer 模块的输出为一个维度为 (Batch size, Sequence length, Hidden size) 的三维张量，其中

- **Batch size**: 每次输入模型的序列数量，即每次输入多少个句子，上例中为 2；
- **Sequence length**: 序列的长度，即每个句子被切分为多少个 tokens，上例中为 16；
- **Hidden size**: 每一个 token 经过模型编码后的输出向量的维度。

> 目前预训练模型编码后的输出向量的维度都很大，例如常见的 Bert 模型 base 版本的输出为 768 维，一些更大的模型的输出维度为 3072 甚至更高。

对于上面使用的模型，我们可以打印出它的输出维度：

```python
from transformers import AutoTokenizer, AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

```
torch.Size([2, 16, 768])
```

Transformers 模型的输出格式类似 `namedtuple` 或字典，可以像上面那样通过属性访问，也可以通过键（`outputs["last_hidden_state"]`），甚至直接通过索引访问（`outputs[0]`）。

对于情感分析任务，很明显我们最后需要使用的是一个序列分类 head，因此实际上我们不会使用 `AutoModel` 类，而是使用  `AutoModelForSequenceClassification`：

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)
```

```
torch.Size([2, 2])
```

可以看到，模型的输出维度只有两维（每一维对应一个标签）。

### 对模型输出进行后处理

模型的输出只是一些数值，并不适合人类理解，例如我们打印出上面例子的输出：

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

```
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
```

模型对第一个句子输出 $[-1.5607, 1.6123]$，对第二个句子输出 $[ 4.1692, -3.3464]$，可以看到它们并不是概率值，而是模型最后一层输出的 logits 值。要将他们转换为概率值，还需要让它们经过一个 [SoftMax](https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0) 层，例如：

```python
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
```

> 所有 Transformers 模型都会输出 logits 值，因为训练的损失函数通常会自动结合激活函数（例如 SoftMax）与实际的损失函数（例如交叉熵 cross entropy）。

这样模型的预测结果就是容易理解的概率值：第一个句子 $[0.0402, 0.9598]$，第二个句子 $[0.9995, 0.0005]$。为了得到对应位置的标签，可以读取模型 config 中提供的 id2label 属性：

```python
print(model.config.id2label)
```

```
{0: 'NEGATIVE', 1: 'POSITIVE'}
```

于是我们可以得到最终的预测结果：

- 第一个句子: NEGATIVE: 0.0402, POSITIVE: 0.9598
- 第二个句子: NEGATIVE: 0.9995, POSITIVE: 0.0005

在本文中我们初步了解了如何使用 Transformers 包提供的 pipeline 对象来处理各种 NLP 任务，并且对 pipeline 背后的工作原理进行了介绍。在下面一篇文章中，我们会具体地介绍 pipeline 中包含的各个组件的参数以及使用方式。

## 参考

[1] [Transformers 官方文档](https://huggingface.co/docs/transformers/index)  
[2] [HuggingFace 在线教程](https://huggingface.co/course/chapter1/1)
