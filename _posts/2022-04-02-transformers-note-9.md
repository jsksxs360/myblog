---
layout: article
title: Hugging Face 的 Transformers 库快速入门（九）：抽取式问答
tags:
    - Transformers
    - NLP
    - Pytorch
mathjax: true
sidebar:
  nav: transformers-note
---

本文我们将运用 Transformers 库来完成抽取式问答任务。自动问答 (Question Answering, QA) 是经典的 NLP 任务，需要模型基于给定的上下文回答问题，根据产生回答方式的不同可以分为：

- **抽取式 (extractive) 问答：**从上下文中截取片段作为回答，类似于前面介绍的[序列标注](/2022/03/18/transformers-note-6.html)任务；
- **生成式 (generative) 问答：**生成一个文本片段作为回答，类似于[翻译](/2022/03/24/transformers-note-7.html)和[摘要](/2022/03/29/transformers-note-8.html)任务。

抽取式问答模型通常采用纯 Encoder 框架（例如 BERT），它更适用于处理事实性问题，例如“谁发明了 Transformer 架构？”；而生成式问答模型则通常采用 Encoder-Decoder 框架（例如 T5、BART），它更适用于处理开放式问题，例如“天空为什么是蓝色的？”。

本文我们将微调一个 BERT 模型来完成抽取式问答任务：对于每一个问题，从给定的上下文中抽取出概率最大的文本片段作为答案。

> 如果你对生成式问答感兴趣，可以参考 Hugging Face 提供的基于 [ELI5](https://huggingface.co/datasets/eli5) 数据库的 [Demo](https://yjernite.github.io/lfqa.html)。

## 准备数据

我们选择中文阅读理解语料库 [CMRC 2018](https://ymcui.com/cmrc2018/) 作为数据集，该语料是一个类似于 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 的抽取式数据集，可以从这里下载：

[https://github.com/ymcui/cmrc2018/tree/master/squad-style-data](https://github.com/ymcui/cmrc2018/tree/master/squad-style-data)

其中 *cmrc2018_train.json*、*cmrc2018_dev.json* 和 *cmrc2018_trial.json* 分别对应训练集、验证集和测试集。对于每篇文章，CMRC 2018 都标注了一些问题以及对应的答案（包括答案的文本和位置），例如：

```
{
 "context": "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。...", 
 "qas": [{
     "question": "《战国无双3》是由哪两个公司合作开发的？", 
     "id": "DEV_0_QUERY_0", 
     "answers": [{
         "text": "光荣和ω-force", 
         "answer_start": 11
     }, {
         "text": "光荣和ω-force", 
         "answer_start": 11
     }, {
         "text": "光荣和ω-force", 
         "answer_start": 11
     }]
 }, {
     "question": "男女主角亦有专属声优这一模式是由谁改编的？", 
     "id": "DEV_0_QUERY_1", 
     "answers": [{
         "text": "村雨城", 
         "answer_start": 226
     }, {
         "text": "村雨城", 
         "answer_start": 226
     }, {
         "text": "任天堂游戏谜之村雨城", 
         "answer_start": 219
     }]
 }, ...
 ]
}
```

一个问题可能对应有多个相同或不同的答案，在训练时我们只选择其中一个作为标签，在验证/测试时，我们则将预测答案和所有的参考答案都送入打分函数来评估模型的性能。

### 构建数据集

与之前一样，我们首先编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签。这里我们按照问题划分数据集（一个问题一个样本），问题对应的答案则处理为包含 `text` 和 `answer_start` 字段的字典，分别存储答案文本和位置：

```python
from torch.utils.data import Dataset
import json

class CMRC2018(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data['data']:
                title = article['title']
                context = article['paragraphs'][0]['context']
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']
                    ques = question['question']
                    text = [ans['text'] for ans in question['answers']]
                    answer_start = [ans['answer_start'] for ans in question['answers']]
                    Data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context, 
                        'question': ques,
                        'answers': {
                            'text': text,
                            'answer_start': answer_start
                        }
                    }
                    idx += 1
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = CMRC2018('cmrc2018/cmrc2018_train.json')
valid_data = CMRC2018('cmrc2018/cmrc2018_dev.json')
test_data = CMRC2018('cmrc2018/cmrc2018_trial.json')
```

下面我们输出数据集的尺寸，并且打印出一个训练样本：

```python
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))
```

```
train set size: 10142
valid set size: 3219
test set size: 1002
{
 'id': 'TRAIN_186_QUERY_0', 
 'title': '范廷颂', 
 'context': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；...', 
 'question': '范廷颂是什么时候被任为主教的？', 
 'answers': {
     'text': ['1963年'], 
     'answer_start': [30]
 }
}
```

可以看到训练集 / 验证集 / 测试集大小分别为 10142 / 3219 / 1002，并且样本处理为了我们预期的格式。因为可能会有多个答案，因此答案文本 `text` 和答案位置 `answer_start` 都是列表。

### 数据预处理

接下来，我们就需要通过 `DataLoader` 库按 batch 从数据集中加载数据，将文本转换为模型可以接受的 token IDs，并且构建对应的标签，标记答案在上下文中起始和结束位置。

本文使用 BERT 模型来完成任务，因此我们首先通过 checkpoint 加载对应的分词器：

```python
from transformers import AutoTokenizer

model_checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

> 可以将 checkpoint 更换为其他实现了快速分词器的模型，可以通过[该表](https://huggingface.co/docs/transformers/index#supported-frameworks)查询是否支持快速分词器。

正如之前在[自动问答任务](/2022/03/08/transformers-note-5.html#自动问答任务)中介绍的那样，我们会将问题和上下文共同编码为：

```
[CLS] question [SEP] context [SEP]
```

标签是答案在上下文中起始/结束 token 的索引，模型的任务就是预测每个 token 为答案片段的起始/结束的概率，即为每个 token 预测一个起始 logit 值和结束 logit 值。例如对于下面的文本，理想标签为：

![qa_labels](/img/article/transformers-note-9/qa_labels.svg)

我们在[问答 pipeline](/2022/03/08/transformers-note-5.html#处理长文本) 中就讨论过，由于上下文与问题拼接编码得到的 token 序列可能超过模型的最大输入长度，因此我们可以将长文切分为短文本块 (chunk) 来处理，同时为了缓解答案被截断的问题，我们使用滑窗使得切分出的文本块之间有重叠。

> 如果对分块操作感到陌生，可以参见快速分词器中的[处理长文本](/2022/03/08/transformers-note-5.html#处理长文本)小节，下面只做简单回顾。

例如我们编码第一个训练样本，将最大序列长度设为 300，滑窗大小设为 50：

- `max_length`：设置编码后的最大长度（这里设为 300）；
- `truncation="only_second"`：只对上下文进行分块（这里上下文是第二个输入）；
- `stride`：两个连续文本块之间重合 token 的数量（这里设为 50）；
- `return_overflowing_tokens=True`：设定分词器支持返回重叠 token。

```python
context = train_data[0]["context"]
question = train_data[0]["question"]

inputs = tokenizer(
    question,
    context,
    max_length=300,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```

```
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 范 廷 颂 枢 机 （ ， ） ， 圣 名 保 禄 · 若 瑟 （ ） ， 是 越 南 罗 马 天 主 教 枢 机 。 1963 年 被 任 为 主 教 ； 1990 年 被 擢 升 为 天 主 教 河 内 总 教 区 宗 座 署 理 ； 1994 年 被 擢 升 为 总 主 教 ， 同 年 年 底 被 擢 升 为 枢 机 ； 2009 年 2 月 离 世 。 范 廷 颂 于 1919 年 6 月 15 日 在 越 南 宁 平 省 天 主 教 发 艳 教 区 出 生 ； 童 年 时 接 受 良 好 教 育 后 ， 被 一 位 越 南 神 父 带 到 河 内 继 续 其 学 业 。 范 廷 颂 于 1940 年 在 河 内 大 修 道 院 完 成 神 学 学 业 。 范 廷 颂 于 1949 年 6 月 6 日 在 河 内 的 主 教 座 堂 晋 铎 ； 及 后 被 派 到 圣 女 小 德 兰 孤 儿 院 服 务 。 1950 年 代 ， 范 廷 颂 在 河 内 堂 区 创 建 移 民 接 待 中 心 以 收 容 到 河 内 避 战 的 难 民 。 1954 年 ， 法 越 战 争 结 束 ， 越 南 民 主 共 和 国 建 都 河 内 ， 当 时 很 多 天 主 教 神 职 人 员 逃 至 越 南 的 南 方 ， 但 范 廷 颂 仍 然 留 在 河 内 。 翌 年 [SEP]
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 越 战 争 结 束 ， 越 南 民 主 共 和 国 建 都 河 内 ， 当 时 很 多 天 主 教 神 职 人 员 逃 至 越 南 的 南 方 ， 但 范 廷 颂 仍 然 留 在 河 内 。 翌 年 管 理 圣 若 望 小 修 院 ； 惟 在 1960 年 因 捍 卫 修 院 的 自 由 、 自 治 及 拒 绝 政 府 在 修 院 设 政 治 课 的 要 求 而 被 捕 。 1963 年 4 月 5 日 ， 教 宗 任 命 范 廷 颂 为 天 主 教 北 宁 教 区 主 教 ， 同 年 8 月 15 日 就 任 ； 其 牧 铭 为 「 我 信 天 主 的 爱 」 。 由 于 范 廷 颂 被 越 南 政 府 软 禁 差 不 多 30 年 ， 因 此 他 无 法 到 所 属 堂 区 进 行 牧 灵 工 作 而 专 注 研 读 等 工 作 。 范 廷 颂 除 了 面 对 战 争 、 贫 困 、 被 当 局 迫 害 天 主 教 会 等 问 题 外 ， 也 秘 密 恢 复 修 院 、 创 建 女 修 会 团 体 等 。 1990 年 ， 教 宗 若 望 保 禄 二 世 在 同 年 6 月 18 日 擢 升 范 廷 颂 为 天 主 教 河 内 总 教 区 宗 座 署 理 以 填 补 该 教 区 总 主 教 的 空 缺 。 1994 年 3 月 23 日 [SEP]
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 若 望 保 禄 二 世 在 同 年 6 月 18 日 擢 升 范 廷 颂 为 天 主 教 河 内 总 教 区 宗 座 署 理 以 填 补 该 教 区 总 主 教 的 空 缺 。 1994 年 3 月 23 日 ， 范 廷 颂 被 教 宗 若 望 保 禄 二 世 擢 升 为 天 主 教 河 内 总 教 区 总 主 教 并 兼 天 主 教 谅 山 教 区 宗 座 署 理 ； 同 年 11 月 26 日 ， 若 望 保 禄 二 世 擢 升 范 廷 颂 为 枢 机 。 范 廷 颂 在 1995 年 至 2001 年 期 间 出 任 天 主 教 越 南 主 教 团 主 席 。 2003 年 4 月 26 日 ， 教 宗 若 望 保 禄 二 世 任 命 天 主 教 谅 山 教 区 兼 天 主 教 高 平 教 区 吴 光 杰 主 教 为 天 主 教 河 内 总 教 区 署 理 主 教 ； 及 至 2005 年 2 月 19 日 ， 范 廷 颂 因 获 批 辞 去 总 主 教 职 务 而 荣 休 ； 吴 光 杰 同 日 真 除 天 主 教 河 内 总 教 区 总 主 教 职 务 。 范 廷 颂 于 2009 年 2 月 22 日 清 晨 在 河 内 离 世 ， 享 年 89 岁 ； 其 葬 礼 于 同 月 26 日 上 午 在 天 主 教 河 内 总 教 区 总 主 教 座 堂 [SEP]
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 职 务 。 范 廷 颂 于 2009 年 2 月 22 日 清 晨 在 河 内 离 世 ， 享 年 89 岁 ； 其 葬 礼 于 同 月 26 日 上 午 在 天 主 教 河 内 总 教 区 总 主 教 座 堂 举 行 。 [SEP]
```

可以看到，对上下文的分块使得这个样本被切分为了 4 个新样本。

对于包含答案的样本，标签就是答案起始和结束 token 的索引；对于不包含答案或者由于只有部分答案的样本，其对应的标签都为 `start_position = end_position = 0`（即 `[CLS]`）。因此我们还需要设置分词器参数 `return_offsets_mapping=True`，这样就可以运用快速分词器提供的 offset mapping 映射得到对应的 token 索引。例如我们处理前 4 个训练样本：

```python
contexts = [train_data[idx]["context"] for idx in range(4)]
questions = [train_data[idx]["question"] for idx in range(4)]

inputs = tokenizer(
    questions,
    contexts,
    max_length=300,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

print(inputs.keys())
print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
```

```
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
The 4 examples gave 14 features.
Here is where each comes from: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3].
```

可以看到，通过设置 `return_overflowing_tokens` 和 `return_offsets_mapping`，编码结果中除了 input IDs、token type IDs 和 attention mask 以外，还返回了记录 token 到原文映射的 `offset_mapping`，以及记录分块样本到原始样本映射的 `overflow_to_sample_mapping`。这里 4 个原始样本被分块成了 14 个新样本，其中 0、3 样本被分块成了 4 个新样本，1、2 样本被分块成了 3 个新样本。

获得这两个映射之后，我们就可以方便地将答案标签映射到 token 索引了：

```python
answers = [train_data[idx]["answers"] for idx in range(4)]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # If the answer is not fully inside the context, label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # Otherwise it's the start and end token positions
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

print(start_positions)
print(end_positions)
```

```
[47, 0, 0, 0, 53, 0, 0, 100, 0, 0, 0, 0, 61, 0]
[48, 0, 0, 0, 70, 0, 0, 124, 0, 0, 0, 0, 106, 0]
```

> 为了找到 token 序列中上下文的索引范围，我们可以直接使用 token type IDs，但是一些模型（例如 DistilBERT）的分词器并不会输出该项，因此这里使用快速分词器返回 BatchEncoding 提供的 `sequence_ids()` 函数。

下面我们做个简单的验证，例如对于第一个新样本，可以看到处理后的答案标签为 `(47, 48)`，我们将对应的 token 解码并与标注答案进行对比：

```python
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")
```

```
Theoretical answer: 1963年, labels give: 1963 年
```

> **注意：**如果使用 XLNet 等模型，padding 操作会在序列左侧进行，并且问题和上下文也会调换，`[CLS]` 也可能不在 0 位置。

**训练批处理函数**

最后，我们合并上面的这些操作，编写对应于训练集的批处理函数：

```python
from torch.utils.data import DataLoader

max_length = 384
stride = 128

def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
        batch_answers.append(sample['answers'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )
    
    offset_mapping = batch_data.pop('offset_mapping')
    sample_map = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    batch_data['start_positions'] = start_positions
    batch_data['end_positions'] = end_positions
    return batch_data
 
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=train_collote_fn)
```

由于分块后的新样本长度都差不多，因此没有必要再按 batch 进行动态 padding，这里我们简单地将所有新样本都 pad 到设置的最大长度。

我们尝试打印出一个 batch 的数据，以验证是否处理正确，并且计算分块后新数据集的大小：

```python
import torch

batch = next(iter(train_dataloader))
batch = {k: torch.tensor(v) for k, v in batch.items()}
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)

print('train set size: ', )
print(len(train_data), '->', sum([len(batch_data['input_ids']) for batch_data in train_dataloader]))
```

```
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'])
batch shape: {
    'input_ids': torch.Size([14, 384]), 
    'token_type_ids': torch.Size([14, 384]), 
    'attention_mask': torch.Size([14, 384]), 
    'start_positions': torch.Size([14]), 
    'end_positions': torch.Size([14])
}
{'input_ids': tensor([
        [ 101,  100, 6858,  ...,    0,    0,    0],
        [ 101,  784,  720,  ..., 1184, 7481,  102],
        [ 101,  784,  720,  ..., 3341, 8024,  102],
        ...,
        [ 101, 4227, 2398,  ...,    0,    0,    0],
        [ 101, 5855, 2209,  ...,    0,    0,    0],
        [ 101, 3330, 1439,  ...,    0,    0,    0]]), 
 'token_type_ids': tensor([
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 1, 1, 1],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 
 'attention_mask': tensor([
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), 
 'start_positions': tensor(
        [ 98,  10,   0,   0,  62,   0, 132,   0,  44,   0,   0, 148,  20, 146]), 
 'end_positions': tensor(
        [100,  35,   0,   0,  65,   0, 140,   0,  54,   0,   0, 168,  45, 177])}

train set size: 
10142 -> 18960
```

可以看到，训练批处理函数成功生成了对应答案起始/结束索引的 `start_positions` 和 `end_positions`，这样后面我们就可以直接将编码后的数据送入 Transformers 库自带的 `AutoModelForQuestionAnswering` 模型进行训练。经过分块操作后，4 个原始样本被切分成了 14 个新样本，整个训练集的大小从 10142 增长到了 18960。

> 分块操作使得每一个 batch 处理后的大小参差不齐，即每次送入模型的样本数并不一致，这可能会影响模型的训练。更好地方式是为分块后的新样本重新建立一个 Dataset，然后按批加载新的数据集：
>
> ```python
> from transformers import default_data_collator
> 
> train_dataloader = DataLoader(
>     new_train_dataset,
>     shuffle=True,
>     collate_fn=default_data_collator,
>     batch_size=8,
> )
> ```

**验证/测试批处理函数**

对于验证/测试集，我们并不在意模型预测的标签，而是关注预测出的答案文本，这就需要：记录每个样本被分块成了哪几个新样本，从而合并对应的预测结果；在 offset mapping 中标记问题的对应 token，从而在后处理阶段可以区分哪些位置的 token 来自于上下文。

因此，对应于验证集/测试集的批处理函数为：

```python
def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_map = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_map[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = batch_data["offset_mapping"][i]
        batch_data["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    batch_data["example_id"] = example_ids
    return batch_data

valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False, collate_fn=test_collote_fn)
```

同样我们打印出一个 batch 编码后的数据，并且计算分块后新数据集的大小：

```python
batch = next(iter(valid_dataloader))
print(batch.keys())
print(batch['example_id'])

print('valid set size: ')
print(len(valid_data), '->', sum([len(batch_data['input_ids']) for batch_data in valid_dataloader]))
```

```
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id'])

['DEV_0_QUERY_0', 'DEV_0_QUERY_0', 'DEV_0_QUERY_1', 'DEV_0_QUERY_1', 'DEV_0_QUERY_2', 'DEV_0_QUERY_2', 'DEV_1_QUERY_0', 'DEV_1_QUERY_0']

valid set size: 
3219 -> 6254
```

可以看到，编码结果中除了 input IDs、token type IDs 和 attention mask 之外，还包含了我们处理后的 `offset_mapping` 以及记录分块样本对应 ID 的 `example_id`。经过分块操作后，整个测试集的样本数量从 3219 增长到了 6254。

至此，数据预处理部分就全部完成了！

## 训练模型

本文我们直接使用 Transformers 库自带的 `AutoModelForQuestionAnswering` 函数来构建模型，前面已经通过批处理函数将训练集处理成了特定格式，因此可以直接送入模型进行训练：

```python
from transformers import AutoModelForQuestionAnswering

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model = model.to(device)
```

### 训练循环

使用 `AutoModelForQuestionAnswering` 构造的模型已经封装好了对应的损失函数，计算出的损失会直接包含在模型的输出 `outputs` 中（可以通过 `outputs.loss` 获得），因此训练循环为：

```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = {k: torch.tensor(v).to(device) for k, v in batch_data.items()}
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss
```

### 后处理

因为最终是根据预测出的答案文本来评估模型的性能，所以在编写验证/测试循环之前，我们先讨论一下问答模型的后处理操作，即怎么将模型的预测结果转换为答案文本。

之前在[自动问答任务](/2022/03/08/transformers-note-5.html#自动问答任务)中已经介绍过，对每个样本，问答模型都会输出两个张量，分别对应答案起始/结束位置的 logits 值，我们回顾一下之前的后处理过程：

1. 遮盖掉除上下文之外的其他 token 的起始/结束 logits 值；

2. 通过 softmax 函数将起始/结束 logits 值转换为概率值；

3. 通过计算概率值的乘积估计每一对 `(start_token, end_token)` 为答案的分数；

4. 输出合理的（例如 `start_token` 要小于 `end_token`）分数最大的对作为答案。

本文我们会稍微做一些调整：首先，我们只关心最后预测出的答案文本，因此可以跳过 softmax 函数，直接基于 logits 值来估计答案分数，从原来计算概率值的乘积变成计算 logits 值的和（因为 $\log(ab) = \log(a) + \log(b)$）；其次，为了减少计算量，我们不再为所有可能的 `(start_token, end_token)` 对打分，而是只计算 logits 值最高的前 n_best 个 token  组成的对。

由于我们的 BERT 模型还没有进行微调，因此这里我们选择一个已经预训练好的问答模型 [Chinese RoBERTa-Base Model for QA](https://huggingface.co/uer/roberta-base-chinese-extractive-qa) 进行演示，并且只对验证集上的前 10 个样本进行处理：

```python
valid_data = CMRC2018('cmrc2018/cmrc2018_dev.json')
small_eval_set = [valid_data[idx] for idx in range(10)]

trained_checkpoint = "uer/roberta-base-chinese-extractive-qa"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = DataLoader(small_eval_set, batch_size=4, shuffle=False, collate_fn=test_collote_fn)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoModelForQuestionAnswering
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)
```

接下来，与之前任务中的验证/测试循环一样，在 `torch.no_grad()` 上下文管理器下，使用模型对所有分块后的新样本进行预测，并且汇总预测出的起始/结束 logits 值：

```python
start_logits = []
end_logits = []

trained_model.eval()
for batch_data in eval_set:
    del batch_data['offset_mapping']
    del batch_data['example_id']
    batch_data = {k: torch.tensor(batch_data[k]).to(device) for k in batch_data.keys()}
    with torch.no_grad():
        outputs = trained_model(**batch_data)
    start_logits.append(outputs.start_logits.cpu().numpy())
    end_logits.append(outputs.end_logits.cpu().numpy())

import numpy as np
start_logits = np.concatenate(start_logits)
end_logits = np.concatenate(end_logits)
```

在将预测结果转换为文本之前，我们还需要知道每个样本被分块为了哪几个新样本，从而汇总对应的预测结果，因此下面先构造一个记录样本 ID 到新样本索引的映射：

```python
all_example_ids = []
all_offset_mapping = []
for batch_data in eval_set:
    all_example_ids += batch_data['example_id']
    all_offset_mapping += batch_data['offset_mapping']

import collections
example_to_features = collections.defaultdict(list)
for idx, feature_id in enumerate(all_example_ids):
    example_to_features[feature_id].append(idx)

print(example_to_features)
```

```
defaultdict(<class 'list'>, {
 'DEV_0_QUERY_0': [0, 1], 'DEV_0_QUERY_1': [2, 3], 'DEV_0_QUERY_2': [4, 5], 
 'DEV_1_QUERY_0': [6, 7], 'DEV_1_QUERY_1': [8, 9], 'DEV_1_QUERY_2': [10, 11], 
 'DEV_1_QUERY_3': [12, 13], 'DEV_2_QUERY_0': [14, 15], 'DEV_2_QUERY_1': [16, 17], 
 'DEV_2_QUERY_2': [18, 19]})
```

接下来我们只需要遍历数据集中的样本，汇总由其分块出的新样本的预测结果，取出每个新样本最高的前 `n_best` 个起始/结束 logits 值，评估对应的 token 片段为答案的分数：

```python
n_best = 20
max_answer_length = 30
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = all_offset_mapping[feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                    continue
                answers.append(
                    {
                        "start": offsets[start_index][0],
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append({
            "id": example_id, 
            "prediction_text": best_answer["text"], 
            "answer_start": best_answer["start"]
        })
    else:
        predicted_answers.append({
            "id": example_id, 
            "prediction_text": "", 
            "answer_start": 0
        })
```

> 这里我们还通过限制答案的最大长度来进一步减小计算量。

下面我们同步打印出预测和标注的答案来进行对比：

```python
for pred, label in zip(predicted_answers, theoretical_answers):
    print(pred['id'])
    print('pred:', pred['prediction_text'])
    print('label:', label['answers']['text'])
```

```
DEV_0_QUERY_0
pred: 光荣和ω-force
label: ['光荣和ω-force', '光荣和ω-force', '光荣和ω-force']
DEV_0_QUERY_1
pred: 任天堂游戏谜之村雨城
label: ['村雨城', '村雨城', '任天堂游戏谜之村雨城']
...
```

可以看到，由于我们选择的 [Chinese RoBERTa-Base Model for QA](https://huggingface.co/uer/roberta-base-chinese-extractive-qa) 模型本身的预训练数据就包含了 [CMRC 2018](https://ymcui.com/cmrc2018/)，因此模型的预测结果还是不错的。

在成功获取到预测的答案片段之后，就可以对模型的性能进行评估了。这里我们对 CMRC 2018 自带的[评估脚本](https://github.com/ymcui/cmrc2018/blob/master/squad-style-data/cmrc2018_evaluate.py)进行修改，使其支持本文模型的输出格式：

```python
import re
import sys
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenize = lambda x: tokenizer(x).tokens()[1:-1]

# import nltk
# tokenize = lambda x: nltk.word_tokenize(x)

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                # ss = nltk.word_tokenize(temp_str)
                ss = tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        # ss = nltk.word_tokenize(temp_str)
        ss = tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax

def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision     = 1.0*lcs_len/len(prediction_segs)
        recall         = 1.0*lcs_len/len(ans_segs)
        f1             = (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)

def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

def evaluate(predictions, references):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    pred = dict([(data['id'], data['prediction_text']) for data in predictions])
    ref = dict([(data['id'], data['answers']['text']) for data in references])
    for query_id, answers in ref.items():
        total_count += 1
        if query_id not in pred:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue
        prediction = pred[query_id]
        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {
        'avg': (em_score + f1_score) * 0.5, 
        'f1': f1_score, 
        'em': em_score, 
        'total': total_count, 
        'skip': skip_count
    }
```

> 请将上面的代码存放在 *cmrc2018_evaluate.py* 文件中，后续直接使用其中的 `evaluate` 函数进行评估。

最后，我们将上面的预测结果送入 `evaluate` 函数进行评估：

```python
from cmrc2018_evaluate import evaluate

result = evaluate(predicted_answers, theoretical_answers)
print(f"F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
```

```
F1: 91.15 EM: 70.00 AVG: 80.58
```

### 测试循环

熟悉了后处理操作之后，编写验证/测试循环就很简单了，只需对上面的这些步骤稍作整合即可。这里由于我们还需要使用到样本的原始文本，因此将数据集也作为参数传入：

```python
import collections
from cmrc2018_evaluate import evaluate

n_best = 20
max_answer_length = 30

def test_loop(dataloader, dataset, model, mode='Test'):
    assert mode in ['Valid', 'Test']

    all_example_ids = []
    all_offset_mapping = []
    for batch_data in dataloader:
        all_example_ids += batch_data['example_id']
        all_offset_mapping += batch_data['offset_mapping']

    model.eval()
    start_logits = []
    end_logits = []
    for batch_data in tqdm(dataloader):
        del batch_data['offset_mapping']
        del batch_data['example_id']
        batch_data = {k: torch.tensor(batch_data[k]).to(device) for k in batch_data.keys()}
        with torch.no_grad():
            outputs = model(**batch_data)
        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)
    
    theoretical_answers = [
        {"id": dataset[idx]["id"], "answers": dataset[idx]["answers"]} for idx in range(len(dataset))
    ]
    predicted_answers = []
    for idx in tqdm(range(len(dataset))):
        example_id = dataset[idx]["id"]
        context = dataset[idx]["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
    result = evaluate(predicted_answers, theoretical_answers)
    print(f"{mode} F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
    return result
```

### 保存和加载模型

与之前一样，我们会根据模型在验证集上的性能来调整超参数以及选出最好的模型，然后将选出的模型应用于测试集进行评估。这里继续使用 AdamW 优化器，并且通过 `get_scheduler()` 函数定义学习率调度器：

```python
from transformers import AdamW, get_scheduler

learning_rate = 2e-5
epoch_num = 3

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_score = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_scores = test_loop(valid_dataloader, valid_data, model, mode='Valid')
    avg_score = valid_scores['avg']
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_avg_{avg_score:0.4f}_model_weights.bin')
print("Done!")
```

下面，我们正式开始训练，完整的训练代码如下：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AdamW, get_scheduler
import json
from tqdm.auto import tqdm
import collections
import random
import numpy as np
import os
import sys
sys.path.append('./')
from cmrc2018_evaluate import evaluate

max_length = 384
stride = 128
n_best = 20
max_answer_length = 30
batch_size = 4
learning_rate = 2e-5
epoch_num = 3

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class CMRC2018(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data['data']:
                title = article['title']
                context = article['paragraphs'][0]['context']
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']
                    ques = question['question']
                    text = [ans['text'] for ans in question['answers']]
                    answer_start = [ans['answer_start'] for ans in question['answers']]
                    Data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context, 
                        'question': ques,
                        'answers': {
                            'text': text,
                            'answer_start': answer_start
                        }
                    }
                    idx += 1
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = CMRC2018('cmrc2018/cmrc2018_train.json')
valid_data = CMRC2018('cmrc2018/cmrc2018_dev.json')
test_data = CMRC2018('cmrc2018/cmrc2018_trial.json')

model_checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model = model.to(device)

def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
        batch_answers.append(sample['answers'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )
    
    offset_mapping = batch_data.pop('offset_mapping')
    sample_map = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    batch_data['start_positions'] = start_positions
    batch_data['end_positions'] = end_positions
    return batch_data

def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_map = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_map[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = batch_data["offset_mapping"][i]
        batch_data["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    batch_data["example_id"] = example_ids
    return batch_data

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)

print('train set size: ', )
print(len(train_data), '->', sum([len(batch_data['input_ids']) for batch_data in train_dataloader]))
print('valid set size: ')
print(len(valid_data), '->', sum([len(batch_data['input_ids']) for batch_data in valid_dataloader]))

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = {k: torch.tensor(v).to(device) for k, v in batch_data.items()}
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, dataset, model, mode='Test'):
    assert mode in ['Valid', 'Test']

    all_example_ids = []
    all_offset_mapping = []
    for batch_data in dataloader:
        all_example_ids += batch_data['example_id']
        all_offset_mapping += batch_data['offset_mapping']

    model.eval()
    start_logits = []
    end_logits = []
    for batch_data in tqdm(dataloader):
        del batch_data['offset_mapping']
        del batch_data['example_id']
        batch_data = {k: torch.tensor(batch_data[k]).to(device) for k in batch_data.keys()}
        with torch.no_grad():
            outputs = model(**batch_data)
        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)
    
    theoretical_answers = [
        {"id": dataset[idx]["id"], "answers": dataset[idx]["answers"]} for idx in range(len(dataset))
    ]
    predicted_answers = []
    for idx in tqdm(range(len(dataset))):
        example_id = dataset[idx]["id"]
        context = dataset[idx]["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
    result = evaluate(predicted_answers, theoretical_answers)
    print(f"{mode} F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
    return result

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_score = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_scores = test_loop(valid_dataloader, valid_data, model, mode='Valid')
    avg_score = valid_scores['avg']
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_avg_{avg_score:0.4f}_model_weights.bin')
print("Done!")
```

```
Using cuda device

train set size: 
10142 -> 18960
valid set size: 
3219 -> 6254

Epoch 1/3
-------------------------------
loss: 1.381956: 100%|███████| 2536/2536 [08:19<00:00,  5.07it/s]
100%|█████████████████████████| 805/805 [00:50<00:00, 16.02it/s]
100%|█████████████████████████| 3219/3219 [00:00<00:00, 3730.55it/s]
Valid F1: 85.72 EM: 67.94 AVG: 76.83

saving new weights...

Epoch 2/3
-------------------------------
loss: 1.070632: 100%|███████| 2536/2536 [08:20<00:00,  5.07it/s]
100%|█████████████████████████| 805/805 [00:50<00:00, 16.01it/s]
100%|█████████████████████████| 3219/3219 [00:00<00:00, 3847.03it/s]
Valid F1: 83.61 EM: 63.00 AVG: 73.30

Epoch 3/3
-------------------------------
loss: 0.872092: 100%|███████| 2536/2536 [08:19<00:00,  5.07it/s]
100%|█████████████████████████| 805/805 [00:50<00:00, 16.00it/s]
100%|█████████████████████████| 3219/3219 [00:00<00:00, 3847.18it/s]
Valid F1: 84.41 EM: 63.68 AVG: 74.05

Done!
```

可以看到，随着训练的进行，模型在验证集上的性能先降后升，并且最后一轮也没有取得更好的结果。因此，3 轮训练结束后，目录下只保存了首轮训练后的模型权重：

```
epoch_1_valid_avg_76.8278_model_weights.bin
```

最后，我们加载这个模型权重，评估模型在测试集上的性能：

```python
test_data = CMRC2018('cmrc2018/cmrc2018_trial.json')
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)

model.load_state_dict(torch.load('epoch_1_valid_avg_76.8278_model_weights.bin'))
test_loop(test_dataloader, test_data, model, mode='Test')
```

```
Test F1: 66.93 EM: 30.34 AVG: 48.64
```

可以看到，最终问答模型在测试集上取得了 F1 值 66.93、EM 值 30.34 的结果，虽然不是太理想，但是证明了我们对模型的微调是成功的。

> 前面我们只保存了模型的权重（并没有同时保存模型结构），因此如果要单独调用上面的代码，需要首先实例化一个结构完全一样的模型，再通过 `model.load_state_dict()` 函数加载权重。

至此，我们使用 Transformers 库进行抽取式问答任务就全部完成了！

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[3]](https://huggingface.co/docs/transformers/index) Transformers 官方文档