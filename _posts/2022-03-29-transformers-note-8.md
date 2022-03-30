---
layout: article
title: Hugging Face 的 Transformers 库快速入门（八）：文本摘要任务
tags:
    - Transformers
    - NLP
    - Pytorch
mathjax: true
sidebar:
  nav: transformers-note
---

本文我们将运用 Transformers 库来完成文本摘要任务。文本摘要是一项极具挑战性的任务，模型需要在理解文本语义的基础上，生成包含文本主要话题的连贯文本。

虽然在 [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=summarization&sort=downloads) 上已经有各种用于文本摘要的模型，但是它们大部分只能处理英文，因此本文将微调一个多语言文本摘要模型用于完成中文摘要任务：为新浪微博中的短文本新闻生成摘要。

## 文本摘要模型

实际上，文本摘要与[翻译](/2022/03/24/transformers-note-7.html)任务非常相似，摘要的过程可以视作是将长文本“翻译”为捕获关键信息的短文本。因此，大部分用于摘要的 Transformer 模型都采用 Encoder-Decoder 框架。

> 当然了，也有一些非  Encoder-Decoder 框架的摘要模型，例如 GPT 家族也可以通过 few-shot 学习完成文本摘要任务。

下面是一些目前流行的可用于摘要任务的模型：

- **[GPT-2](https://huggingface.co/gpt2-xl)：**虽然是自回归 (auto-regressive) 语言模型，但是可以通过在输入文本的末尾添加 `TL;DR` 来使 GPT-2 生成摘要；
- **[PEGASUS](https://huggingface.co/google/pegasus-large)：**与普通的语言模型通过预测被遮掩掉的词语来进行训练不同，PEGASUS 通过预测多句文本中被遮掩掉的句子来进行训练。由于预训练目标与摘要任务接近，因此 PEGASUS 在摘要任务上的表现很好；
- **[T5](https://huggingface.co/t5-base)：**将各种 NLP 任务都转换到 text-to-text 框架来完成的通用 Transformer 架构，例如要进行摘要任务，就只需在输入文本前添加 `summarize:` 前缀；
- **[mT5](https://huggingface.co/google/mt5-base)：**T5 的多语言版本，在多语言通用爬虫语料库 mC4 上预训练，覆盖 101 种语言；
- **[BART](https://huggingface.co/facebook/bart-base)：**包含一个 Encoder 和一个 Decoder stack 的 Transformer 架构，训练目标是重构损坏的输入，同时还结合了 BERT 和 GPT-2 的预训练方案；
- **[mBART-50](https://huggingface.co/facebook/mbart-large-50)：**BART 的多语言版本，在 50 种语言上进行了预训练。

本文我们将专注于微调基于 T5 的多语言 mT5 模型。T5 模型通过特定的模板前缀 (prompt prefix) 将各种 NLP 任务都转换到 text-to-text 框架进行预训练，例如摘要任务的前缀就是 `summarize:`，模型以前缀作为条件生成符合模板的文本，这使得一个模型就可以完成多种 NLP 任务：

![t5.png](/img/article/transformers-note-8/t5.png)

mT5 模型不使用前缀，但是具备 T5 模型大部分的多功能性，并且可以处理多种语言。

## 准备数据

我们选择大规模中文短文本摘要语料库 [LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html) 作为数据集，该语料基于新浪微博短文本新闻构建，规模超过 200 万，我们粗略地将新闻的标题作为摘要来训练文本摘要模型。这里我们直接使用[和鲸社区](https://www.heywhale.com/home)中用户处理好的 LCSTS 语料：

[https://www.heywhale.com/mw/dataset/5e6740a5f278cf002d5335b8](https://www.heywhale.com/mw/dataset/5e6740a5f278cf002d5335b8)

该语料已经划分好了训练集、验证集和测试集，分别包含 2400591 / 10666 / 1106 个样本，一行是一个“标题-正文”的组合，采用 `!=!` 分隔：

```
媒体融合关键是以人为本!=!受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
```

### 构建数据集

与之前一样，我们首先编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签。考虑到使用 LCSTS 完整的两百多万条样本进行训练耗时过长，这里我们只抽取训练集中的前 20 万条数据：

```python
from torch.utils.data import Dataset

max_dataset_size = 200000

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS('lcsts_tsv/data1.tsv')
valid_data = LCSTS('lcsts_tsv/data2.tsv')
```

下面我们输出数据集的尺寸，并且打印出一个训练样本：

```python
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))
```

```
train set size: 200000
valid set size: 10666
test set size: 1106
{'title': '修改后的立法法全文公布', 'content': '新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。'}
```

可以看到数据集按照我们的设置进行了划分。

### 数据预处理

接下来，我们就需要通过 `DataLoader` 库按 batch 从数据集中加载数据，将文本转换为模型可以接受的 token IDs。与[翻译](/2022/03/24/transformers-note-7.html)任务类似，我们需要运用分词器对原文和摘要都进行编码，这里我们选择 [BUET CSE NLP Group](https://csebuetnlp.github.io/) 提供的 [mT5 摘要模型](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum)：

```python
from transformers import AutoTokenizer

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

我们先尝试使用 mT5 tokenizer 对文本进行分词：

```python
inputs = tokenizer("我叫张三，在苏州大学学习计算机。")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids))
```

```
{'input_ids': [259, 3003, 27333, 8922, 2092, 261, 1083, 117707, 9792, 24920, 123553, 306, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
['▁', '我', '叫', '张', '三', ',', '在', '苏州', '大学', '学习', '计算机', '。', '</s>']
```

特殊的 Unicode 字符 `▁` 以及序列结束 token `</s>` 表明我们使用的是一个基于 Unigram 切分算法的 SentencePiece 分词器。Unigram 对于处理多语言语料库特别有用，它使得 SentencePiece 可以在不知道重音、标点符号以及没有空格分隔字符（例如中文）的情况下对文本进行分词。

与[翻译](/2022/03/24/transformers-note-7.html)任务类似，由于输入和标签都是文本，因此我们需要应用截断操作来防止分词结果超过模型最大的接受长度，这里我们同样使用分词器提供的 `as_target_tokenizer()` 函数来并行地对输入和标签进行分词。其他的数据预处理操作也与[翻译](/2022/03/24/transformers-note-7.html)任务类似，例如将标签序列中的 pad 字符设置为 -100 以便在计算损失时将它们忽略，通过模型自带的 `prepare_decoder_input_ids_from_labels` 函数对标签进行移位操作以生成 decoder input IDs：

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM

max_input_length = 512
max_target_length = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
```

由于本文直接使用 Transformers 库自带的 `AutoModelForSeq2SeqLM` 函数来构建模型，因此我们上面将每一个 batch 中的数据都处理为该模型可接受的格式：一个包含 `'attention_mask'`、`'input_ids'`、`'labels'` 和 `'decoder_input_ids'` 键的字典。

下面我们尝试打印出一个 batch 的数据，以验证是否处理正确：

```python
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)
```

```
dict_keys(['input_ids', 'attention_mask', 'decoder_input_ids', 'labels'])
batch shape: {
    'input_ids': torch.Size([4, 78]), 
    'attention_mask': torch.Size([4, 78]), 
    'decoder_input_ids': torch.Size([4, 23]), 
    'labels': torch.Size([4, 23])
}
{'input_ids': tensor([
        [   259,  46420,   1083,  73451,    493,   3582,  14219,  98660, 111234,
           9455,  10139,    261,  11688,  56462,   7031,  71079,  31324,  94274,
           2037, 203743,   9911,  16834,   1107,   6929,  31063,    306,   2372,
            891,    261, 221805,   1455,  31571, 118447,    493,  56462,   7031,
          71079, 124732,   3937,  23224,   2037, 203743,   9911, 199662,  22064,
          31063,    261,   7609,   5705,  18988, 160700, 154547,  43803,  40678,
           3519,    306,      1,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0],
        [   259, 101737,  36059,    261, 157186,  47685,   8854, 124583, 218664,
           5705,   8363,   7216,  30921,  27032,  59754, 127646,  62558,  98901,
            261,   8868,   4110,   5705,  73334,  25265,  26553,   4153,    261,
           7274,  58402,   5435,  12914,    591,   2991, 162028,  22151,   4925,
         157186,  34499, 101737,  36059,  14520,  11201,  89746,  11017,    261,
           3763,   8868, 157186,  47685,   8854, 150259,  90707,   4417,  35388,
           3751,   2037,   3763, 194391,  81024,    261, 124025, 239583,  72939,
            306,   4925,  28216,  11242,  51563,   3094,    261, 157186, 142783,
           8868,  51191,  43239,   3763,    306,      1],
        [   259,  13732,   5705, 165437,  36814,  29650,    261, 120834, 201540,
          64493,  36814,  69169,    306,  13381,   5859,  14456,  21562,  16408,
         201540,   9692,   1374, 116772,  35988,   2188,  36079, 214133,    261,
          13505,   9127,   2542, 161781, 101017,    261, 101737,  36059,   7321,
          14219,   7519,  21929,    460, 100987,    261,   9903,   5848,  72308,
         101017,    261,   2123,  19394, 164872,   5162, 125883,  21562,  43138,
          37575,  15937,  66211,   5162,   3377,    848,  27349,   2446, 198562,
         154832,    261,  11883,  65386,    353, 106219,    261,  27674,    939,
          76364,   5507,  31568,   9809,  54172,      1],
        [   259,  77554,   1193,  74380,    493,    590,    487,    538,    495,
         198437,   8041,   6907, 219169, 122000,  10220,  28426,   6994,  36236,
          74380,  30733,    306,  40921, 218505,   1083,   5685,  14469,   2884,
           1637, 198437,  17723,  94708,  22695,    306,  12267,   1374,  13733,
           1543, 224495, 164497,  17286, 143553,  30464, 198437,  17723, 113940,
         176540, 143553,    306,  36017,   1374,  13733,  13342,  88397,  94708,
          22695,    261,   1083,   5685,  14469,  10458,   9692,   4070,  13342,
         115813,  27385,    306,      1,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0]]), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0]]), 
 'decoder_input_ids': tensor([
        [     0,    259,  11688,  56462,   7031,  71079,  73451,   3592,   3751,
           9911,  17938,  16834,   1107,   6929,  31063,  63095,    291,      1,
              0,      0,      0,      0,      0],
        [     0,    259, 157186,  47685,   8854, 107850,  14520,  11201,  89746,
          11017,  10973,   2219, 239583,  72939, 108358,    267,   1597,  43239,
          11242,  51563,   3094,      1,      0],
        [     0,    259,  13732,   2123,  19394,  94689,   2029,  26544,  17684,
           4074,  33119,  62428,  76364,      1,      0,      0,      0,      0,
              0,      0,      0,      0,      0],
        [     0,    447,    487,    538,    495, 198437,   8041,   6907,  86248,
          74380, 100644,  12267,    338, 225859,    261,  40921,    353,   3094,
          53737,   1083,  16311,  58407,  23616]]), 
 'labels': tensor([
        [   259,  11688,  56462,   7031,  71079,  73451,   3592,   3751,   9911,
          17938,  16834,   1107,   6929,  31063,  63095,    291,      1,   -100,
           -100,   -100,   -100,   -100,   -100],
        [   259, 157186,  47685,   8854, 107850,  14520,  11201,  89746,  11017,
          10973,   2219, 239583,  72939, 108358,    267,   1597,  43239,  11242,
          51563,   3094,      1,   -100,   -100],
        [   259,  13732,   2123,  19394,  94689,   2029,  26544,  17684,   4074,
          33119,  62428,  76364,      1,   -100,   -100,   -100,   -100,   -100,
           -100,   -100,   -100,   -100,   -100],
        [   447,    487,    538,    495, 198437,   8041,   6907,  86248,  74380,
         100644,  12267,    338, 225859,    261,  40921,    353,   3094,  53737,
           1083,  16311,  58407,  23616,      1]])}
```

可以看到，DataLoader 按照我们设置的 batch size，每次对 4 个样本进行编码，并且在标签序列中，pad token 对应的索引都被设置为 -100。decoder input IDs 尺寸与标签序列完全相同，且通过向后移位在序列头部添加了特殊的“序列起始符”，例如第一个样本：

```
'labels': 
        [   259,  11688,  56462,   7031,  71079,  73451,   3592,   3751,   9911,
          17938,  16834,   1107,   6929,  31063,  63095,    291,      1,   -100,
           -100,   -100,   -100,   -100,   -100]
'decoder_input_ids': 
        [     0,    259,  11688,  56462,   7031,  71079,  73451,   3592,   3751,
           9911,  17938,  16834,   1107,   6929,  31063,  63095,    291,      1,
              0,      0,      0,      0,      0]
```

至此，数据预处理部分就全部完成了！

## 训练模型

本文直接使用 Transformers 库自带的 `AutoModelForSeq2SeqLM` 函数来构建模型，因此下面只需要实现 Epoch 中的”训练循环”和”验证/测试循环”。

### 优化模型参数

使用 `AutoModelForSeq2SeqLM` 构造的模型已经封装好了对应的损失函数，并且计算出的损失会直接包含在模型的输出 `outputs` 中，可以直接通过 `outputs.loss` 获得，因此训练循环为：

```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
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

验证/测试循环负责评估模型的性能。对于文本摘要任务，常用评估指标是 [ROUGE 值](https://en.wikipedia.org/wiki/ROUGE_(metric)) (short for Recall-Oriented Understudy for Gisting Evaluation)，它可以度量两个词语序列之间的词语重合率。ROUGE 值的召回率表示参考摘要在多大程度上被生成摘要覆盖，如果我们只比较词语，那么召回率就是：

$$
\text{Recall} = \frac{\text{Number of overlapping words}}{\text{Total number of words in reference summary}}
$$

准确率则表示生成的摘要中有多少词语与参考摘要相关：

$$
\text{Precision}=\frac{\text{Number of overlapping words}}{\text{Total number of words in generated summary}} 
$$

最后再基于准确率和召回率来计算 F1 值。实际操作中，我们可以通过 [rouge 库](https://github.com/pltrdy/rouge)来方便地计算这些 ROUGE 值，例如 ROUGE-1 度量 unigrams 的重合情况，ROUGE-2 度量 bigrams 的重合情况，而 ROUGE-L 则通过在生成摘要和参考摘要中寻找最长公共子串来度量最长的单词匹配序列，例如：

```python
from rouge import Rouge

generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"

rouge = Rouge()

scores = rouge.get_scores(
    hyps=[generated_summary], refs=[reference_summary]
)[0]
print(scores)
```

```
{
 'rouge-1': {'r': 1.0, 'p': 0.8571428571428571, 'f': 0.9230769181065088}, 
 'rouge-2': {'r': 0.8, 'p': 0.6666666666666666, 'f': 0.7272727223140496}, 
 'rouge-l': {'r': 1.0, 'p': 0.8571428571428571, 'f': 0.9230769181065088}
}
```

rouge 库默认使用空格进行分词，因此无法处理中文、日文等语言，最简单的办法是按字进行切分，当然也可以使用分词器分词后再进行计算，否则会计算出不正确的 ROUGE 值：

```python
from rouge import Rouge

generated_summary = "我在苏州大学学习计算机，苏州大学很美丽。"
reference_summary = "我在环境优美的苏州大学学习计算机。"

rouge = Rouge()

TOKENIZE_CHINESE = lambda x: ' '.join(x)

# from transformers import AutoTokenizer
# model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# TOKENIZE_CHINESE = lambda x: ' '.join(
#     tokenizer.convert_ids_to_tokens(tokenizer(x).input_ids, skip_special_tokens=True)
# )

scores = rouge.get_scores(
    hyps=[TOKENIZE_CHINESE(generated_summary)], 
    refs=[TOKENIZE_CHINESE(reference_summary)]
)[0]
print('ROUGE:', scores)
scores = rouge.get_scores(
    hyps=[generated_summary], 
    refs=[reference_summary]
)[0]
print('wrong ROUGE:', scores)
```

```
ROUGE: {
 'rouge-1': {'r': 0.75, 'p': 0.8, 'f': 0.7741935433922998}, 
 'rouge-2': {'r': 0.5625, 'p': 0.5625, 'f': 0.562499995}, 
 'rouge-l': {'r': 0.6875, 'p': 0.7333333333333333, 'f': 0.7096774143600416}
}
wrong ROUGE: {
 'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}
}
```

使用 `AutoModelForSeq2SeqLM` 构造的模型对 Decoder 的解码过程进行了封装，因此只需要调用模型的 `generate()` 函数就可以自动地逐个生成预测 token。例如，我们可以直接调用预训练好的 mT5 摘要模型生成摘要：

```python
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

article_text = """
受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。
媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。
这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
"""

input_ids = tokenizer(
    article_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
generated_tokens = model.generate(
    input_ids["input_ids"],
    attention_mask=input_ids["attention_mask"],
    max_length=32,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]
summary = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(summary)
```

```
Using cpu device
媒体融合发展是当下中国面临的一大难题。
```

当然了，摘要多个句子也没有问题：

```python
article_texts = [
"""
受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。
媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。
这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
""",
"""
新华社受权于18日全文播发修改后的《中华人民共和国立法法》，
修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、
自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。
"""
]

input_ids = tokenizer(
    article_texts,
    padding=True, 
    return_tensors="pt",
    truncation=True,
    max_length=512
)
generated_tokens = model.generate(
    input_ids["input_ids"],
    attention_mask=input_ids["attention_mask"],
    max_length=32,
    no_repeat_ngram_size=2,
    num_beams=4
)
summarys = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(summarys)
```

```
[
 '媒体融合发展是当下中国面临的一大难题。', 
 '中国官方新华社周一(18日)全文播发修改后的《中华人民共和国立法法》。'
]
```

因此，在验证/测试循环中，我们首先通过 `model.generate()` 函数获取预测结果，然后将预测结果和正确标签都处理为 rouge 库接受的文本列表格式，并且将标签序列中的 -100 替换为 pad token 的 ID 以便于分词器解码，最后送入到 rouge 库计算各项 ROUGE 值：

```python
import numpy as np
from rouge import Rouge

rouge = Rouge()

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=4,
                no_repeat_ngram_size=2,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
    scores = rouge.get_scores(hyps=preds, refs=labels)[0]
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"{mode} Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result
```

为了方便后续保存验证集上最好的模型，我们还在验证/测试循环中返回评估出的 ROUGE 值。

###  保存和加载模型

与之前一样，我们会根据模型在验证集上的性能来调整超参数以及选出最好的模型，然后将选出的模型应用于测试集进行评估。这里我们继续使用 AdamW 优化器，并且通过 `get_scheduler()` 函数定义学习率调度器：

```python
from transformers import AdamW, get_scheduler

learning_rate = 2e-5
epoch_num = 10

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_rouge = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model, mode='Valid')
    print(valid_rouge)
    rouge_avg = valid_rouge['avg']
    if rouge_avg > best_avg_rouge:
        best_avg_rouge = rouge_avg
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
print("Done!")
```

在开始训练之前，我们先评估一下没有经过微调的模型在 LCSTS 测试集上的性能。

```python
test_data = LCSTS('lcsts_tsv/data3.tsv')
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collote_fn)

test_loop(test_dataloader, model, mode='Test')
```

```
Using cuda device
100%|█████████████████████████| 139/139 [03:06<00:00,  1.34s/it]
Test Rouge1: 20.00 Rouge2: 14.29 RougeL: 20.00
```

可以看到 ROUGE-1、ROUGE-2、ROUGE-L 值分别为 20.00、14.29 和 20.00，说明该模型虽然具备文本摘要的能力，但是在我们的“短文本新闻摘要”任务上表现不佳。然后，我们正式开始训练，完整代码如下：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from rouge import Rouge
import random
import numpy as np
import os

max_dataset_size = 200000
max_input_length = 512
max_target_length = 32
train_batch_size = 8
test_batch_size = 8
learning_rate = 2e-5
epoch_num = 3
beam_size = 4
no_repeat_ngram_size = 2

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS('lcsts_tsv/data1.tsv')
valid_data = LCSTS('lcsts_tsv/data2.tsv')
test_data = LCSTS('lcsts_tsv/data3.tsv')

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, collate_fn=collote_fn)

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
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

rouge = Rouge()

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
    scores = rouge.get_scores(hyps=preds, refs=labels)[0]
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"{mode} Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_rouge = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model, mode='Valid')
    rouge_avg = valid_rouge['avg']
    if rouge_avg > best_avg_rouge:
        best_avg_rouge = rouge_avg
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
print("Done!")
```

```
Using cuda device
Epoch 1/3
-------------------------------
loss: 3.322389: 100%|███████| 25000/25000 [1:24:25<00:00,  4.94it/s]
100%|█████████████████████████| 1334/1334 [21:04<00:00,  1.05it/s]
Valid Rouge1: 13.33 Rouge2: 0.00 RougeL: 6.67

saving new weights...

Epoch 2/3
-------------------------------
loss: 3.180292: 100%|███████| 25000/25000 [1:24:04<00:00,  4.96it/s]
100%|█████████████████████████| 1334/1334 [20:38<00:00,  1.08it/s]
Valid Rouge1: 13.33 Rouge2: 0.00 RougeL: 6.67

Epoch 3/3
-------------------------------
loss: 3.090493: 100%|███████| 25000/25000 [1:25:05<00:00,  4.90it/s]
100%|█████████████████████████| 1334/1334 [21:06<00:00,  1.05it/s]
Valid Rouge1: 13.33 Rouge2: 0.00 RougeL: 6.67
```

可以看到，3 轮训练中，模型在验证集上的 ROUGE 值没有发生变化，首轮训练后模型就已经收敛，达到了最好的性能。因此，3 轮训练结束后，目录下只保存了首轮训练后的模型权重：

```
epoch_1_valid_rouge_6.6667_model_weights.bin
```

最后，我们加载这个模型权重，再次评估模型在测试集上的性能：

```python
test_data = LCSTS('lcsts_tsv/data3.tsv')
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, collate_fn=collote_fn)

model.load_state_dict(torch.load('epoch_1_valid_rouge_6.6667_model_weights.bin'))
test_loop(test_dataloader, model, mode='Test')
```

```
Using cuda device
100%|█████████████████████████| 139/139 [02:08<00:00,  1.08it/s]
Test Rouge1: 70.00 Rouge2: 55.56 RougeL: 70.00
```

可以看到，经过我们的微调，模型在 ROUGE-1、ROUGE-2 和 ROUGE-L 值上分别提升了 50.00、41.27 和 50.00，证明了我们对模型的微调是成功的。

> 前面我们只保存了模型的权重（并没有同时保存模型结构），因此如果要单独调用上面的代码，需要首先实例化一个结构完全一样的模型，再通过 `model.load_state_dict()` 函数加载权重。

至此，我们使用 Transformers 库进行文本摘要任务就全部完成了！

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[3]](https://huggingface.co/docs/transformers/index) Transformers 官方文档
