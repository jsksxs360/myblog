---
layout: article
title: Hugging Face 的 Transformers 库快速入门（七）：翻译任务
tags:
    - Transformers
    - NLP
    - Pytorch
mathjax: true
sidebar:
  nav: transformers-note
---

本文我们将运用 Transformers 库来完成翻译任务。翻译是典型的 Seq2Seq (sequence-to-sequence) 任务，即对于给定的词语序列，输出一个对应的词语序列。翻译任务不仅与文本摘要任务很相似，而且我们可以将本文的操作应用于其他的 Seq2Seq 任务，例如：

- **风格转换 (Style transfer)：**将采用某种风格书写的文本转换为另一种风格，例如将文言文转换为白话文、将莎士比亚式英语转换为现代英语；
- **生成式问答 (Generative question answering)：**对于给定的问题，基于上下文生成对应的答案。

如果有足够多的语料，我们可以从头训练一个翻译模型，但是微调预训练好的翻译模型会更快，比如将 mT5、mBART 等多语言模型微调到特定的语言对。

本文我们将微调一个 Marian 翻译模型进行汉英翻译，该模型已经基于大规模的 [Opus](https://opus.nlpl.eu/) 语料库对汉英翻译任务进行了预训练，因此可以直接用于翻译。而通过我们的微调，可以进一步提升该模型在特定语料上的性能。

## 准备数据

我们选择 [translation2019zh](https://github.com/brightmart/nlp_chinese_corpus#5%E7%BF%BB%E8%AF%91%E8%AF%AD%E6%96%99translation2019zh) 语料库作为数据集，该语料共包含中英文平行语料 520 万对，因此可以用于训练中英文翻译模型。本文我们将基于该语料，微调一个预训练好的汉英翻译模型。其 Github 仓库中只提供了 [Google Drive](https://drive.google.com/open?id=1EX8eE5YWBxCaohBO8Fh4e2j3b9C2bTVQ) 链接，我们也可以通过[和鲸社区](https://www.heywhale.com/home)的镜像下载：

[https://www.heywhale.com/mw/dataset/5de5fcafca27f8002c4ca993/content](https://www.heywhale.com/mw/dataset/5de5fcafca27f8002c4ca993/content)

该语料已经划分好了训练集和验证集，分别包含 516 万和 3.9 万个样本，语料以 json 格式提供，一行是一个中英文对照句子对：

```
{"english": "In Italy, there is no real public pressure for a new, fairer tax system.", "chinese": "在意大利，公众不会真的向政府施压，要求实行新的、更公平的税收制度。"}
```

### 构建数据集

与之前一样，我们首先编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签。考虑到 translation2019zh 并没有提供测试集，而且使用五百多万条样本进行训练耗时过长，这里我们只抽取训练集中的前 22 万条数据，并从中划分出 2 万条数据作为验证集，然后将 translation2019zh 中的验证集作为测试集：

```python
from torch.utils.data import Dataset
import json

max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000

class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = TRANS('translation2019zh/translation2019zh_train.json')
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
test_data = TRANS('translation2019zh/translation2019zh_valid.json')
```

下面我们输出数据集的尺寸，并且打印出一个训练样本：

```python
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data))
```

```
train set size: 200000
valid set size: 20000
test set size: 39323
{'english': 'The robust RJ45 connectors are also shielded against electrical interference.', 'chinese': '强劲的RJ45连接器也屏蔽防止电波干扰。'}
```

可以看到数据集按照我们的设置进行了划分。

### 数据预处理

接着，我们就需要通过 `DataLoader` 库来按 batch 从数据集中加载数据，将文本转换为模型可以接受的 token IDs。对于翻译任务，我们需要运用分词器对源文本和目标文本都进行编码，这里我们选择 [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) 提供的汉英翻译模型对应的分词器：

```python
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

你也可以尝试别的语言，[Helsinki-NLP](https://huggingface.co/Helsinki-NLP) 提供了超过了一千种模型用于在不同语言之间进行翻译，只需要将 `model_checkpoint` 设置为对应的语言即可。如果你想使用多语言模型的分词器，例如 mBART、mBART-50、M2M100，就需要通过设置 `tokenizer.src_lang` 和 `tokenizer.tgt_lang` 来手工设定源/目标语言。

默认情况下分词器会采用源语言的设定来编码文本，因此对于目标语言，需要将其放入上下文管理器 (context manager) `as_target_tokenizer()`：

```python
zh_sentence = train_data[0]["chinese"]
en_sentence = train_data[0]["english"]

inputs = tokenizer(zh_sentence)
with tokenizer.as_target_tokenizer():
    targets = tokenizer(en_sentence)
```

如果你忘记了使用上下文管理器，就会使用源语言的分词器对目标语言进行编码，从而生成糟糕的分词结果：

```python
wrong_targets = tokenizer(en_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(targets["input_ids"]))
```

```
['▁The', '▁', 'ro', 'b', 'ust', '▁R', 'J', '45', '▁con', 'ne', 'c', 'tor', 's', '▁are', '▁al', 'so', '▁', 'shi', 'el', 'd', 'ed', '▁', 'aga', 'in', 'st', '▁', 'e', 'le', 'c', 'tri', 'c', 'al', '▁', 'inter', 'f', 'er', 'ence', '.', '</s>']

['▁The', '▁robust', '▁R', 'J', '45', '▁connect', 'ors', '▁are', '▁also', '▁shield', 'ed', '▁against', '▁electrical', '▁interference', '.', '</s>']
```

可以看到，由于中文分词器无法识别大部分的英文单词，用它处理英文文本会生成更多的 tokens，例如这里将"connect"切分为了"con"、"ne"、"c"等。

对于翻译任务，标签序列就是目标语言的 token ID 序列。与[序列标注任务](https://xiaosheng.run/2022/03/18/transformers-note-6.html)类似，我们会在模型预测出的类别序列与标签序列之间计算损失来调整模型参数，因此需要将填充的 pad 字符设置为 -100，以便在使用交叉熵计算序列损失时将它们忽略：

```python
import torch

model_inputs = tokenizer(
    inputs, 
    padding=True, 
    max_length=max_input_length, 
    truncation=True
)
with tokenizer.as_target_tokenizer():
    labels = tokenizer(
        targets, 
        padding=True, 
        max_length=max_target_length, 
        truncation=True
    )["input_ids"]

end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
for idx, end_idx in enumerate(end_token_index):
    labels[idx][end_idx+1:] = -100
```

我们使用的 Marian 模型会在分词结果的结尾加上特殊 token `'</s>'`，因此这里通过 `tokenizer.eos_token_id` 定位其在序列中的索引，然后将其之后的 pad 字符设置为 -100。

> 如果你使用的是 T5 模型（checkpoint 为 `t5-xxx`），模型的输入还需要包含指明任务类型的前缀 (prefix)，例如翻译任务就需要在输入前添加 `Chinese to English:`。

与我们之前任务中使用的纯 Encoder 模型不同，Seq2Seq 任务对应的模型采用的是 Encoder-Decoder 框架：Encoder 负责编码输入序列，Decoder 负责循环地逐个生成输出 token。因此，对于每一个样本，我们还需要额外准备 decoder input IDs 用作 Decoder 的输入。decoder input IDs 是标签序列的移位，在序列的开始位置增加了一个特殊的“序列起始符”。

在训练过程中，模型会基于 decoder input IDs 和 attention mask 来确保在预测某个 token 时不会使用到该 token 及其之后的 token 的信息。即在 Decoder 预测某个目标 token 时，只能基于“整个输入序列”和“当前已经预测出的目标 token”信息来进行预测，如果提前看到要预测的 token 甚至是再后面的 token，就相当于是“作弊”了（信息泄露）。因此在训练时，会通过特殊的“三角形” Mask 来遮掩掉预测 token 及其之后的 token 的信息。

> 如果对这一块感到困惑，可以参考苏剑林的博文[《从语言模型到Seq2Seq：Transformer如戏，全靠Mask》](https://kexue.fm/archives/6933)。

考虑到不同模型结构的移位操作可能存在差异，因此我们通过模型自带的 `prepare_decoder_input_ids_from_labels` 函数来完成。完整的批处理函数为：

```python
import torch
from transformers import AutoModelForSeq2SeqLM

max_input_length = 128
max_target_length = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
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

注意，由于本文直接使用 Transformers 库自带的 `AutoModelForSeq2SeqLM` 函数来构建模型，因此我们上面将每一个 batch 中的数据处理为该模型可接受的格式：一个包含 `'attention_mask'`、`'input_ids'`、`'labels'` 和 `'decoder_input_ids'` 键的字典。

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
    'input_ids': torch.Size([4, 42]), 
    'attention_mask': torch.Size([4, 42]), 
    'decoder_input_ids': torch.Size([4, 33]), 
    'labels': torch.Size([4, 33])
}

{'input_ids': tensor([
        [    7, 23296,  1982,   551, 11120,  4776,   938,   999,  9060,   796,
          2956, 23969,     2, 12942,  4776,  1823,   615,    69,  6300,  3485,
          2424,  1795,  3976,  3016, 17190,  2416,   836,  4269,   627, 12308,
            11, 25509, 15408,     9,     0, 65000, 65000, 65000, 65000, 65000,
         65000, 65000],
        [    7,  4585,   387, 20443,   546, 11367,    25,     0, 65000, 65000,
         65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000,
         65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000,
         65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000,
         65000, 65000],
        [ 6302,  2797,  2692,  6502,  8275,     2,  1966,  1056,   670,    75,
             2, 36923, 31262,    36,  4067, 12792,  1090,  6412, 23093,   142,
          1018, 42661, 48257,   478, 10650,   478,     2,   258,    69,   854,
          1211, 23443, 42661, 48257,     2, 34376,   266,  2353,  6477, 42367,
             9,     0],
        [    7,  4555,  2862,     2,   146, 30112,  6372, 52691,    16,  6372,
         12823, 11895,  3213,  5925,  8395,     2, 13233,  1163,  2751,  4672,
            36, 25862,  2138,     9,     0, 65000, 65000, 65000, 65000, 65000,
         65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000,
         65000, 65000]]), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'decoder_input_ids': tensor([
        [65000, 13098, 10460,   111,     3,  6279,     4,  4917, 51855,   119,
          8661,  1653,     2, 20920, 10460,  3523,    42,  2026, 47566,  6355,
          7590, 55451,   187,    12,  1960,  1511,   436,    56,  3549,    21,
            22,  8477,     5],
        [65000, 17705,  7340,  4889,  1206,    25,     0, 65000, 65000, 65000,
         65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000,
         65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 65000,
         65000, 65000, 65000],
        [65000,  3368,   372, 52458, 29425,    95, 50405,   250,    57, 51801,
         14114,   362,    12,  2195,  9046, 15319,    46,     3, 15379,  9162,
          7754,   601,   216,     2,  2403,     8,     3, 27525,     5,     0,
         65000, 65000, 65000],
        [65000, 56146,     2,     3,   273, 11386, 21049,    37,  1246,   294,
          1298,    18,    12,  7625,    30,  1352,    61, 51192,    22,     6,
         31111,     2,    60,    44,   611,  3396, 16946,     5,     0, 65000,
         65000, 65000, 65000]]), 
 'labels': tensor([
        [13098, 10460,   111,     3,  6279,     4,  4917, 51855,   119,  8661,
          1653,     2, 20920, 10460,  3523,    42,  2026, 47566,  6355,  7590,
         55451,   187,    12,  1960,  1511,   436,    56,  3549,    21,    22,
          8477,     5,     0],
        [17705,  7340,  4889,  1206,    25,     0,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100],
        [ 3368,   372, 52458, 29425,    95, 50405,   250,    57, 51801, 14114,
           362,    12,  2195,  9046, 15319,    46,     3, 15379,  9162,  7754,
           601,   216,     2,  2403,     8,     3, 27525,     5,     0,  -100,
          -100,  -100,  -100],
        [56146,     2,     3,   273, 11386, 21049,    37,  1246,   294,  1298,
            18,    12,  7625,    30,  1352,    61, 51192,    22,     6, 31111,
             2,    60,    44,   611,  3396, 16946,     5,     0,  -100,  -100,
          -100,  -100,  -100]])}
```

可以看到，DataLoader 按照我们设置的 batch size，每次对 4 个样本进行编码，并且在标签序列中，pad token 对应的索引都被设置为 -100。decoder input IDs 尺寸与标签序列完全相同，且通过向后移位在序列头部添加了特殊的“序列起始符”，例如第一个样本：

```
'labels': 
        [13098, 10460,   111,     3,  6279,     4,  4917, 51855,   119,  8661,
          1653,     2, 20920, 10460,  3523,    42,  2026, 47566,  6355,  7590,
         55451,   187,    12,  1960,  1511,   436,    56,  3549,    21,    22,
          8477,     5,     0]
'decoder_input_ids': 
        [65000, 13098, 10460,   111,     3,  6279,     4,  4917, 51855,   119,
          8661,  1653,     2, 20920, 10460,  3523,    42,  2026, 47566,  6355,
          7590, 55451,   187,    12,  1960,  1511,   436,    56,  3549,    21,
            22,  8477,     5]
```

至此，数据预处理部分就全部完成了！

> 经笔者测试，即使我们在 batch 数据中没有包含 decoder input IDs，模型也能正常训练，它会自动调用模型的 `prepare_decoder_input_ids_from_labels` 函数来构造 `decoder_input_ids`。

## 训练模型

本文直接使用 Transformers 库自带的 `AutoModelForSeq2SeqLM` 函数来构建模型，并且在批处理函数中还调用了模型自带的 `prepare_decoder_input_ids_from_labels` 函数，因此下面只需要实现 Epoch 中的”训练循环”和”验证/测试循环”。

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

验证/测试循环负责评估模型的性能。对于序列生成任务，经典的评估指标是 Kishore Papineni 等人在[《BLEU: a Method for Automatic Evaluation of Machine Translation》](https://aclanthology.org/P02-1040.pdf)中提出的 [BLEU 值](https://en.wikipedia.org/wiki/BLEU)，它可以度量两个词语序列之间的一致性，但是并不会衡量生成文本的语义连贯性或者语法正确性。

由于计算 BLEU 需要输入分好词的文本，而分词方式的不同会造成评估的差异，因此现在最常用的翻译评估指标是 [SacreBLEU](https://github.com/mjpost/sacrebleu)，它对分词的过程进行了标准化。SacreBLEU 直接以未分词的文本作为输入，并且考虑到一个句子可以有多种翻译结果，SacreBLEU 对于同一个输入可以接受多个目标作为参考。虽然我们使用的 translation2019zh 语料对于每一个句子只有一个参考，也需要将其包装为一个句子列表的列表，例如：

```python
from sacrebleu.metrics import BLEU

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
bad_predictions_1 = ["This This This This"]
bad_predictions_2 = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]

bleu = BLEU()
print(bleu.corpus_score(predictions, references).score)
print(bleu.corpus_score(bad_predictions_1, references).score)
print(bleu.corpus_score(bad_predictions_2, references).score)
```

```
46.750469682990165
1.683602693167689
0.0
```

BLEU 值的范围从 0 到 100，越高越高。可以看到，对于一些槽糕的翻译结果，例如包含大量重复词语或者长度过短的预测结果，会计算出非常低的 BLEU 值。

> SacreBLEU 默认会采用 mteval-v13a.pl 分词器对文本进行分词，但是它无法处理中文、日文等语言。对于中文就需要通过设置参数 `tokenize='zh'` 手动使用中文分词器，否则会计算出不正确的 BLEU 值：
>
> ```python
> from sacrebleu.metrics import BLEU
> 
> predictions = [
>     "我在苏州大学学习计算机，苏州大学很美丽。"
> ]
> 
> references = [
>     [
>         "我在环境优美的苏州大学学习计算机。"
>     ]
> ]
> 
> bleu = BLEU(tokenize='zh')
> print(f'BLEU: {bleu.corpus_score(predictions, references).score}')
> bleu = BLEU()
> print(f'wrong BLEU: {bleu.corpus_score(predictions, references).score}')
> ```
>
> ```
> BLEU: 45.340106118883256
> wrong BLEU: 0.0
> ```

使用 `AutoModelForSeq2SeqLM` 构造的模型同样对 Decoder 的解码过程进行了封装，我们只需要调用模型的 `generate()` 函数就可以自动地逐个生成预测 token。例如，我们可以直接调用预训练好的 Marian 模型进行翻译：

```python
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

sentence = '我叫张三，我住在苏州。'

sentence_inputs = tokenizer(sentence, return_tensors="pt").to(device)
sentence_generated_tokens = model.generate(
    sentence_inputs["input_ids"],
    attention_mask=sentence_inputs["attention_mask"],
    max_length=128
)
sentence_decoded_preds = tokenizer.batch_decode(sentence_generated_tokens, skip_special_tokens=True)
print(sentence_decoded_preds[0])
```

```
Using cpu device
My name is Zhang San, and I live in Suzhou.
```

当然了，翻译多个句子也没有问题：

```python
sentences = ['我叫张三，我住在苏州。', '我在环境优美的苏州大学学习计算机。']

sentences_inputs = tokenizer(
    sentences, 
    padding=True, 
    max_length=128,
    truncation=True, 
    return_tensors="pt"
).to(device)
sentences_generated_tokens = model.generate(
    sentences_inputs["input_ids"],
    attention_mask=sentences_inputs["attention_mask"],
    max_length=128
)
sentences_decoded_preds = tokenizer.batch_decode(sentences_generated_tokens, skip_special_tokens=True)
print(sentences_decoded_preds)
```

```
[
    'My name is Zhang San, and I live in Suzhou.', 
    "I'm studying computers at Suzhou University in a beautiful environment."
]
```

因此，在验证/测试循环中，我们首先通过 `model.generate()` 函数获取预测结果，然后将预测结果和正确标签都处理为 SacreBLEU 接受的文本列表形式，并且将标签序列中的 -100 替换为 pad token 的 ID，以便于分词器解码，最后送入到 SacreBLEU 中计算 BLEU 值：

```python
from sacrebleu.metrics import BLEU
bleu = BLEU()

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
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"{mode} BLEU: {bleu_score:>0.2f}\n")
    return bleu_score
```

为了方便后续保存验证集上最好的模型，我们还在验证/测试循环中返回评估出的 BLEU 值。

### 保存和加载模型

与之前一样，我们会根据模型在验证集上的性能来调整超参数以及选出最好的模型，然后将选出的模型应用于测试集以评估最终的性能。这里我们继续使用 AdamW 优化器，并且通过 `get_scheduler()` 函数定义学习率调度器：

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
best_bleu = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_bleu = test_loop(valid_dataloader, model, mode='Valid')
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin')
print("Done!")
```

在开始训练之前，我们先评估一下没有微调的模型在测试集上的性能。这个过程比较耗时，你可以在它执行的时候喝杯咖啡:)

```python
test_data = TRANS('translation2019zh/translation2019zh_valid.json')
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)

test_loop(test_dataloader, model, mode='Test')
```

```
Using cuda device
100%|█████████████████████████| 1229/1229 [31:57<00:00,  1.56s/it]
Test BLEU: 42.61
```

可以看到 BLEU 值为 42.61，证明我们的模型即使不进行微调，也已经具有不错的汉英翻译能力。然后，我们正式开始训练，完整的训练代码如下：

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU
import json
import random
import numpy as np
import os

max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000

max_input_length = 128
max_target_length = 128
train_batch_size = 8
test_batch_size = 8
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

class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = TRANS('translation2019zh/translation2019zh_train.json')
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
test_data = TRANS('translation2019zh/translation2019zh_valid.json')

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
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
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, collate_fn=collote_fn)

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

bleu = BLEU()

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
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"{mode} BLEU: {bleu_score:>0.2f}\n")
    return bleu_score

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_bleu = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_bleu = test_loop(valid_dataloader, model, mode='Valid')
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin')
print("Done!")
```

```
Using cuda device
Epoch 1/3
-------------------------------
loss: 2.561596: 100%|███████| 25000/25000 [37:44<00:00, 11.04it/s]
100%|█████████████████████████| 2500/2500 [32:52<00:00,  1.27it/s]
Valid BLEU: 38.11

saving new weights...

Epoch 2/3
-------------------------------
loss: 2.431701: 100%|███████| 25000/25000 [37:44<00:00, 11.04it/s]
100%|█████████████████████████| 2500/2500 [32:18<00:00,  1.29it/s]
Valid BLEU: 37.92

Epoch 3/3
-------------------------------
loss: 2.343894: 100%|███████| 25000/25000 [37:44<00:00, 11.04it/s]
100%|█████████████████████████| 2500/2500 [32:20<00:00,  1.29it/s]
Valid BLEU: 38.11
```

可以看到，随着训练的进行，模型在验证集上的 BLEU 值先降低后提升（在 38 左右波动），并且最后一轮也没有取得更好的性能。因此，3 轮 Epoch 训练结束后，目录下只保存了首轮训练后的模型权重：

```
epoch_1_valid_bleu_38.11_model_weights.bin
```

最后，我们加载这个模型权重，再次评估模型在测试集上的性能：

```python
test_data = TRANS('translation2019zh/translation2019zh_valid.json')
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)

model.load_state_dict(torch.load('epoch_1_valid_bleu_38.11_model_weights.bin'))
test_loop(test_dataloader, model, mode='Test')
```

```
Using cuda device
100%|█████████████████████████| 1229/1229 [33:43<00:00,  1.65s/it]
Test BLEU: 52.59
```

可以看到，经过微调，模型取得了近 10 个点的 BLEU 值提升，证明了我们对模型的微调是成功的。

> 注意：前面我们只保存了模型的权重（并没有同时保存模型结构），因此如果要单独调用上面的代码，需要首先实例化一个结构完全一样的模型，再通过 `model.load_state_dict()` 函数加载权重。

至此，我们使用 Transformers 库进行翻译任务就全部完成了！

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[3]](https://huggingface.co/docs/transformers/index) Transformers 官方文档
