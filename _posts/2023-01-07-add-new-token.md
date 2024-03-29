---
layout: article
title: 如何向 Transformers 模型词表中添加新 token
tags:
    - Transformers
    - NLP
mathjax: true
--- 

在使用预训练模型时，我们有时需要使用一些自定义 token 来增强输入，例如使用 `[ENT_START]` 和 `[ENT_END]` 在文本中标记出实体。由于自定义 token 并不在预训练模型原来的词表中，因此直接运用分词器 (Tokenizer) 处理输入就会出现问题。

例如直接使用 BERT 分词器处理下面的句子：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'
print(tokenizer(sentence).tokens())
```

```
['[CLS]', 'Two', '[', 'E', '##NT', '_', 'ST', '##AR', '##T', ']', 'cars', '[', 'E', '##NT', '_', 'E', '##ND', ']', 'collided', 'in', 'a', '[', 'E', '##NT', '_', 'ST', '##AR', '##T', ']', 'tunnel', '[', 'E', '##NT', '_', 'E', '##ND', ']', 'this', 'morning', '.', '[SEP]']
```

由于分词器无法识别 `[ENT_START]` 和 `[ENT_END]` ，将它们都当作未知字符处理，例如 `[ENT_END]` 被切分成了 `'['`、`'E'`、`'##NT'`、`'_'`、`'E'`、`'##ND'`、`']'` 七个 token，很明显不符合我们的预期。

此外，有时我们还会遇到一些领域相关词汇，例如医学领域的文本通常会包含大量的医学术语，它们可能并不在模型的词表中（例如一些术语是使用多个词语的缩写拼接而成），这时也会出现上面的问题。

此时我们就需要将这些新 token 添加到模型的词表中，让分词器与模型可以识别并处理这些 token。

## 添加新 token

### 添加方法

Huggingface 的 Transformers 库提供了两种方式来添加新 token，分别是：

- **[`add_tokens()`](https://huggingface.co/docs/transformers/v4.25.1/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_tokens) 添加普通 token：**添加新 token 列表，如果 token 不在词表中，就会被添加到词表的最后。

  ```python
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  
  num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
  print("We have added", num_added_toks, "tokens")
  ```

  ```
  We have added 2 tokens
  ```

  为了防止 token 已经包含在词表中，我们还可以预先对新 token 列表进行过滤：

  ```python
  new_tokens = ["new_tok1", "my_new-tok2"]
  new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
  tokenizer.add_tokens(list(new_tokens))
  ```

- **[`add_special_tokens()`](https://huggingface.co/docs/transformers/v4.25.1/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_special_tokens) 添加特殊 token：**添加包含特殊 token 的字典，键值从 `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens` 中选择。与 `add_tokens()` 类似，如果 token 不在词表中，就会被添加到词表的最后。添加后，还可以通过特殊属性来访问这些 token，例如 `tokenizer.cls_token` 就指向 cls token。

  ```python
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  
  special_tokens_dict = {"cls_token": "[MY_CLS]"}
  
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  print("We have added", num_added_toks, "tokens")
  
  assert tokenizer.cls_token == "[MY_CLS]"
  ```

  ```
  We have added 1 tokens
  ```

  我们也可以使用 `add_tokens()` 添加特殊 token，只需要额外设置参数 `special_tokens=True`：

  ```python
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  
  num_added_toks = tokenizer.add_tokens(["[NEW_tok1]", "[NEW_tok2]"])
  num_added_toks = tokenizer.add_tokens(["[NEW_tok3]", "[NEW_tok4]"], special_tokens=True)
  
  print("We have added", num_added_toks, "tokens")
  print(tokenizer('[NEW_tok1] Hello [NEW_tok2] [NEW_tok3] World [NEW_tok4]!').tokens())
  ```

  ```
  We have added 2 tokens
  ['[CLS]', '[new_tok1]', 'hello', '[new_tok2]', '[NEW_tok3]', 'world', '[NEW_tok4]', '!', '[SEP]']
  ```

  特殊 token 的标准化 (normalization) 过程与普通 token 有一些不同，比如不会被小写。这里我们使用的是不区分大小写的 BERT 模型，因此分词后添加的普通 token `[NEW_tok1]` 和 `[NEW_tok2]` 都被处理为了小写，而特殊 token `[NEW_tok3]` 和 `[NEW_tok4]` 则维持大写，与 `[CLS]` 等自带特殊 token 保持一致。

对于之前的例子，很明显实体标记符 `[ENT_START]` 和 `[ENT_END]` 属于特殊 token，因此按添加特殊 token 的方式进行。如果使用 `add_tokens()` 则需要额外设置 `special_tokens=True`，或者也可以直接使用 `add_special_tokens()`。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
# num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT_START]', '[ENT_END]']})
print("We have added", num_added_toks, "tokens")

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'

print(tokenizer(sentence).tokens())
```

```
We have added 2 tokens
['[CLS]', 'two', '[ENT_START]', 'cars', '[ENT_END]', 'collided', 'in', 'a', '[ENT_START]', 'tunnel', '[ENT_END]', 'this', 'morning', '.', '[SEP]']
```

可以看到，分词器成功地将 `[ENT_START]` 和 `[ENT_END]` 识别为 token，并且依旧保持大写。

### 调整 embedding 矩阵

**注意！无论使用哪种方式向词表中添加新 token 后，都需要重置模型 token embedding 矩阵的大小**，也就是向矩阵中添加新 token 对应的 embedding，这样模型才可以正常工作（将 token 映射到对应的 embedding）。

该操作通过调用预训练模型的 `resize_token_embeddings()` 函数来实现，例如对于上面的例子：

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

print(len(tokenizer))
num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
print("We have added", num_added_toks, "tokens")
print(len(tokenizer))

model.resize_token_embeddings(len(tokenizer))
print(model.embeddings.word_embeddings.weight.size())

# Randomly generated matrix
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
30522
We have added 2 tokens
30524
torch.Size([30524, 768])

tensor([[-0.0325, -0.0224,  0.0044,  ..., -0.0088, -0.0078, -0.0110],
        [-0.0005, -0.0167, -0.0009,  ...,  0.0110, -0.0282, -0.0013]],
       grad_fn=<SliceBackward0>)
```

可以看到，在添加了特殊 token `[ENT_START]` 和 `[ENT_END]`  之后，分词器的词表大小从 30522 增加到了 30524，并且模型的 token embedding 矩阵大小也成功调整为了 $30524\times 768$。

我们还尝试打印出新添加 token 对应的 embedding。因为新 token 会添加在词表的末尾，因此只需打印出矩阵最后两行。如果你重复运行一下上面的代码，就会发现每次打印出的 `[ENT_START]` 和 `[ENT_END]`  的 embedding 是不同的。这是因为在默认情况下，这些**新 token 的 embedding 是随机初始化的**。

## token embedding 初始化

如果有充分的训练语料对模型进行微调或者继续预训练，那么将新添加 token 初始化为随机向量没什么问题。但是如果训练语料较少，甚至是只有很少语料的 few-shot learning 场景下，这种做法就可能存在问题。研究表明，在训练数据不够多的情况下，这些新添加 token 的 embedding 只会在初始值附近小幅波动。换句话说，即使经过训练，它们的值事实上还是随机的。

因此，在很多情况下，我们需要手工初始化这些新 token 的 embedding。对于 Transformers 库来说，可以通过直接对 embedding 矩阵赋值来实现。例如对于上面的例子，我们将这两个新 token 的 embedding 都初始化为全零向量：

```python
with torch.no_grad():
    model.embeddings.word_embeddings.weight[-2:, :] = torch.zeros([2, model.config.hidden_size], requires_grad=True)
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], grad_fn=<SliceBackward0>)
```

注意，初始化 embedding 的过程并不可导，因此这里通过 `torch.no_grad()` 暂停梯度的计算。

### 初始化为已有 token 的值

比较常见的操作是根据新添加 token 的语义，将其值初始化为训练好 token 的 embedding。例如对于上面的例子，我们可以将  `[ENT_START]` 和 `[ENT_END]` 的值都初始化为“entity”对应的 embedding。因为 token id 就是 token 在矩阵中的索引，因此我们可以直接通过 `weight[token_id]` 取出“entity”对应的 embedding。

```python
token_id = tokenizer.convert_tokens_to_ids('entity')
token_embedding = model.embeddings.word_embeddings.weight[token_id]
print(token_id)

with torch.no_grad():
    for i in range(1, num_added_toks+1):
        model.embeddings.word_embeddings.weight[-i:, :] = token_embedding.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
9178
tensor([[-0.0039, -0.0131, -0.0946,  ..., -0.0223,  0.0107, -0.0419],
        [-0.0039, -0.0131, -0.0946,  ..., -0.0223,  0.0107, -0.0419]],
       grad_fn=<SliceBackward0>)
```

可以看到最终结果符合我们的预期，`[ENT_START]` 和 `[ENT_END]` 被初始化为相同的 embedding。

另一种常见的做法是根据新 token 的语义，使用对应的描述文本来完成初始化。例如将值初始化为描述文本中所有 token 的平均值，假设新 token $t_i$ 的描述文本为 $\{w_{i,1},w_{i,2},...,w_{i,n}\}$，那么 $t_i$ 的初始化 embedding 为：

$$
\boldsymbol{E}(t_i) = \frac{1}{n}\sum_{j=1}^n \boldsymbol{E}(w_{i,j})
$$

这里 $\boldsymbol{E}$ 表示预训练模型的 token embedding 矩阵。对于上面的例子，我们可以分别为 `[ENT_START]` 和 `[ENT_END]` 编写对应的描述，然后再对它们的值进行初始化：

```python
descriptions = ['start of entity', 'end of entity']

with torch.no_grad():
    for i, token in enumerate(reversed(descriptions), start=1):
        tokenized = tokenizer.tokenize(token)
        print(tokenized)
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
        new_embedding = model.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
        model.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
['end', 'of', 'entity']
['start', 'of', 'entity']
tensor([[-0.0340, -0.0144, -0.0441,  ..., -0.0016,  0.0318, -0.0151],
        [-0.0060, -0.0202, -0.0312,  ..., -0.0084,  0.0193, -0.0296]],
       grad_fn=<SliceBackward0>)
```

可以看到，这里成功地将 `[ENT_START]` 初始化为“start”、“of”、“entity”三个 token embedding 的平均值，将 `[ENT_END]` 初始化为“end”、“of”、“entity” embedding 的平均值。

## 参考

[[1]](https://www.depends-on-the-definition.com/how-to-add-new-tokens-to-huggingface-transformers/) How to add new tokens to huggingface transformers vocabulary. [Tobias Sterbak](https://www.depends-on-the-definition.com/about/).  
[[2]](https://github.com/huggingface/transformers/issues/1413) Github 讨论 Adding New Vocabulary Tokens to the Models.  
[[3]](https://huggingface.co/docs/transformers/v4.25.1/en/internal/tokenization_utils) 官方文档 Utilities for Tokenizers 章节