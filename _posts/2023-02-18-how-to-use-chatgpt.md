---
layout: article
title: ChatGPT 使用教程 (Python)
tags:
    - NLP
mathjax: false
--- 

作为近年来自然语言处理 (NLP) 技术的集大成者，ChatGPT 一经推出就引起了巨大的轰动，很多人甚至将其视为真正的人工智能。无论如何，ChatGPT 的确是一件能够帮助人们提高工作效率的强大工具，因此学习它的使用方法很有必要。

本文将通过一系列 Python 代码示例为大家演示 ChatGPT 的使用方法，希望对你能有帮助。

## ChatGPT 能做什么

### 什么是 ChatGPT

ChatGPT 是基于 GPT 构建且专门用于对话场景的一个生成语言模型。顾名思义，生成语言模型就是能自动生成文本的语言模型，它们通过训练能够生成符合人类习惯的流畅文本，还能根据用户需求完成特定的任务。很明显，对于“将一篇长文压缩为短文”和“回答苏轼是哪个朝代的人”，虽然模型都是生成一段文本作为答案，但使用场景的不同的。

先前的许多 NLP 模型已经可以完成这些任务，但是它们有两个缺陷：

1. 一个模型只能完成一种任务。因此如果要构建一个 AI 工具在后台需要包装一大堆的模型，还需要人工规则来判断到底调用哪一个。此外，这种框架下模型无法通过完成多种任务来提高对语言的理解。理论上一个真正学会人类语言的模型，应该能够完成多种语言任务。
2. 无法记住上下文。模型只能对于当前的输入给出答复，而无法记住上下文语境。例如先问模型“苏轼是哪个朝代的人？”，等模型回复后又问“他的代表作品有哪些？”，模型就会回答“不知道”，因为模型不知道这里的“他”指的是谁。

幸运的是，得益于模板 (Prompt) 技术和强化学习的应用，ChatGPT 对这两个问题都得到了较好的解决。

模板技术就是运用人工定义的模板来引导模型完成特定任务。例如使用模板“请将下面的文本翻译为英文：xxx”和“请为下面的文本生成摘要：xxx”来分别进行翻译和摘要，如果模型能够正确理解模板中的指令 (Instruction)，一个模型就能完成各种任务。此外，由于模板本身就是人类文本，因此只需要在每次交互前将已完成的对话内容都拼接到模板中，模型就可以基于完整的上下文来生成回复，大幅减少“前言不搭后语”的情况。

那么 ChatGPT 真的具有智能了吗？

ChatGPT 本质上是一个语言模型，这意味着它最擅长的是生成符合人类习惯的文本。简单地来说，ChatGPT 就是首先通过训练记住了大部分“喂”给它的数据，然后再基于记住的知识回复用户的提问。由于它训练的目标是模仿人类写作，其能力强大到甚至对不了解的问题都可以给出以假乱真的回复。而且因为模型不可能真的记住所有看过的信息，所以丢失部分细节以及“记忆错乱”就无法避免。更重要的是，即使模型通过引入思维链（CoT）、在代码语料上训练等方式改进了推理能力，它本质上依然只是在“记住”看过的信息，而不是在“理解”，还远达不到大家想象中的人工智能。

总而言之，**ChatGPT 就是一个记住了海量信息的“天才”写手**，它可以帮我们高效地处理信息，但是偶尔也会犯错，而且它的回答是基于训练时看过信息的提炼，而不是真的“思考”出来的结果。

### 模板设计

ChatGPT 的主要功能就是文本续写，即将用户给定的文本片段续写成为完整的文档，而实现续写功能的核心就是用户构造的模板 (Prompt)。正如前面所说，模板技术的引入使得我们使用一个模型就可以处理几乎所有的文字任务，唯一需要关注的就是构建合适的模板，也就是模板设计：

- **阐述清楚任务需求**：模板通过描述性指令和示例（或者两者的结合）来阐述任务需求。通常对任务的描述越清晰，给出的例子越有代表性，模型的表现就越好。
- **提供足够且高质量的数据：**如果要进行文本分类或者希望模型遵循某一范式，就需要在模板中提供足够的示例（你也不可能只看一道例题就学会某个数学概念），并且还要保证这些示例是正确的。
- **设置正确的参数**：生成语言模型的输出具有一定的随机性，对于相同的模板也可能回复不同的内容，这可以通过设置 `top_p` 参数来控制。减小 `top_p` 可以降低模型回复的“随机性”，反之增大 `top_p` 就能使模型的回复更多样化。核心思想就是要在“严谨”和“创造性”之间寻找平衡。

下面为大家展示如何构建模板来完成一些常规任务：

- **分类**：我们可以通过提供关于任务的描述以及一些示例来构建文本分类器。例如对微博进行情感分类：

  ```
  判断一条微博的情感极性是积极、消极还是中性。
  
  微博: 流浪地球拍的真好，特效完全不输国外的科幻大片。
  情感:
  ```

  ```
  积极
  ```

  这里：1）通过描述中的“微博”和“情感”来指示输入和输出。虽然也可以使用缩写或者记号来指定输入和输出，但实践表明最开始的时候使用简单的语言描述通常会更好；2）告诉模型对于不同的类别如何回复。例如这里我们在指令中明确需要模型返回“积极“、”消极“和”中性”三个标签；3）对于常规任务只需提供很少的示例。这里因为模型熟悉情感分类任务，我们甚至没有给出任何示例。如果你处理的是一个比较个性化的分类任务，那么最好还是先给出一些示例。

  我们还可以让模型一次性返回多个结果以提高效率，例如同时判断多条微博的情感极性：

  ```
  判断下面这些微博的情感极性是积极、消极还是中性。
  
  1. “流浪地球拍的真好，特效完全不输国外的科幻大片。”
  2. “这些天的天气真是糟透了，天天下雨”
  3. “我准备去吃饭了”
  
  微博情感极性:
  ```

  ```
  1. 积极
  2. 消极
  3. 中性
  ```

  注意，当我们要模型创建列表或者评估文本时，需要设置合适的参数（Top P 或 Temperature）以防止出现漂移问题。例如列表的长度就不能太长，否则就容易出现漂移。

- **生成**：这是大部分的使用场景，即让模型帮我们想一些新的点子。ChatGPT 的生成功能非常强大，可以生成从故事创意、人物描述到商业计划、营销口号等在内的各种文本。例如：

  ```
  想一些关于结合 VR 和健身的想法:
  ```

  ```
  1. 创建一个虚拟现实体验，在里面可以进行真实的健身活动。例如，用户可以在虚拟现实中运行、跳绳、俯卧撑等。
  2. 让用户可以通过 VR 技术来了解到各式各样的健身方法的正确姿势，帮助他们正确执行各项动作。
  3. 通过 VR 技术来制作出一些不断变化的健身课程，使得用户总是能够感受到新鲜感。
  4. 为用户提供一套充满乐趣且对整体健康有益的 VR 健身课程（如舞剧、武术之旅、冲浪之旅、气球大作战…）
  ```

  当然，你也可以通过在模板中给出一些示例来进一步提高模型回复的质量。

- **对话**：ChatGPT 非常擅长与人类甚至自己进行对话。只需要通过指令描述对话语境和模型扮演的角色，它就可以扮演从乐于助人的客服到喜欢开玩笑的伙伴在内的各种对话角色。同样我们也可以通过提供样例来进一步改进模型的表现。

  ```
  下面是与客服机器人的对话. 这个客服机器人热情、乐于助人并且非常友好。
  
  人类: 你好，你是谁?
  客服机器人: 
  ```

  ```
  你好，我是客服机器人。我可以帮助您解决许多问题。您有什么问题吗？
  ```

  这里我们不仅设置了对话的语境（客服机器人），而且还规定了模型该如何表现（“热情”、“乐于助人”）。默认情况下模型没有内在身份，但是指定其扮演的角色通常会更好。例如让模型以一名女性生物学家的身份进行对话，就能够获得更加智能和深思熟虑的回复。同样通过描述设定模型的表现也会使模型的回复更加符合我们的预期。

  例如让模型扮演一个尖酸刻薄且不愿意回答问题的人：

  ```
  张三是一个尖酸刻薄且不愿意回答问题的人。
  
  你: 你好，你知道苏轼是哪个朝代的人吗？
  张三: 这种问题还需要来问我？当然是宋朝。
  
  你: 珠穆拉玛峰有多高？
  张三: 你是不是不会用百度啊？一搜就知道是8848米。
  
  你: 第一架飞机是什么时候制造出来的啊？
  张三:
  ```

  ```
  这也是你要问我的？1903年12月17日，费德勒兄弟驾驶飞机在美国完成了第一次飞行。
  ```

  可以看到模型仅仅通过我们给出的几个示例就识别出了这种模式，从而能够生成无穷无尽的挖苦回复。

  注意，正如前面所说，对于事实性问题，ChatGPT 偶尔也会出现”记忆偏差“，甚至编造出以假乱真的答案。这可以通过两种方式来缓解：1）在模板中直接提供事实（例如维基百科条目）；2）设置低概率，并且通过示例告诉模型在什么时候回复”我不知道“。

- **翻译**：作为一种多语言模型，ChatGPT 熟悉各种不同语言中词语、字符的表示方式，因此基于某种语言的语义将其用其他语言表达出来自然不在话下。例如：

  ```
  将这句话翻译为1. 英语、2. 法语和3. 日语：今天天气真好，你想不想和我一起去爬山？
  
  翻译结果:
  ```

  ```
  1. The weather is so nice today, would you like to go hiking with me?
  2. La météo est si belle aujourd'hui, aimeriez-vous aller randonner avec moi ?
  3. 今日は天気がとても良いですね、私と一緒にハイキングに行きません
  ```

  注意，这个例子能够正确翻译是因为模型的训练数据中包含有这些语言的文本。如果你想要处理一些小语种，就需要在模板中提供更多的示例，甚至需要[微调模型](https://platform.openai.com/docs/guides/fine-tuning)来增加模型对该语言的理解。

- **摘要**：ChatGPT 可以把握长文本的核心内容并且使用简短的文本将其重新表述出来，我们甚至还可以设定摘要的目标对象。例如：

  ```
  为二年级小学生生成下面文本的摘要：
  
  木星是距离太阳第五近的行星，也是太阳系中体积最大的行星，目前已知有92颗卫星。天文学家很早就发现了这颗行星，罗马人以他们的神称这颗行星为朱庇特。古代中国则称木星为岁星、太岁，取其绕行天球一周约为12年，与地支相同之故，且产生了岁星纪年法。到西汉时期，《史记‧天官书》作者天文学家司马迁从实际观测发现岁星呈青色，与“五行”学说联系在一起，正式把它命名为木星。木星是颗巨行星，质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍。太阳系的行星中，木星和土星是气体巨星（天王星和海王星是冰巨星）。
  ```

  ```
  木星是太阳系中第五近的行星，体积最大，有92颗卫星，古代称其为岁星或太岁，与地支相同，产生了岁星纪年法。西汉时期，司马迁发现它呈青色，正式把它命名为木星，质量是太阳的千分之一，但是太阳系其他行星质量总和的2.5倍，是气体巨星。
  ```

- **续写**：虽然 ChatGPT 所有的任务都是以续写的方式进行的，但是续写本身也是一项很实用的功能。当写作时遇到思路卡顿的情况时，或许 ChatGPT 能够提供一些思路。例如：

  ```
  随着城市化的发展，越来越多人向城镇聚集，这也带来了许多的问题，比如
  ```
  
  ```
  交通拥堵、环境污染、住房紧张等。为了解决这些问题，城市必须采取相应的措施。
  ```
  
  注意，对于续写任务，如果想要模型更围绕着模板中的信息展开，就需要降低采样温度 (temperature)（可以在 Python 接口中设置），反之如果你希望回复能够更加天马行空，那么就可以增加该值。

## ChatGPT 使用方法

下面我们以 Python 语言为例，向大家展示如何通过构建模板和设置参数来调用 ChatGPT 完成各种任务。

### 准备工作

在使用接口前，我们首先需要[申请](https://platform.openai.com/account/api-keys)一个秘钥 (secret API key)，之后所有的调用会使用该秘钥进行。此外，我们还需要安装 `openai` 包：

```shell
python3 -m pip install openai
```

安装好后可以通过下面简单的对话例子测试一下接口是否可用。这里 `openai.api_key` 就是前面申请的秘钥。

```python
import openai

openai.api_key = "sk-xxx"
prompt = "下面是和 AI 助手的对话。这个助手热情、聪明、友好。\n\n人类: 你好，你是谁？\nAI 助手:"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.6,
  stop=[" 人类:", " AI 助手:"]
)

print(response['choices'][0]['text'])
```

如果调用成功，就会输出类似下面的内容。

```
你好，我是一个 AI 助手！我能帮助您的任何问题。请问有什么可以帮您的？
```

可以看到，我们与模型的所有交互都是通过续写接口 `openai.Completion` 完成的。其中参数 `model` 和 `prompt` 分别对应调用的模型和构建的模板。目前 OpenAI 提供了以下 4 款 GPT-3 模型可供调用：

| 名称             | 描述                                                         | 最大长度(token) | 训练数据 |
| ---------------- | ------------------------------------------------------------ | --------------- | -------- |
| text-davinci-003 | 最强大的 GPT-3 模型。能完成其他模型的所有任务，并且输出的长度更长、质量更高，并且更加遵循模板中指令。 | 4,000           | 2021.06  |
| text-curie-001   | 也很强大。相比 Davinci 速度更快、成本更低。                  | 2,048           | 2019.10  |
| text-babbage-001 | 擅长处理直接简单的任务，速度非常快并且成本很低。             | 2,048           | 2019.10  |
| text-ada-001     | 擅长处理非常简单的任务，速度最快、成本最低                   | 2,048           | 2019.10  |

ChatGPT 默认调用的就是 Davinci 模型，正式它强大的文本处理能力使得 ChatGPT 大放光彩，但是这依赖大量的计算资源，因此 Davinci 的调用成本很高，而且不如其他模型速度快。

> 实际上，Curie 在许多任务上都与 Davinci 表现相近，但是速度更快并且开销只有1/10。实际应用中我们也应该评估其他模型的性能表现，从而选择最合适的模型，还可以通过在特定任务上微调来进一步改进模型表现。

- **Davinci**：最强模型，只需要在模板中加入很少的指令就可以完成几乎所有任务。 Davinci 擅长于处理需要对文本内容有大量理解的任务，例如针对特定受众的摘要和创意内容生成。Davinci 的另一个亮点是能够理解文本的意图，尤其擅长于解决逻辑问题和解释人物的动机，目前 Davinci 甚至能够解决一些涉及因果关系的困难问题。

  *擅长于*：复杂的意图、因果，以及为特定受众进行摘要。

- **Curie**：同样很强大，并且速度非常快。虽然在分析复杂文本方面不如 Davinci，但同样能够胜任许多任务，例如情感分类和摘要。Curie 还非常擅长回答问题和执行对话。

  *擅长于*：翻译、高难度分类、情感分析和摘要。

- **Babbage**：可以执行相对简单的任务，例如较为直接的分类。 在语义搜索方面，它也非常有能力对文档与搜索查询的匹配程度进行排名。

  *擅长于*：中等难度分类和语义搜索分类。

- **Ada**：最快的 GPT-3 模型，可以执行文本解析、地址更正和难度较低的分类任务。Ada 的性能通常可以通过提供更多上下文来提高。

  *擅长于*：文本解析、简单分类、地址修正和关键词抽取

除了模型参数 `model` 和模板参数 `prompt` 以外，其他参数也非常重要，参数设置的不同会对回复有巨大的影响。下面时一些常用的参数：

- `max_tokens`：回复的最大长度。注意，大部分的语言模型都只能处理一定长度内的文本，例如 2048 或 4096 个字符 (token)，因此我们输入的模板长度加上 `max_tokens` 不能超过这个限制。
- `temperature`：采样温度，值介于 0 到 2 之间。值越高，输出就越随机。例如你希望对于同一个问题，模型每次的回复都能有较大的差异，就可以设置为一个较高的值。该参数与 `top_p` 通常只建议修改其中一个。
- `top_p`：一种替代温度采样的方法，称为核采样。模型只考虑具有 top_p 概率质量的 token 的结果。例如设为 0.1 模型就只考虑构成前 10% 概率质量的 token。该参数与 `temperature` 通常只建议修改其中一个。
- `presence_penalty`：值介于 -2.0 到 2.0 之间。正值会根据是否已经出现在文本中来惩罚新生成的 token，从而鼓励模型生成新的内容，避免出现大段重复的文本。
- `frequency_penalty`：值介于 -2.0 到 2.0 之间。正值会根据新生成 token 在文本中的频率对其进行惩罚，从而降低模型逐字重复同一行的可能性。

我们只要构建出合适的模板并且正确地设置这些参数，就能够运用 ChatGPT 来解决大部分的文字性任务。下面我们正式上手，通过调用接口来处理多种任务。

### 对话

前面介绍过，我们首先通过指令来设定对话情景（包括对话语境、希望模型的表现等），然后再拼接上问题送入模型进行续写。例如：

```python
prompt = "我是一个聪明且友好的问答机器人。如果你问我的是事实性的问题，我会给你答案。如果你问我的是没有意义、玩笑或者没有清晰答案的问题，我会回复“不知道”。\n\n问: 你知道苏轼是哪个朝代的诗人吗？\n答:"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)
```

```
苏轼是宋朝的诗人。
```

ChatGPT 可以记住对话的上下文来给出回复，只需要将已完成的对话内容拼接后全部输入模型。例如我们继续提问”那他的代表作有哪些？“：

```python
prompt = "我是一个聪明且友好的问答机器人。如果你问我的是事实性的问题，我会给你答案。如果你问我的是没有意义、玩笑或者没有清晰答案的问题，我会回复“不知道”。\n\n问: 你知道苏轼是哪个朝代的诗人吗？\n答: 苏轼是宋朝的诗人。\n\n问: 那他的代表作有哪些？\n答:"
```

```
苏轼的代表作有《水调歌头》、《念奴娇·赤壁怀古》、《江城子·乙卯正月二十日夜记梦》等。
```

模型成功地识别出问题中的“他”指的是苏轼。下面我们再提问一个玩笑问题，看看模型能否按照我们的设定回答：

```python
prompt = "我是一个聪明且友好的问答机器人。如果你问我的是事实性的问题，我会给你答案。如果你问我的是没有意义、玩笑或者没有清晰答案的问题，我会回复“不知道”。\n\n问: 你知道苏轼是哪个朝代的诗人吗？\n答: 苏轼是宋朝的诗人。\n\n问: 那他的代表作有哪些？\n答: 苏轼的代表作有《水调歌头》、《念奴娇·赤壁怀古》、《江城子·乙卯正月二十日夜记梦》等。\n\n问: 你知道猴子的爸爸叫什么名字吗？\n答:"
```

```
不知道。
```

我们还可以设定在不同情况下的回复。例如对没有意义的问题或者是玩笑回复“什么鬼问题！”，对没有清晰答案的问题回复“这个问题我无法回答。”，还是对于上面的玩笑：

```python
prompt = "我是一个聪明且友好的问答机器人。如果你问我的是事实性的问题，我会给你答案。如果你问我的是没有意义的问题或者是玩笑，我会回复“什么鬼问题！”。如果你问我的是没有清晰答案的问题，我会回复“这个问题我无法回答。”。\n\n问: 你知道苏轼是哪个朝代的诗人吗？\n答: 苏轼是宋朝的诗人。\n\n问: 那他的代表作有哪些？\n答: 苏轼的代表作有《水调歌头》、《念奴娇·赤壁怀古》、《江城子·乙卯正月二十日夜记梦》等。\n\n问: 你知道猴子的爸爸叫什么名字吗？\n答:"
```

```
什么鬼问题！
```

而对于模型无法给出清晰答案的问题，模型也能按照我们的设定进行回答：

```python
prompt = "我是一个聪明且友好的问答机器人。如果你问我的是事实性的问题，我会给你答案。如果你问我的是没有意义的问题或者是玩笑，我会回复“什么鬼问题！”。如果你问我的是没有清晰答案的问题，我会回复“这个问题我无法回答。”。\n\n问: 你知道苏轼是哪个朝代的诗人吗？\n答: 苏轼是宋朝的诗人。\n\n问: 那他的代表作有哪些？\n答: 苏轼的代表作有《水调歌头》、《念奴娇·赤壁怀古》、《江城子·乙卯正月二十日夜记梦》等。\n\n问: 你知道猴子的爸爸叫什么名字吗？\n答: 什么鬼问题！\n\n问: 你认为一百年之后的美国总统会是谁？\n答:"
```

```
这个问题我无法回答。
```

当然，对于扮演常用的 AI 助手、客服类机器人，ChatGPT 也是信手拈来：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="下面是与AI助手的对话。这个助手热心、有创造性、聪明并且非常友好。\n\n人类: 你好，你是谁？\nAI: 你好，我是AI助手。我很高兴认识你！我可以为你提供服务，例如查看天气预报，为你推荐有趣的新闻，甚至可以回答你在生活中遇到的问题！\n\n人类: 你好，我想要取消订单。\nAI:",
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.6,
  stop=[" 人类:", " AI:"]
)
```

```python
当然，我可以帮助你取消订单。请问你需要取消哪个订单？
```

只要是在模型支持的最大长度范围内，模型就可以无休止地与你聊下去：

### 摘要

摘要任务需要模型能够理解文章的语义，并且重新遣词造句用简短的文本将其表述出来。例如，生成木星描述的摘要：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="为下面的文本生成摘要:\n\n木星是距离太阳第五近的行星，也是太阳系中体积最大的行星，目前已知有92颗卫星。天文学家很早就发现了这颗行星，罗马人以他们的神称这颗行星为朱庇特。古代中国则称木星为岁星、太岁，取其绕行天球一周约为12年，与地支相同之故，且产生了岁星纪年法。到西汉时期，《史记‧天官书》作者天文学家司马迁从实际观测发现岁星呈青色，与“五行”学说联系在一起，正式把它命名为木星。木星是颗巨行星，质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍。太阳系的行星中，木星和土星是气体巨星（天王星和海王星是冰巨星）。",
  temperature=0.7,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
木星是太阳系第五近行星，体积最大，已知有92颗卫星，罗马人称其为朱庇特，古代中国称为岁星，西汉时期司马迁将其正式命名为木星，质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍，是气体巨星。
```

前面介绍过，我们还可以通过添加指令让模型为特定的受众生成摘要。例如：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="对于下面的文本，为小学生生成一个活泼有趣的摘要:\n\n木星是距离太阳第五近的行星，也是太阳系中体积最大的行星，目前已知有92颗卫星。天文学家很早就发现了这颗行星，罗马人以他们的神称这颗行星为朱庇特。古代中国则称木星为岁星、太岁，取其绕行天球一周约为12年，与地支相同之故，且产生了岁星纪年法。到西汉时期，《史记‧天官书》作者天文学家司马迁从实际观测发现岁星呈青色，与“五行”学说联系在一起，正式把它命名为木星。木星是颗巨行星，质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍。太阳系的行星中，木星和土星是气体巨星（天王星和海王星是冰巨星）。",
  temperature=0.7,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
小学生摘要：木星是太阳系中最大的行星，它距离太阳第五近，目前已知有92颗卫星。古代人称它为朱庇特或岁星，它每12年绕行一次天球，古代中国把它叫做太岁，它的质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍。它是一颗气体巨星（天王星和海王星是冰巨星）。
```

可以看到，此时模型在遣词造句上选择了更为通俗易懂的表达。除此以外，ChatGPT 还可以为对话、会议等更为复杂的场景撰写摘要，通过分析对话的内容来把握核心要点。例如：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="将我的会议速记转换为一份会议内容纪要：\n\n张三: 利润增长了50%\n李四: 新的服务已经上线\n王五: 需要更多时间来修复软件问题\n小帅: Beta 测试即将完成",
  temperature=0,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
会议内容纪要：本次会议的主要议题是关于公司的利润增长和新的服务上线。张三报告了公司利润增长了50%的情况。李四报告了新的服务已经上线的情况。王五指出需要更多时间来修复软件问题。小帅报告了Beta测试即将完成的情况。
```

### 文本分类

由于 ChatGPT 已经掌握了许多事实性知识，因此通常只需要在模板中对分类要求进行简单的描述并给出几个示例，模型就能自动捕获到用户需要的分类模式。对于一些常规任务，例如情感分析，甚至可以在不给出任何示例的情况下进行：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="判断下面这条微博的情感极性。\n\n微博: “流浪地球拍的真好，特效完全不输国外科幻大片。”\n情感极性:",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0
)
```

```
正面
```

为了提高使用效率，也可以让模型一次性判断多条微博的情感极性：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="判断下面这些微博的情感极性。\n\n1. “流浪地球拍的真好，特效完全不输国外的科幻大片。”\n2. “这些天的天气真是糟透了，天天下雨”\n3. “虽然等了好久，但是这家店菜的味道真的不错。”\n\n微博情感极性:",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
1. 正面
2. 负面
3. 正面
```

当然，我们也可以通过给出更细致的指令来进一步约束模型的行为，例如明确规定模型对于不同的情况应该返回怎样的标签。比如这里我们让模型返回“积极“、”消极“和”中性”三种情感标签：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="判断下面这些微博的情感极性是“积极”、“消极”还是“中性”。\n\n1. “流浪地球拍的真好，特效完全不输国外的科幻大片。”\n2. “这些天的天气真是糟透了，天天下雨”\n3. “虽然等了好久，但是这家店菜的味道真的不错。”\n4. “我准备去吃饭了。”\n\n微博情感极性:",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
1. 积极
2. 消极
3. 积极
4. 中性
```

如果你要处理的是更个性化的分类任务，就需要在前缀中添加更多对每个类别的描述，以及模型对不同情况的返回标签，而且还需要给出一些示例以更好地指引模型。

### 关键信息抽取

从文本中抽取关键信息也是日常工作中常见的操作，例如抽取关键词，此时采样温度 `temperature` 参数就非常重要，较低的值会使得模型尽可能从原文中摘取关键词，而较高的值会使得模型生成一些不在原文中但是相关的词：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="从下面的文本中抽取关键词:\n\n木星是距离太阳第五近的行星，也是太阳系中体积最大的行星，目前已知有92颗卫星。天文学家很早就发现了这颗行星，罗马人以他们的神称这颗行星为朱庇特。古代中国则称木星为岁星、太岁，取其绕行天球一周约为12年，与地支相同之故，且产生了岁星纪年法。到西汉时期，《史记‧天官书》作者天文学家司马迁从实际观测发现岁星呈青色，与“五行”学说联系在一起，正式把它命名为木星。木星是颗巨行星，质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍。太阳系的行星中，木星和土星是气体巨星（天王星和海王星是冰巨星）。",
  temperature=0.5,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.8,
  presence_penalty=0
)
```

```
关键词：木星、太阳系、卫星、罗马人、古代中国、岁星纪年法、
```

还可以指定要抽取的信息类型，比如从邮件中抽取出姓名和地址：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="从下面的邮件中抽取出姓名和地址:\n\n作者，您好！\n\n经过编辑以及三位专业评审的讨论，我们已经决定录用您的稿件。\n\n我们还需要您的授权确认书，请按照我们网站上的要求填写，并尽快邮寄到以下地址：苏州市姑苏区苏州大学出版社\n\n此致,\n\n敬礼\n\n小明\n\n姓名:",
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
小明
地址: 苏州市姑苏区苏州大学出版社
```

### 文字类创作

自动生成是 ChatGPT 最为拿手的绝活，我们只需要通过指令表述要生成的内容，模型就可以自动为我们撰写各种文案。例如为8848手机编写广告：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="为下面的产品写一个富有创意的广告，用来在飞机上向商务人士投放:\n\n产品: 8848是国内首款钛金手机，由著名企业家王石代言。选择珍贵钛金属和高端牛皮，内置硬件级安全系统，是一款不仅奢华还很实用的手机，向成功者致敬。",
  temperature=0.5,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
商务人士，让我们一起来体验8848，这款由著名企业家王石代言的国内首款钛金手机！它不仅外观精致，内置硬件级安全系统，还搭载了最新的操作系统，让你在商务中更加高效！8848是一款为成功者而设计的奢华手机，它可以帮助你在商务上取得成功，让你更加自信！让我们一起来体验这款钛金手机，让你的成功更加完美！
```

如果老板不满意，我们还可以让模型帮助我们想一些奢华手机与商业成功之间的联系：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="为宣传8848手机想一些关于奢华手机与商业成功之间联系的文案:",
  temperature=0.5,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
1、8848手机，奢华的外表暗示着你拥有了商业成功的魅力；
2、8848手机，让你在商业场合轻松携带奢华，体现你的商业成就；
3、8848手机，在奢华的外表中蕴藏着你的商业智慧，成就你的商业辉煌；
4、8848手机，让你的商业成功更加耀眼，让你的奢华更加具有魅力；
5、8848手机，让你在商业场合中拥有更多的奢华和成功，成就你的商业梦想。
```

或者让 ChatGPT 围绕着某个主题写一些较短的小说。你也可以让模型撰写长文，但是如果想让生成文本质量更好，就需要在模板中对任务需求给出更为详细的描述或者给出更多的示例。

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="围绕下面主题写一个只包含五个句子的短篇恐怖小说。\n\n主题: 晚餐\n五个句子的短篇恐怖小说:",
  temperature=0.8,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0
)
```

```
弗兰克为晚餐准备了一顿丰盛的食物，却没有吃完，因为他听到了来自厨房里恐怖的声音。
他靠近厨房，发现一只巨大的怪物正在仔细品尝他准备的食物。
弗兰克整个身体颤抖着，惊慌失措地找到了安全的藏身之处。
怪物不断地吃着弗兰克准备的食物，直到最后一口，然后便消失了。
弗兰克对这一天发生的事情感到震惊和害怕，决定以后绝不再尝试为晚餐准备如此丰盛的食物。
```

让模型为即将撰写的报告写一个提纲：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="为关于牛顿及其科学贡献的报告写一份中文提纲:",
  temperature=0.3,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
一、牛顿的生平
（1）出生背景
（2）教育经历
（3）职业生涯

二、牛顿的科学贡献
（1）牛顿力学
（2）光学
（3）数学
（4）天文学

三、牛顿的影响
（1）对科学发展的重要性
（2）对现代科学的影响
```

或者在学习某个内容之前，先咨询一下 ChatGPT 有哪些需要掌握的关键知识点：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="在学习自然语言处理时，有哪5个关键要点是我必须掌握的？",
  temperature=0.3,
  max_tokens=1024,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
1. 语言模型：语言模型是自然语言处理的基础，它是一种用于计算句子概率的模型，可以帮助我们预测句子的可能性。
2. 词法分析：词法分析是指将句子分解成有意义的单词和短语，以便进行进一步的处理。
3. 语义分析：语义分析是指将句子的语义（意义）抽取出来，以便进行进一步的处理。
4. 语用分析：语用分析是指根据上下文和语境，从句子中抽取出句子的语用信息，以便进行进一步的处理。
5. 机器翻译：机器翻译是指将一种语言翻译成另一种语言，它是自然语言处理的重要组成部分。
```

让 ChatGPT 推荐一些特定领域的书：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="推荐 10 本中文玄幻小说:",
  temperature=0.5,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0.52,
  presence_penalty=0.5,
  stop=["11."]
)
```

```
1. 《天龙八部》（金庸）
2. 《碧血剑》（古龙）
3. 《鹿鼎记》（金庸）
4. 《斗破苍穹》（天蚕土豆）
5. 《神墓》（辰东）
6. 《星战风暴》(叶非夜)
7. 《大主宰》(天蚕土豆)
8. 《诛仙·青云志》(唐家三少)
9. 《武动乾坤》(净无痕)
10. 《斗罗大陆III龙王传说》(唐家三少)
```

通过设置一些关键词让 ChatGPT 自动撰写评论，例如：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="基于以下要点写一条餐厅的评价:\n\n名称: 豪客来牛排\n牛排味道不错、安静、服务很好、价格实惠。\n\n评价:",
  temperature=0.5,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
我最近去了豪客来牛排，牛排味道不错，环境安静，服务很好，而且价格实惠，非常满意！
```

让 ChatGPT 创建一个采访使用的问题列表：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="为我即将对一位企业家的采访写一份包含8个问题的列表:",
  temperature=0.5,
  max_tokens=512,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
1.什么是您最初启动企业的原因？
2.您如何确定您的企业的成功方向？
3.您有哪些特别的管理技巧？
4.您如何处理企业发展中的挑战？
5.您如何帮助企业拥有更多的竞争优势？
6.您是如何培养员工的？
7.您是如何应对技术变革的？
8.您对未来的企业有什么期望？
```

### 翻译和语法

由于 ChatGPT 具有非常强大的语言建模能力，因此对于修正小的语法错误非常在行：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="将下面的句子转换为标准英语:\n\nShe no went to the market.",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
She didn't go to the market.
```

当然，在训练时已经掌握的语言之间进行翻译也没有问题：

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="将下面的句子翻译为 1. 英语, 2. 德语和 3. 日语:\n\n今天有空出来一起吃饭吗?\n\n1.",
  temperature=0.3,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

```
Do you have time to go out for a meal together today?
2. Hast du heute Zeit, um gemeinsam essen zu gehen?
3. 今日は一緒に食事をする時間がありますか？
```

注意，对于模型并不熟悉的语言，我们不仅需要给出更多的示例，甚至需要在新语言文本上微调模型以增加模型对新语言的理解。

## 总结

得益于 ChatGPT 可以理解并遵循模板中的指令来进行生成，使得我们可以使用 ChatGPT 来处理几乎任何类型的文字类工作。本文只给出了一些常见任务的示例，更多的功能还等待读者自己去发掘。相信只要合理使用 ChatGPT，一定能让你事半功倍。