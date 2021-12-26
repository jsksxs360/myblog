---
layout: article
title: 维基百科中文语料库词向量的训练：处理维基百科中文语料
tags:
    - NLP
mathjax: false
---

在 NLP 自然语言处理中，语义层面的理解一直是一大难题，其中度量词语或句子的语义相关性又是语义理解的基础。目前的主流做法大多首先把词语转化为词向量，然后通过计算向量之间的距离来衡量词语的相关性。目前主要使用 Google 提出的 [Word2Vec](https://github.com/svn2github/word2vec) 方法来训练词向量，文本将简单梳理一下从数据处理到词向量训练的全过程。

## 获取并处理维基百科中文语料库

中文维基百科语料库的下载链接为：[https://dumps.wikimedia.org/zhwiki/](https://dumps.wikimedia.org/zhwiki/), 里面按照日期提供了多个版本的中文语料，每个版本都提供了很多类型的可选项，例如只包含标题、摘要等等。我们选用的是最新版本包含标题和正文的 [zhwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)。

### 抽取内容

**[WikiExtractor](https://github.com/attardi/wikiextractor)** 是一个开源的用于抽取维基百科语料库的工具，由 python 写成，通过这个工具可以很容易地从语料库中抽取出相关内容。使用方法如下：

```bash
$ python wikiextractor/WikiExtractor.py  -b 2000M -o zhwiki_extracted zhwiki-latest-pages-articles.xml.bz2
```

这个工具就是一个 python 脚本，因此无需安装，`-b` 参数指对提取出来的内容进行切片后每个文件的大小，如果要将所有内容保存在同一个文件，那么就需要把这个参数设得大一下，`-o` 的参数指提取出来的文件放置的目录。更多参数可参考其 github 主页的说明。

抽取后的内容格式为每篇文章被一对 `<doc> </doc>` 包起来，而 `<doc>` 中的包含了属性有文章的 id、url 和 title 属性，如 `<doc id="13" url="https://zh.wikipedia.org/wiki?curid=13" title="数学">`。

### 去除标点符号

去除标点符号有两个问题需要解决，一是像下面这种为了解决各地术语名称不同的问题：

```
他的主要成就包括Emacs及後來的GNU Emacs，GNU C 編譯器及-{zh-hant:GNU 除錯器;zh-hans:GDB 调试器}-。
```

另外一个就是将所有标点符号替换成空字符，通过正则表达式均可解决这两个问题，下面是具体实现的 python 代码 **script.py**。

```python
#!/usr/bin/python
# -*- coding: utf-8 -*- 

import sys
import re
import io

def pre_process(input_file, output_file):
	multi_version = re.compile('-\{|zh-hans|zh-hant|zh-cn|\}-')
	punctuation = re.compile("[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
	with io.open(output_file, mode = 'w', encoding = 'utf-8') as outfile:
		with io.open(input_file, mode = 'r', encoding ='utf-8') as infile:
			for line in infile:
				if line.startswith('<doc') or line.startswith('</doc'):
					print(line)
					continue
				line = multi_version.sub('', line)
				line = punctuation.sub(' ', line)
				outfile.write(line)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: python script.py input_file output_file")
		sys.exit()
	input_file, output_file = sys.argv[1], sys.argv[2]
	pre_process(input_file, output_file)
```

### 繁简转换及分词

维基百科中文语料库同时包含了简体和繁体中文的内容，因而我们首先要做繁简转换，把所有的内容都转换为简体。词向量的训练需要输入分好词的文本，因而我们还需要对处理后的文本进行分词。这里使用 Java 编写的 [AHANLP](https://github.com/jsksxs360/AHANLP) 包来进行上述操作。

首先从 GitHub 上[下载](https://github.com/jsksxs360/AHANLP/releases)最新版本 **ahanlp.jar** 和对应的基础数据包 **AHANLP_base**，将配置文件 ahanlp.properties 和 hanlp.properties 放入 src 目录下。因为这里的任务只需要使用到分词，所以只需要下载基础数据包 [AHANLP_base](https://github.com/jsksxs360/AHANLP/releases)，将解压出的 `dictionary` 目录和 `model` 目录存放到项目的 `data/` 目录下。

AHANLP 项目各项配置完成后，就可以开始我们的工作了：

```java
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import me.xiaosheng.chnlp.AHANLP;

public class Wiki {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("E:\\wiki_text.txt"));
        BufferedWriter bw = new BufferedWriter(new FileWriter("E:\\wiki_seg.txt"));
        String line = null;
        while((line = br.readLine()) != null) {
            line = line.replaceAll("[\r\n\t]", ""); // 去掉特殊符号
            String sc = AHANLP.convertTC2SC(line); // 繁体转简体
            // 标准分词，去停用词
            List<String> words = AHANLP.getWordList(AHANLP.StandardSegment(sc, true));
            if (words.size() == 0) continue;
            for (String word : words) {
                bw.write(word + " "); 
            }
            bw.write("\n");
        }
        br.close();
        bw.flush();
        bw.close();
    }
}
```

至此，维基百科中文语料的处理就算是完成了。

## 通过 Word2Vec 训练词向量

Google 实现的 C 语言版的 word2vec 是目前公认的准确率最高的 word2vec 版本，因而训练更推荐直接使用 google 的原版。

下面提供两个 GitHub 上克隆的 Google word2vec 项目：

- [word2vec:](https://github.com/svn2github/word2vec) Google 原版的克隆
- [word2vec:](https://github.com/dav/word2vec) 在 Google 原版上稍作修改，可以在 MacOS 上编译。

下载后使用 `make` 命令编译，之后使用编译出的 **word2vec** 来训练模型：

```shell
./word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1
```

简单说明一下：
`TEXT_DATA` 为训练文本文件路径，词之间使用空格分隔；`VECTOR_DATA` 为输出的模型文件；不使用 cbow 模型，默认为 Skip-Gram 模型；每个单词的向量维度是 200；训练的窗口大小为 5；不使用 NEG 方法，使用 HS 方法；`-sampe` 指的是采样的阈值，如果一个词语在训练样本中出现的频率越大，那么就越会被采样；`-binary` 为 1 指的是结果二进制存储，为 0 是普通存储。

训练完成后，可以将得到的模型放置到 AHANLP 项目的 `data/model/` 目录下，然后在 ahanlp.properties 文件中进行配置，把 `word2vecModel = data/model/wiki_chinese_word2vec(Google).model` 项设置为我们自己的模型名称。

配置好后就可以调用 AHANLP 中的 **wordSimilarity** 和 **sentenceSimilarity** 函数进行测试了，例如：

```java
System.out.println("猫 | 狗 : " + AHANLP.wordSimilarity("猫", "狗"));
System.out.println("计算机 | 电脑 : " + AHANLP.wordSimilarity("计算机", "电脑"));
System.out.println("计算机 | 男人 : " + AHANLP.wordSimilarity("计算机", "男人"));

String s1 = "苏州有多条公路正在施工，造成局部地区汽车行驶非常缓慢。";
String s2 = "苏州最近有多条公路在施工，导致部分地区交通拥堵，汽车难以通行。";
String s3 = "苏州是一座美丽的城市，四季分明，雨量充沛。";
System.out.println("s1 | s1 : " + AHANLP.sentenceSimilarity(s1, s1));
System.out.println("s1 | s2 : " + AHANLP.sentenceSimilarity(s1, s2));
System.out.println("s1 | s3 : " + AHANLP.sentenceSimilarity(s1, s3));
```

## 参考

吴良超的学习笔记[《中文维基百科语料库词向量的训练》](http://wulc.me/2016/10/12/%E4%B8%AD%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E7%9A%84%E8%AF%8D%E5%90%91%E9%87%8F%E7%9A%84%E8%AE%AD%E7%BB%83/)

