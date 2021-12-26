---
layout: article
title: Python 模块、包和程序：程序组织结构与标准库
tags:
    - Python
mathjax: false
---

在“远离” Python 很久之后，再回看之前的代码，感觉对 Python 的 `import` 语句还是存在困惑。总是对 Python 程序的结构不是很清楚，不知道什么时候直接用 `import xxx`，什么时候又要用 `from xxx import xxx`。正好借这篇文章，梳理一下。而且要写出实用的大型 Python 程序，也确实要理清楚这些细节。废话不多说，我们开始。

## 独立的程序

独立的 Python 程序存放在以 **.py** 为后缀的文件中，例如创建一个文件 test1.py，包含以下代码：

```python
print("This standalone program works!")
```

如果要在文本终端或者终端窗口运行 Python，需要键入 Python 程序名，后面跟上程序的文件名：

```shell
$ python test1.py
This standalone program works!
```

## 命令行参数

通过 **sys** 模块，Python 程序可以很方便的获取到命令行传递给它的参数。例如创建文件 test2.py，包含下面两行：

```python
import sys
print('Program arguments:',sys.argv)
```

现在，使用 Python 运行这段程序。下面是在 Linux 或者 Mac OS X 系统的标准 shell 程序下的运行结果：

```shell
$ python test2.py
Program arguments: ['test2.py']
$ python test2.py tra la la
Program arguments: ['test2.py', 'tra', 'la', 'la']
```

## 模块和 import 语句

一个**模块**仅仅是 Python 代码的一个文件。

一本书的内容通常按照这样的层次组织：单词、句子、段落以及章。Python 代码也有类似的自底向上的组织层次：数据类型类似于单词，语句类似于句子，函数类似于段落，模块类似于章。

引用其他模块的代码时使用 `import` 语句，使被引用模块中的代码和变量对该程序可见。

### 导入模块

`import` 语句最简单的用法是 `import` 模块，模块是不带 .py 扩展的另外一个 Python 文件的文件名。现在来模拟一个气象站，并输出天气预报。其中一个主程序输出报告，一个单独的具有单个函数的模块返回天气的描述。

下面是主程序（**weatherman.py**）：

```python
import report

description = report.get_description()
print("Today's weather:", description)
```

以下是天气模块的代码（**report.py**）：

```python
def get_description():
    """Return random weather, just like the pros"""
    from random import choice #从标准模块 random 中导入函数 choice
    possibilities = ['rain', 'snow', 'sleet', 'fog', 'sun', 'who knows']
    return choice(possibilities)
```

如果上述两个文件在同一个目录下，通过 Python 运行主程序 weatherman.py，会引用 **report** 模块，执行函数 `get_description()`。函数 `get_description()` 从字符串列表中返回一个随机结果。例如：

```shell
$ python weatherman.py
Today's weather: who knows
$ python weatherman.py
Today's weather: sun
$ python weatherman.py
Today's weather: sleet
```

注意，我们以两种不同的方式使用了 `import`：

- 主程序调用 `import report`，然后运行 `report.get_description()`
- `get_description()` 函数调用 `from random import choice`，然后运行 `choice(possibilities)`

第一种情况下，我们导入了整个 **report** 模块，但是需要把 `report.` 作为 `get_description()` 的前缀。在这个 `import` 语句之后，只要在名称前加 `report.`，report.py 的所有内容（代码和变量）就会对主程序可见。**通过模块名称限定模块的内容，可以避免命名冲突。**因为其他模块可能也有函数 `get_description()`，这样做不会被错误地调用。

第二种情况下，所有代码都在同一个函数下，并且没有其他名为 `choice` 的函数，所以我们直接从 `random` 模块导入函数 `choice()`。

在函数 `get_description()` 中，我们直接在函数内部导入了 **random** 模块，我们也可以在函数外部导入：

```python
from random import choice

def get_description():
    possibilities = ['rain', 'snow', 'sleet', 'fog', 'sun', 'who knows']
    return choice(possibilities)
```

如果被导入的代码被多次使用，就应该考虑在函数外部导入；如果被导入的代码使用有限，就在函数内部导入。一些人更喜欢把所有的 `import` 都放在文件的开头，从而使代码之间的依赖关系清晰。两种方法都是可行的。

### 使用别名导入模块

在主程序 weatherman.py 中，我们调用了 `import report`。但是，如果存在同名的另一个模块或者你想使用更短更好记的名字，该如何做呢？在这种情况下，可以使用**别名** `wr` 进行导入：

```python
import report as wr
description = wr.get_description()
print("Today's weather:", description)
```

### 导入模块的一部分

在 Python 中，可以导入一个模块的若干部分。每一部分都有自己的原始名字或者你起的别名。首先，从 **report** 模块中用原始名字导入函数 `get_description()`：

```Python
from report import get_description
description = get_description()
print("Today's weather:", description)
```

用它的别名 `do_it` 导入：

```python
from report import get_description as do_it
description = do_it()
print("Today's weather:", description)
```

### 模块搜索路径

Python 会在什么地方寻找文件来导入模块？使用命名为 `path` 变量的存储在标准 **sys** 模块下的一系列目录名和 ZIP 压缩文件。你可以读取和修改这个列表。下面是 Python 3.3 的 `sys.path` 的内容：

```shell
>>> import sys
>>> for place in sys.path:
...        print(place)
...
/Library/Frameworks/Python.framework/Versions/3.3/lib/python33.zip
/Library/Frameworks/Python.framework/Versions/3.3/lib/python3.3
/Library/Frameworks/Python.framework/Versions/3.3/lib/python3.3/plat-darwin
/Library/Frameworks/Python.framework/Versions/3.3/lib/python3.3/lib-dynload
/Library/Frameworks/Python.framework/Versions/3.3/lib/python3.3/site-packages
```

最开始的空白输出行是空字符串 `''`，代表当前目录。如果空字符串是在 `sys.path` 的开始位置，Python 会先搜索当前目录：`import report` 会寻找文件 report.py。

第一个匹配到的模块会先被使用，这也就意味着如果你在标准库之前的搜索路径上定义一个模块 `random`，就不会导入标准库中的 `random` 模块。

## 包

为了使 Python 应用更具可扩展性，还可以把多个模块组织成文件层次，称之为**包**。

例如我们需要两种类型的天气预报：一种是次日的，一种是下周的。一种可行的方式是新建目录 sources，在该目录中新建两个模块 daily.py 和 weekly.py。每一个模块都有一个函数 `forecast`。每天的版本返回一个字符串，每周的版本返回包含 7 个字符串的列表。

下面是主程序和两个模块（函数 `enumerate()` 拆分一个列表，并对列表中的每一项通过 `for` 循环增加数字下标）。

主程序（**boxes/weather.py**）：

```python
from sources import daily, weekly

print("Daily forecast:", daily.forecast())
print("Weekly forecast:")
for number, outlook in enumerate(weekly.forecast(), 1):
    print(number, outlook)
```

模块 1 （**boxes/sources/daily.py**）：

```python
def forecast():
    'fake daily forecast'
    return 'like yesterday'
```

模块 2 （**boxes/sources/weekly.py**）：

```python
def forecast():
    """Fake weekly forecast"""
    return ['snow', 'more snow', 'sleet', 'freezing rain', 'rain', 'fog', 'hail']
```

还需要在 sources 目录下添加一个文件：init.py。这个文件可以是空的，但是 Python 需要它，以便把该目录作为一个包。

接下来，我们运行主程序 weather.py：

```shell
$ python weather.py
Daily forecast: like yesterday
Weekly forecast:
1 snow
2 more snow
3 sleet
4 freezing rain
5 rain
6 fog
7 hail
```

## Python 标准库

Python 的一个显著特点是具有庞大的模块标准库，这些模块可以执行很多有用的任务，并且和核心 Python 语言分开以避免臃肿。接下来我们讨论一些常用的标准模块。

### 使用 setdefault() 和 defaultdict() 处理缺失的键

读取字典中不存在的键的值会抛出异常，可以使用字典函数 `get()` 返回一个默认值会避免异常发生。函数 `setdefault()` 类似于 `get()`，但当键不存在时它会在字典中添加一项：

```shell
>>> periodic_table = {'Hydrogen': 1, 'Helium': 2}
>>> carbon = periodic_table.setdefault('Carbon', 12)
>>> carbon
12
>>> periodic_table
{'Helium': 2, 'Carbon': 12, 'Hydrogen': 1}
```

如果试图把一个不同的默认值赋给已经存在的键，不会改变原来的值，仍将返回初始值：

```shell
>>> helium = periodic_table.setdefault('Helium', 947)
>>> helium
2
>>> periodic_table
{'Helium': 2, 'Carbon': 12, 'Hydrogen': 1}
```

`defaultdict()` 也有同样的用法，但是在创建字典时，对每个新的键都会指定默认值。它的参数是一个函数。在本例中，把函数 **int** 作为参数传入，会按照 `int()` 调用，返回整数 0：

```shell
>>> from collections import defaultdict
>>> periodic_table = defaultdict(int)
>>> periodic_table['Hydrogen'] = 1
>>> periodic_table['Lead']
0
>>> periodic_table
defaultdict(<class 'int'>, {'Lead': 0, 'Hydrogen': 1})
```

注意，函数 `defaultdict()` 的参数是一个函数，它返回赋给缺失键的值。在下面的例子中，`no_idea()` 在需要时会被执行，返回一个值：

```shell
>>> from collections import defaultdict
>>>
>>> def no_idea():
...     return 'Huh?'
...
>>> bestiary = defaultdict(no_idea)
>>> bestiary['A'] = 'Abominable Snowman'
>>> bestiary['B'] = 'Basilisk'
>>> bestiary['A']
'Abominable Snowman'
>>> bestiary['B']
'Basilisk'
>>> bestiary['C']
'Huh?'
```

同样，可以使用函数 `int()`、`list()` 或者 `dict()` 返回默认空的值：`int()` 返回 `0`，`list()` 返回空列表（`[]`），`dict()` 返回空字典（`{}`）。如果你删掉该函数参数，新键的初始值会被设置为 `None`。

顺便提一下，也可以使用 `lambda` 来定义你的默认值函数：

```shell
>>> bestiary = defaultdict(lambda: 'Huh?')
>>> bestiary['E']
'Huh?'
```

使用 `int` 是一种定义计数器的方式：

```shell
>>> from collections import defaultdict
>>> food_counter = defaultdict(int)
>>> for food in ['spam', 'spam', 'eggs', 'spam']:
...     food_counter[food] += 1
...
>>> for food, count in food_counter.items():
...     print(food, count)
...
eggs 1
spam 3
```

如果 `food_counter` 是一个普通的字典，那每次试图自增字典元素 `food_counter[food]` 值时，Python 会抛出一个异常，因为我们没有对它进行初始化。我们需要做额外的工作，如下所示：

```shell
>>> dict_counter = {}
>>> for food in ['spam', 'spam', 'eggs', 'spam']:
...     if not food in dict_counter:
...         dict_counter[food] = 0
...     dict_counter[food] += 1
...
>>> for food, count in dict_counter.items():
...     print(food, count)
...
spam 3
eggs 1
```

### 使用 Counter() 计数

Python 标准库有一个计数器，它可以胜任很多工作：

```shell
>>> from collections import Counter
>>> breakfast = ['spam', 'spam', 'eggs', 'spam']
>>> breakfast_counter = Counter(breakfast)
>>> breakfast_counter
Counter({'spam': 3, 'eggs': 1})
```

函数 `most_common()` 以降序返回所有元素，或者如果给定一个数字，会返回该数字前的的元素：

```shell
>>> breakfast_counter.most_common()
[('spam', 3), ('eggs', 1)]
>>> breakfast_counter.most_common(1)
[('spam', 3)]
```

也可以组合计数器。首先来看一下 `breakfast_counter`：

```shell
>>> breakfast_counter
>>> Counter({'spam': 3, 'eggs': 1})
```

这一次，新建一个列表 `lunch` 和一个计数器 `lunch_counter`：

```shell
>>> lunch = ['eggs', 'eggs', 'bacon']
>>> lunch_counter = Counter(lunch)
>>> lunch_counter
Counter({'eggs': 2, 'bacon': 1})
```

第一种组合计数器的方式是使用 `+`：

```shell
>>> breakfast_counter + lunch_counter
Counter({'spam': 3, 'eggs': 3, 'bacon': 1})
```

你也可能想到，从一个计数器去掉另一个，可以使用 `-`。什么是早餐有的而午餐没有的呢？

```shell
>>> breakfast_counter - lunch_counter
Counter({'spam': 3})
```

那么什么又是午餐有的而早餐没有的呢 ?

```shell
>>> lunch_counter - breakfast_counter
Counter({'bacon': 1, 'eggs': 1})
```

还可以使用交集运算符 `&` 得到二者共有的项：

```shell
>>> breakfast_counter & lunch_counter
Counter({'eggs': 1})
```

两者的交集通过取两者中的较小计数，得到共同元素 `'eggs'`。这合情合理：早餐仅提供一个鸡蛋，因此也是共有的计数。最后，使用并集运算符 `|` 得到所有元素：

```shell
>>> breakfast_counter | lunch_counter
Counter({'spam': 3, 'eggs': 2, 'bacon': 1})
```

`'eggs'` 又是两者共有的项。不同于合并，并集没有把计数加起来，而是取其中较大的值。

### 使用有序字典 OrderedDict() 按键排序

有序字典 `OrderedDict()` 记忆字典键添加的顺序，然后从一个迭代器按照相同的顺序返回。试着用元组（**键**，**值**）创建一个有序字典：

```shell
>>> from collections import OrderedDict
>>> quotes = OrderedDict([
...     ('Moe', 'A wise guy, huh?'),
...     ('Larry', 'Ow!'),
...     ('Curly', 'Nyuk nyuk!'),
...     ])
>>>
>>> for stooge in quotes:
...     print(stooge)
...
Moe
Larry
Curly
```

### 双端队列：栈 + 队列

`deque` 是一种双端队列，同时具有栈和队列的特征，它可以从序列的任何一端添加和删除项。

现在，我们从一个词的两端扫向中间，判断是否为回文。函数 `popleft()` 去掉最左边的项并返回该项，`pop()` 去掉最右边的项并返回该项。从两边一直向中间扫描，只要两端的字符匹配，一直弹出直到到达中间：

```shell
>>> def palindrome(word):
...     from collections import deque
...     dq = deque(word)
...     while len(dq) > 1:
...        if dq.popleft() != dq.pop():
...            return False
...     return True
...
...
>>> palindrome('a')
True
>>> palindrome('racecar')
True
>>> palindrome('')
True
>>> palindrome('radar')
True
>>> palindrome('halibut')
False
```

这里把判断回文作为双端队列的一个简单说明。如果想要写一个快速的判断回文的程序，只需要把字符串反转和原字符串进行比较。Python 没有对字符串进行反转的函数 `reverse()`，但还是可以利用反向切片的方式进行反转，如下所示：

```shell
>>> def another_palindrome(word):
...     return word == word[::-1]
...
>>> another_palindrome('radar')
True
>>> another_palindrome('halibut')
False
```

### 使用 itertools 迭代代码结构

[**itertools**]([https://docs.python.org/3/library/itertools.htm](https://docs.python.org/3/library/itertools.html)) 包含特殊用途的迭代器函数。在 `for ... in` 循环中调用迭代函数，每次会返回一项，并记住当前调用的状态。

即使 `chain()` 的参数只是单个迭代对象，它也会使用参数进行迭代：

```shell
>>> import itertools
>>> for item in itertools.chain([1, 2], ['a', 'b']):
...     print(item)
...
1
2
a
b
```

`cycle()` 是一个在它的参数之间循环的无限迭代器：

```shell
>>> import itertools
>>> for item in itertools.cycle([1, 2]):
...     print(item)
...
1
2
1
2
.
.
.
```

`accumulate()` 计算累积的值。默认的话，它计算的是累加和：

```shell
>>> import itertools
>>> for item in itertools.accumulate([1, 2, 3, 4]):
...     print(item)
...
1
3
6
10
```

你可以把一个函数作为 `accumulate()` 的第二个参数，代替默认的加法函数。这个参数函数应该接受两个参数，返回单个结果。下面的例子计算的是乘积：

```shell
>>> import itertools
>>> def multiply(a, b):
...     return a * b
...
>>> for item in itertools.accumulate([1, 2, 3, 4], multiply):
...     print(item)
...
1
2
6
24
```

`itertools` 模块有很多其他的函数，有一些可以用在需要节省时间的组合和排列问题上。

### 使用 pprint() 友好输出

我们通常都使用 `print()` 打印输出，但有时输出结果的可读性较差。这时，我们需要一个友好输出函数，比如 `pprint()`：

```shell
>>> from pprint import pprint
>>> quotes = OrderedDict([
...     ('Moe', 'A wise guy, huh?'),
...     ('Larry', 'Ow!'),
...     ('Curly', 'Nyuk nyuk!'),
...     ])
>>>
```

普通的 `print()` 直接列出所有结果：

```shell
>>> print(quotes)
OrderedDict([('Moe', 'A wise guy, huh?'), ('Larry', 'Ow!'), ('Curly', 'Nyuk nyuk!')])
```

但是，`pprint()` 尽量排列输出元素从而增加可读性：

```shell
>>> pprint(quotes)
{'Moe': 'A wise guy, huh?',
 'Larry': 'Ow!',
 'Curly': 'Nyuk nyuk!'}
```
## 参考

Bill Lubanovic[《Python语言及其应用》](http://www.ituring.com.cn/book/1560)
