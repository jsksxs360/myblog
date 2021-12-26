---
layout: article
title: Python 多进程、多线程（下）：让你的程序飞起来吧
author: 廖雪峰
tags:
    - NLP
mathjax: false
---

> 本文摘自廖雪峰[《Python教程》](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)，部分内容有修改。

## 进程 vs 线程

在上一篇[《Python 多进程、多线程（上）》](/2017/04/01/python-multi-process-1.html)中我们介绍了多进程和多线程，这是实现多任务最常用的两种方式。接下来我们讨论一下这两种方式的优缺点。

要实现多任务，通常我们会设计 Master-Worker 模式，Master 负责分配任务，Worker 负责执行任务。因此，多任务环境下，通常是一个 Master，多个 Worker。如果用多进程实现 Master-Worker，主进程就是 Master，其他进程就是 Worker。如果用多线程实现 Master-Worker，主线程就是 Master，其他线程就是 Worker。

- 多进程模式最大的优点就是稳定性高，因为一个子进程崩溃了，不会影响主进程和其他子进程。（当然主进程挂了所有进程就全挂了，但是 Master 进程只负责分配任务，挂掉的概率低）著名的 Apache 最早就是采用多进程模式。
- 多进程模式的缺点是创建进程的代价大，在 Unix/Linux 系统下，用 `fork` 调用还行，在 Windows 下创建进程开销巨大。另外，操作系统能同时运行的进程数也是有限的，在内存和 CPU 的限制下，如果有几千个进程同时运行，操作系统连调度都会成问题。

多线程模式通常比多进程快一点，但是也快不到哪去，而且，多线程模式致命的缺点就是任何一个线程挂掉都可能直接造成整个进程崩溃，因为所有线程共享进程的内存。在 Windows 上，如果一个线程执行的代码出了问题，操作系统会强制结束整个进程。

在 Windows 下，多线程的效率比多进程要高，所以微软的 IIS 服务器默认采用多线程模式。由于多线程存在稳定性的问题，IIS 的稳定性就不如 Apache。为了缓解这个问题，IIS 和 Apache 现在又有多进程+多线程的混合模式，真是把问题越搞越复杂。

## 计算密集型 vs IO密集型

我们可以把任务分为计算密集型和 IO 密集型。

### 计算密集型任务

计算密集型任务的特点是要进行大量的计算，消耗 CPU 资源，比如计算圆周率、对视频进行高清解码等等，全靠 CPU 的运算能力。这种计算密集型任务虽然也可以用多任务完成，但是任务越多，花在任务切换的时间就越多， CPU 执行任务的效率就越低，所以，要最高效地利用 CPU，计算密集型任务同时进行的数量应当等于 CPU 的核心数。

计算密集型任务由于主要消耗 CPU 资源，因此，代码运行效率至关重要。Python 这样的脚本语言运行效率很低，完全不适合计算密集型任务。对于计算密集型任务，最好用 C 语言编写。

### IO 密集型任务

涉及到网络、磁盘 IO 的任务都是 IO 密集型任务，这类任务的特点是 CPU 消耗很少，任务的大部分时间都在等待 IO 操作完成（因为 IO 的速度远远低于 CPU 和内存的速度）。对于 IO 密集型任务，任务越多，CPU 效率越高，但也有一个限度。常见的大部分任务都是 IO 密集型任务，比如 Web 应用。

IO 密集型任务执行期间，99% 的时间都花在 IO 上，花在 CPU 上的时间很少，因此，用运行速度极快的 C 语言完全无法提升运行效率。对于IO密集型任务，最合适的语言就是开发效率最高（代码量最少）的语言，脚本语言是首选，C语言最差。

## 分布式进程

Python 的 **multiprocessing** 模块不但支持多进程，其中 **managers** 子模块还支持把多进程分布到多台机器上。一个服务进程可以作为调度者，将任务分布到其他多个进程中，依靠网络通信。由于 **managers** 模块封装很好，不必了解网络通信的细节，就可以很容易地编写分布式多进程程序。

例如，我们的机器上有一个通过队列(Queue)通信的多进程程序在运行，现在希望把发送任务的进程和处理任务的进程分布到两台机器上。这时就可以使用分布式进程来实现：通过 **managers** 模块把队列通过网络暴露出去，让其他机器的进程也可以访问。

我们先看服务进程，服务进程负责启动队列(Queue)，并且把队列注册到网络上，然后往队列里面写入任务：

```python
# task_master.py

import random, time, queue
from multiprocessing.managers import BaseManager

task_queue = queue.Queue() # 发送任务的队列:
result_queue = queue.Queue() # 接收结果的队列:

# 从 BaseManager 继承的 QueueManager:
class QueueManager(BaseManager):
    pass

# 把两个 Queue 都注册到网络上, callable 参数关联了 Queue 对象:
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
# 绑定端口 5000, 设置验证码 'abc':
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动 Queue:
manager.start()
# 获得通过网络访问的 Queue 对象:
task = manager.get_task_queue()
result = manager.get_result_queue()
# 放几个任务进去:
for i in range(10):
    n = random.randint(0, 10000)
    print('Put task %d...' % n)
    task.put(n)
# 从 result 队列读取结果:
print('Try get results...')
for i in range(10):
    r = result.get(timeout=10)
    print('Result: %s' % r)
# 关闭:
manager.shutdown()
print('master exit.')
```

当我们在一台机器上写多进程程序时，创建的队列(Queue)可以直接拿来用。但是，在分布式多进程环境下，添加任务到队列不可以直接对原始的 task_queue 进行操作，那样就绕过了 QueueManager 的封装，必须通过 `manager.get_task_queue()` 获得的队列接口添加。

然后，在另一台机器上启动任务进程（本机上启动也可以）：

```python
# task_worker.py

import time, sys, queue
from multiprocessing.managers import BaseManager

# 创建类似的 QueueManager:
class QueueManager(BaseManager):
    pass

# 由于这个 QueueManager 只从网络上获取 Queue，所以注册时只提供名字:
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

# 连接到服务器，也就是运行 task_master.py 的机器:
server_addr = '127.0.0.1'
print('Connect to server %s...' % server_addr)
# 端口和验证码注意保持与 task_master.py 设置的完全一致:
m = QueueManager(address=(server_addr, 5000), authkey=b'abc')
# 从网络连接:
m.connect()
# 获取 Queue 的对象:
task = m.get_task_queue()
result = m.get_result_queue()
# 从task队列取任务,并把结果写入result队列:
for i in range(10):
    try:
        n = task.get(timeout=1)
        print('run task %d * %d...' % (n, n))
        r = '%d * %d = %d' % (n, n, n*n)
        time.sleep(1)
        result.put(r)
    except Queue.Empty:
        print('task queue is empty.')
# 处理结束:
print('worker exit.')
```

任务进程要通过网络连接到服务进程，所以要指定服务进程的 IP。

现在，可以试试分布式进程的工作效果了。先启动 task_master.py 服务进程：

```shell
$ python3 task_master.py 
Put task 3411...
Put task 1605...
Put task 1398...
Put task 4729...
Put task 5300...
Put task 7471...
Put task 68...
Put task 4219...
Put task 339...
Put task 7866...
Try get results...
```

task_master.py 进程发送完任务后，开始等待 result 队列的结果。现在启动 task_worker.py 进程：

```shell
$ python3 task_worker.py
Connect to server 127.0.0.1...
run task 3411 * 3411...
run task 1605 * 1605...
run task 1398 * 1398...
run task 4729 * 4729...
run task 5300 * 5300...
run task 7471 * 7471...
run task 68 * 68...
run task 4219 * 4219...
run task 339 * 339...
run task 7866 * 7866...
worker exit.
```

task_worker.py 进程结束，在 task_master.py 进程中会继续打印出结果：

```shell
Result: 3411 * 3411 = 11634921
Result: 1605 * 1605 = 2576025
Result: 1398 * 1398 = 1954404
Result: 4729 * 4729 = 22363441
Result: 5300 * 5300 = 28090000
Result: 7471 * 7471 = 55815841
Result: 68 * 68 = 4624
Result: 4219 * 4219 = 17799961
Result: 339 * 339 = 114921
Result: 7866 * 7866 = 61873956
```

这个简单的 Master/Worker 模型有什么用？其实这就是一个简单但真正的分布式计算，把代码稍加改造，启动多个 worker，就可以把任务分布到几台甚至几十台机器上，比如把计算 `n*n` 的代码换成发送邮件，就实现了邮件队列的异步发送。

注意到 task_worker.py 中根本没有创建队列的代码，所以队列(Queue)对象存储在 task_master.py 进程中：

<img src="/img/article/python-multi-process-2/queue.png" style="display:block;margin:auto;"/>

而队列之所以能通过网络访问，就是通过 QueueManager 实现的。由于 QueueManager 管理的不止一个队列，所以要给每个队列的网络调用接口起个名字，比如 `get_task_queue()`。

**authkey** 有什么用？这是为了保证两台机器正常通信，不被其他机器恶意干扰。如果 task_worker.py 和 task_master.py 的 authKey 不一致就连接不上。

Python 的分布式进程接口简单，封装良好，适合需要把繁重任务分布到多台机器的环境下。

> 队列(Queue)的作用是用来传递任务和接收结果，每个任务的描述数据量要尽量小。比如发送一个处理日志文件的任务，就应该发送日志文件的存放路径，而不是几百兆的日志文件本身，由 Worker 进程再去共享的磁盘上读取文件。

> 本文摘自廖雪峰[《Python教程》](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)，部分内容有修改。