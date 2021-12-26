---
layout: article
title: Python 多进程、多线程（上）：让你的程序飞起来吧
author: 廖雪峰
tags:
    - NLP
mathjax: false
---

> 本文摘自廖雪峰[《Python教程》](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)，部分内容有修改。

## 多进程与多线程

我们都知道，操作系统中所有的程序都是以进程的方式来运行的，或者说我们把运行着的程序称为**进程(Process)**。例如运行记事本程序就是启动一个记事本进程，运行两个记事本就是启动两个记事本进程。

很多时候，进程还不止同时干一件事，比如Word，它可以同时进行打字、拼写检查、打印等事情。在一个进程内部，要同时干多件事，就需要同时运行多个“子任务”，我们把进程内的这些“子任务”称为**线程(Thread)**。由于每个进程至少要干一件事，所以，一个进程至少有一个线程。

进程和线程的区别主要有：

- 进程之间是相互独立的，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，且互不影响；而同一个进程的多个线程是内存共享的，所有变量都由所有线程共享。
- 由于进程间是独立的，因此一个进程的崩溃不会影响到其他进程；而线程是包含在进程之内的，线程的崩溃就会引发进程的崩溃，继而导致同一进程内的其他线程也奔溃。

真正的多进程或者多线程都需要多核 CPU 才可能实现。现在虽然多核 CPU 已经非常普及，但是任务数量远远多于 CPU 的核心数量，所以，操作系统会自动地把进程或线程轮流调度到每个核心上执行。例如对于核心 a，进程 1 执行 0.01 秒，切换到进程 2，进程 2 执行 0.01 秒，再切换到进程 3……每个进程都是交替执行的，但是由于 CPU 的执行速度实在是太快了，我们感觉就像这些进程都在同时执行一样。多线程的执行方式类似，每一个核心都在操作系统的调度下，在多个线程之间快速切换，让每个线程都短暂地交替运行。

目前，要同时完成多个任务通常有以下几种解决方案：

- 一种是启动多个进程，每个进程虽然只有一个线程，但多个进程可以一块执行多个任务。
- 还有一种方法是启动一个进程，在一个进程内启动多个线程，这样，多个线程也可以一块执行多个任务。
- 当然还有第三种方法，就是启动多个进程，每个进程再启动多个线程，这样同时执行的任务就更多了。但是这种模型很复杂，实际很少采用。

总结一下就是，多任务的实现有3种方式：

- 多进程模式；
- 多线程模式；
- 多进程+多线程模式。

同时执行多个任务通常各个任务之间并不是没有关联的，而是需要相互通信和协调，有时，任务 1 必须暂停等待任务 2 完成后才能继续执行，有时，任务 3 和任务 4 又不能同时执行。所以，多进程和多线程的程序的复杂度要远远高于单进程单线程的程序。

## Python 多进程

对于 Unix/Linux 操作系统来说，系统本身就提供了一个 `fork()` 系统调用，可以非常方便地创建多进程。Python的 **os** 模块封装了常见的系统调用，其中就包括 `fork`：

```python
import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
```

运行结果如下：

```shell
Process (1806) start...
I (1806) just created a child process (1809).
I am child process (1809) and my parent is 1806.
```

> 普通的函数调用，调用一次，返回一次，但是 `fork()` 调用一次，返回两次，因为操作系统自动把当前进程（父进程）复制了一份（子进程），然后，分别在父进程和子进程内返回。子进程永远返回 0，而父进程返回子进程的 ID。

### multiprocessing 模块

由于 Windows 系统没有 `fork` 调用，因而 Python 提供了一个跨平台的多进程模块 **multiprocessing**，模块中使用 **Process** 类来代表一个进程对象。下面的例子演示了启动一个子进程并等待其结束：

```python
from multiprocessing import Process
import os

# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
```

执行结果如下：

```shell
Parent process 928.
Process will start.
Run child process test (929)...
Process end.
```

创建子进程时，只需要传入一个执行函数和函数的参数（target 指定了进程要执行的函数，args 指定了参数）。创建好进程 Process 的实例后，使用 `start()` 方法启动。`join()` 方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步。

### 进程池 Pool

如果要启动大量的子进程，可以用进程池的方式批量创建子进程：

```python
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4) # 设置进程池大小
    for i in range(5):
        p.apply_async(long_time_task, args=(i,)) # 设置每个进程要执行的函数和参数
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
```

 执行结果如下：

```shell
Parent process 669.
Waiting for all subprocesses done...
Run task 0 (671)...
Run task 1 (672)...
Run task 2 (673)...
Run task 3 (674)...
Task 2 runs 0.14 seconds.
Run task 4 (673)...
Task 1 runs 0.27 seconds.
Task 3 runs 0.86 seconds.
Task 0 runs 1.41 seconds.
Task 4 runs 1.91 seconds.
All subprocesses done.
```

在上面的代码中，Pool 用于生成进程池，对 Pool 对象调用 `apply_async()` 方法可以使每个进程异步执行任务，也就说不用等上一个任务执行完才执行下一个任务。对 Pool 对象调用 `join()` 方法会等待所有子进程执行完毕，调用 `join()` 之前必须先调用 `close()` 以关闭进程池，确保没有新的进程加入。

> 输出的结果中，task 4 要等待前面某个 task 完成后才执行，这是因为我们把 Pool 的大小设置成了 4，因此，最多同时执行 4 个进程。如果改成 `p = Pool(5)`，就可以同时跑5个进程。Pool 的默认大小是 CPU 的核数。

### 进程间通信

Process 之间肯定是需要通信的，操作系统提供了很多机制来实现进程间的通信。Python 的 **multiprocessing** 模块包装了底层的机制，提供了队列(Queue)、管道(Pipes)等多种方式来交换数据。

我们以队列(Queue)为例，在父进程中创建两个子进程，一个往队列里写数据，一个从队列里读数据：

```Python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    q = Queue() # 父进程创建 Queue，并传给各个子进程：
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start() # 启动子进程 pw，写入
    pr.start() # 启动子进程 pr，读取
    pw.join() # 等待pw结束
    pr.terminate() # pr进程里是死循环，无法等待其结束，只能强行终止:
```

运行结果如下：

```python
Process to write: 50563
Put A to queue...
Process to read: 50564
Get A from queue.
Put B to queue...
Get B from queue.
Put C to queue...
Get C from queue.
```
## Python 多线程

多任务可以由多进程完成，也可以由一个进程内的多线程完成。我们前面提到了进程是由若干线程组成的，一个进程至少有一个线程。由于线程是操作系统直接支持的执行单元，因此，高级语言通常都内置多线程的支持，Python 也不例外，并且，Python 的线程是真正的 Posix Thread，而不是模拟出来的线程。

Python 的标准库提供了两个模块：**_thread** 和 **threading**。其中，**_thread** 是低级模块，**threading** 是高级模块，对 **_thread** 进行了封装。绝大多数情况下，我们只需要使用 **threading** 这个高级模块。

启动一个线程就是把一个函数传入并创建  Thread 实例，然后调用 `start()` 开始执行：

```python
import time, threading

# 新线程执行的代码:
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)

print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop, name='LoopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)
```

执行结果如下：

```shell
thread MainThread is running...
thread LoopThread is running...
thread LoopThread >>> 1
thread LoopThread >>> 2
thread LoopThread >>> 3
thread LoopThread >>> 4
thread LoopThread >>> 5
thread LoopThread ended.
thread MainThread ended.
```

由于任何进程默认就会启动一个线程，我们把该线程称为主线程，主线程又可以启动新的线程。Python 的 **threading** 模块有个 `current_thread()` 函数，它永远返回当前线程的实例。主线程实例的名字叫 MainThread，子线程的名字在创建时指定。上例中，我们用 LoopThread 命名子线程，如果不起名字 Python 就自动给线程命名为 Thread-1，Thread-2……

### Lock

多线程和多进程最大的不同在于：多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响；而多线程中，所有变量由所有线程共享。因为任何一个线程都可以修改任何一个变量，所以线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。

来看看多个线程同时操作一个变量怎么把内容给改乱了：

```python
import time, threading

balance = 0 # 假定这是你的银行存款:

# 先存后取，结果应该为0
def change_it(n):
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(100000):
        change_it(n)

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
```

我们定义了一个共享变量 balance，初始值为 0，并且启动两个线程，先存后取，理论上结果应该为 0。但是，由于线程的调度是由操作系统决定的，当 t1、t2 交替执行时，只要循环次数足够多，balance 的结果就不一定是 0 了。

原因是因为高级语言的一条语句在 CPU 执行时是若干条语句。例如一个简单的计算 `balance = balance + n`，也分两步：

- 计算 balance + n，存入临时变量中；
- 将临时变量的值赋给 balance。

如果操作系统以下面的顺序执行 t1、t2：

```
初始值 balance = 0

t1: x1 = balance + 5  # x1 = 0 + 5 = 5

t2: x2 = balance + 8  # x2 = 0 + 8 = 8
t2: balance = x2      # balance = 8

t1: balance = x1      # balance = 5
t1: x1 = balance - 5  # x1 = 5 - 5 = 0
t1: balance = x1      # balance = 0

t2: x2 = balance - 8  # x2 = 0 - 8 = -8
t2: balance = x2   # balance = -8

结果 balance = -8
```

究其原因，是因为修改 balance 需要多条语句，而执行这几条语句时，线程可能中断，从而导致多个线程把同一个对象的内容改乱了。

如果我们要确保 balance 计算正确，就要给 `change_it()` 上一把锁。当某个线程开始执行 `change_it()` 时，由于该线程获得了锁，因此其他线程不能同时执行 `change_it()`，只能等待，直到锁被释放，这样就可以避免修改的冲突。创建一个锁可以通过 `threading.Lock()` 来实现：

```python
balance = 0
lock = threading.Lock()

def run_thread(n):
    for i in range(100000):
        # 先要获取锁:
        lock.acquire()
        try:
            # 放心地改吧:
            change_it(n)
        finally:
            # 改完了一定要释放锁:
            lock.release()
```

当多个线程同时执行 `lock.acquire()` 时，只有一个线程能成功地获取锁，然后继续执行代码，其他线程就继续等待直到获得锁为止。获得锁的线程用完后一定要释放锁，否则那些苦苦等待锁的线程将永远等待下去，成为死线程，所以我们用 `try...finally` 来确保锁一定会被释放。

> 锁的好处就是确保了某段关键代码只能由一个线程从头到尾完整地执行，坏处当然也很多。首先是阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行。其次，由于可以存在多个锁，不同的线程持有不同的锁，并试图获取对方持有的锁时，可能会造成死锁，导致多个线程全部挂起，既不能执行，也无法结束，只能靠操作系统强制终止。

### GIL 锁

Python 的线程虽然是真正的线程，但解释器执行代码时，有一个 GIL 锁(Global Interpreter Lock)，任何 Python 线程执行前，必须先获得 GIL 锁。每执行 100 条字节码，解释器就自动释放 GIL 锁，让别的线程有机会执行。这个 GIL 全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在 Python 中只能交替执行，即使 100 个线程跑在 100 核 CPU 上，也只能用到 1 个核。

GIL 是 Python 解释器设计的历史遗留问题，通常我们用的解释器是官方实现的 CPython，要真正利用多核，除非重写一个不带 GIL 的解释器。所以，在 Python 如果一定要通过多线程利用多核，那只能通过 C 扩展来实现。

因而，多线程的并发在 Python 中就是一个美丽的梦，如果想真正实现多核任务，还是通过多进程来实现吧。

> 本文摘自廖雪峰[《Python教程》](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)，部分内容有修改。