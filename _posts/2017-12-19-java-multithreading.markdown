---
layout: article
title: "Fork Join Java 并行编程：轻松实现 Java 多线程"
tags:
    - Java
mathjax: true
---

实现程序高性能的一个重要技术手段，就是将密集的任务分隔成多个可以并行执行的块，以便可以最大化利用机器的计算能力。如今多核处理器已经非常普遍，这给了我们发挥并发编程潜能的机会，因为多个线程可以在多个内核上并发执行。但是编写并发(并行)程序一向比较困难，因为你必须处理线程同步和共享数据的问题。

好在 java 平台在语言级别上对并发编程就提供了支持，Java SE 5 和 Java SE 6 通过引入了一组包提供并发模块，Java SE 7 又进一步增强它们。

<img src="/img/article/java-multithreading/java.jpeg" style="display:block;margin:auto;"/>

## 早期版本的写法

### Java 中普通线程的并发编程

从历史上来看，java 通过 java.lang.Thread 类和 java.lang.Runnable 接口来实现多线程编程，以确保代码对于共享的可变对象表现出正确性和一致性，并且避免不正确的读/写操作，同时不会由于竞争条件上的锁争用而产生死锁。这里是一个基本的线程操作的例子：

```java
Thread thread = new Thread() {
     @Override
     public void run() {
          System.out.println("I am running in a separate thread!");
     }
};
thread.start();
thread.join();
```

上面的代码创建了一个线程，并打印一个字符串到标准输出。通过调用 `join()` 方法，主线程将等待创建的子线程执行完成。对于简单的例子，直接操作线程这种方式是可以的，但对于并发编程，这样的代码很快变得容易出错，特别是好几个线程需要协作来完成一个更大的任务的时候。这种情况下，它们的控制流需要被协调。

例如，一个线程的执行完成可能依赖于其他将要执行完成的线程。通常熟悉的例子就是生产者/消费者的例子：如果消费者队列是满的，那么生产者应该等待消费者；如果生产者队列是空的，那么消费者也应该等待生产者。该需求可以通过共享状态和条件队列来实现，但是你仍然必须通过使用共享对象上的 `java.lang.Object.nofity()` 和 `java.lang.Object.wait()` 来实现同步，这很容易出错。

一个常见的错误就是在大段代码甚至整个方法上使用 synchronize 进行互斥。虽然这种方法能实现线程安全，但是通常由于排斥时间太长而限制了并行性，从而造成性能低下。

### Executor 增强线程

Java SE5 引入了一个叫 java.util.concurrent 的包家族，在 Java SE6 中得到进一步增强。该包家族提供了 Executors 作为增强线程，它们执行任务类似于传递线程(实际上是封装了实现 java.util.Runnable 的实例)。作为一个例子，让我们看看下面的程序：

> Java SE7 引入了新的整数字面值，下划线可以在任何地方插入以提高可读性(比如，1_000_000)。

```java
import java.util.*;
import java.util.concurrent.*;
import static java.util.Arrays.asList;

public class Sums {

  static class Sum implements Callable<Long> {
     private final long from;
     private final long to;
     Sum(long from, long to) {
         this.from = from;
         this.to = to;
     }

     @Override
     public Long call() {
         long acc = 0;
         for (long i = from; i <= to; i++) {
             acc = acc + i;
         }
         return acc;
     }
  }

  public static void main(String[] args) throws Exception {
     ExecutorService executor = Executors.newFixedThreadPool(2);
     List <Future<Long>> results = executor.invokeAll(asList(
         new Sum(0, 10), new Sum(100, 1_000), new Sum(10_000, 1_000_000)
     ));
     executor.shutdown();

     for (Future<Long> result : results) {
         System.out.println(result.get());
     }
  }
}
```

这个例子利用 executor 来计算长整形数值的和。内部的 Sum 类实现了 `Callable` 接口，并被 excutors 用来执行计算，而并发工作则放在 `call` 方法中执行。java.util.concurrent.Executors 类提供了好几个工具方法，比如提供预先配置的 Executors 和包装普通 java.util.Runnable 对象的 Callable 实例。

> 使用 Callable 比 Runnable 更优势的地方在于 Callable 可以有确切的返回值。

该例子使用 executor 分发工作给 2 个线程。`ExecutorService.invokeAll()` 方法放入 Callable 实例的集合，并且等待直到它们都返回。其返回的 Future 对象列表，代表了计算的“未来”结果。如果我们想以异步的方式执行，我们可以检测每个 Future 对象对应的 Callable 是否完成了它的工作和是否抛出了异常，我们甚至可以取消它。因为 `invokeAll()` 是阻塞的，我们可以直接迭代 Future 实例来获取它们的计算和。

> 注意 executor 服务必须被关闭。如果它没有被关闭，主方法执行完后 JVM 不会退出，因为仍然有激活线程存在。

Executors 相对于普通的线程已经是一个很大的进步，因为 executors 很容易管理并发任务。

## Fork/Join 框架

### 概览

有些类型的算法需要创建子任务，并且让它们彼此通信来完成任务。这些都是”分而治之”的算法，也被称为”map and reduce”。想法是将数据区通过算法处理分隔为更小切独立的块，这是”map”阶段。反过来，一旦这些块处理完成了，各部分的结果就可以收集起来以产生最终的结果，这就是”reduce”阶段。

一个简单的例子想要计算出一个庞大的整形数组的和(如下图)。由于加法是可交换的，可以拆分数组为更小的部分，并且用并发线程计算各部分和，最后各部分和再累加从而计算出总和。因为线程可以独立对一个数组的不同区域使用这种算法操作。相比于单线程算法(迭代数组中每个整形)，你将看到在多核架构中有了明显的性能提升。

<img src="/img/article/java-multithreading/multi_thread.png" style="display:block;margin:auto;"/>

通过 executors 解决上面的问题是很容易的：将数组分为 $n$ 部分，创建 Callable 实例来计算每一部分的和，提交它们到一个管理了 $n$ 个线程的池中，并且收集结果计算出最终结果。

然而，对其他类型的算法和数据结构，其执行计划并不是那么简单。特别是，识别出要以有效的方式被独立处理的“足够小”的数据块的”map”阶段并不能提前知道到数据空间的拓扑结构。基于图和基于树的数据结构尤为如此。在这些情况下，算法应该创建层级”划分”，即在部分结果返回之前等待子任务完成。

为了实现分而治之算法，executors 可以创建不相关的子任务，因为一个 Callable 可以无限制的提交一个新的子任务给它的 executors，并且以同步或异步的方式等待它的结果。问题是并行：当一个 Callable 等待另一个 Callable 的结果时，它就处于等待状态，从而浪费了一个机会来处理队列中等待执行的另一个 Callable。

通过 Doug Lea 努力填补了这一缺陷，在 Java SE7 中，**fork/join** 框架被加到了 java.util.concurrent 包中。

### 添加支持并行

**fork/join** 框架最核心的贡献，是添加了新的 ForkJoinPool 执行者，专门执行实现了 ForkJoinTask 接口的实例。ForkJoinTask 对象支持创建子任务来等待子任务完成。当一个任务正在等待另一个任务完成并且有待执行的任务时，executor 就能够通过”偷取”任务，在内部的线程池里分发任务。

ForkJoinTask 对象主要有两个重要的方法：

- `fork()` 方法允许 ForkJoinTask 任务异步执行，也允许一个新的 ForkJoinTask 从存在的 ForkJoinTask 中被启动。
- `join()` 方法允许一个 ForkJoinTask 等待另一个 ForkJoinTask 执行完成。

如下图所示，通过 `fork()` 和 `join()` 实现任务间的相互合作。

> `fork()` 和 `join()` 方法名称不应该与 POSIX 中的进程能够复制自己的过程相混淆。`fork()` 只会让 ForkJoinPool 调度一个新的任务，而不会创建子虚拟机。

<img src="/img/article/java-multithreading/fork_and_join.png" style="display:block;margin:auto;"/>

ForkJoinTask 提供了以下两个子类：

- **RecursiveAction**：用于没有返回结果的任务。
- **RecursiveTask**：用于有返回结果的任务。

通常，RecursiveTask 是首选的，大部分分而治之的算法会在数据集上计算后返回结果。

最后，也是最重要的一点，fork/join 任务应该是纯内存算法，而没有 I/O 操作。此外，应该尽可能避免通过共享状态来进行任务间的通信，因为这通常意味着加锁会被执行。理想情况下，仅当一个任务 fork 另一个任务或一个任务 join 另一个任务时才进行任务通信。

### 例子：整形数组求和

为了阐述新的 fork/join 框架的使用，让我们首先用一个简单的例子来说明。例如前面提到的整形数组求和使用 fork/join 框架可以这样写：

```java
class SumTask extends RecursiveTask<Long> {

    static final int THRESHOLD = 100;
    int[] array;
    int start;
    int end;

    SumTask(int[] array, int start, int end) {
        this.array = array;
        this.start = start;
        this.end = end;
    }

    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // 如果任务足够小,直接计算:
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += array[i];
            }
            System.out.println(String.format("compute %d~%d = %d", start, end, sum));
            return sum;
        }
        // 任务太大,一分为二:
        int middle = (end + start) / 2;
        System.out.println(String.format("split %d~%d ==> %d~%d, %d~%d", start, end, start, middle, middle, end));
        SumTask subtask1 = new SumTask(this.array, start, middle);
        SumTask subtask2 = new SumTask(this.array, middle, end);
        invokeAll(subtask1, subtask2);
        Long subresult1 = subtask1.join();
        Long subresult2 = subtask2.join();
        Long result = subresult1 + subresult2;
        System.out.println("result = " + subresult1 + " + " + subresult2 + " ==> " + result);
        return result;
    }
}
```

在执行任务的 `compute()` 方法内部，先判断任务是不是足够小，如果足够小，就直接计算并返回结果，否则把自身任务一拆为二，分别计算两个子任务，再返回两个子任务的结果之和。

最后写一个 `main()` 方法测试：

```java
public static void main(String[] args) throws Exception {
    // 创建随机数组成的数组:
    int[] array = new int[400];
    fillRandom(array);
    // fork/join task:
    ForkJoinPool fjp = new ForkJoinPool(4); // 最大并发数4
    ForkJoinTask<Long> task = new SumTask(array, 0, array.length);
    long startTime = System.currentTimeMillis();
    Long result = fjp.invoke(task);
    long endTime = System.currentTimeMillis();
    System.out.println("Fork/join sum: " + result + " in " + (endTime - startTime) + " ms.");
}
```

关键代码是 `fjp.invoke(task)` 来提交一个 Fork/Join 任务并发执行，然后获得异步执行的结果。ForkJoinPool 的 `invoke` 方法会等待计算的完成。上面的例子中指定了 ForkJoinPool 的最大并发数，如果为空，并行性将自动匹配硬件可用的处理器单元数(比如，在双核处理器上该值为 2)。

### 例子：计算文档中的单词出现次数

接下来我们再介绍一个稍微复杂一点的例子：计算一个单词在文档集中的出现次数。我们需要操作一个文件目录结构并且加载每一个文件的内容到内存中，因此，我们需要下面的类来表示模型。

文档表示为一些行：

```java
class Document {
    private final List<String> lines;

    Document(List<String> lines) {
        this.lines = lines;
    }

    List<String> getLines() {
        return this.lines;
    }

    static Document fromFile(File file) throws IOException {
        List<String> lines = new LinkedList<>();
        try(BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = reader.readLine();
            while (line != null) {
                lines.add(line);
                line = reader.readLine();
            }
        }
        return new Document(lines);
    }
}
```

文件夹表示为包含的文档和子文件夹：

```java
class Folder {
    private final List<Folder> subFolders;
    private final List<Document> documents;

    Folder(List<Folder> subFolders, List<Document> documents) {
        this.subFolders = subFolders;
        this.documents = documents;
    }

    List<Folder> getSubFolders() {
        return this.subFolders;
    }

    List<Document> getDocuments() {
        return this.documents;
    }

    static Folder fromDirectory(File dir) throws IOException {
        List<Document> documents = new LinkedList<>();
        List<Folder> subFolders = new LinkedList<>();
        for (File entry : dir.listFiles()) {
            if (entry.isDirectory()) {
                subFolders.add(Folder.fromDirectory(entry));
            } else {
                documents.add(Document.fromFile(entry));
            }
        }
        return new Folder(subFolders, documents);
    }
}
```

现在我们可以开始我们的主类了：

```java
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class WordCounter {

    String[] wordsIn(String line) {
        return line.trim().split("(\\s|\\p{Punct})+");
    }

    Long occurrencesCount(Document document, String searchedWord) {
        long count = 0;
        for (String line : document.getLines()) {
            for (String word : wordsIn(line)) {
                if (searchedWord.equals(word)) {
                    count = count + 1;
                }
            }
        }
        return count;
    }
}
```

`occurrencesCount` 方法返回一个单词在文档中的出现次数，利用 `wordIn` 方法产生一行内的单词组，它会基于空格或标点符号来分割每一行。

一个文件夹下的单词出现次数就是该单词在该文件夹下的所有的子文件夹和文档中出现次数的总和。因此我们将实现两种类型的 fork/join 任务：一个任务计数单词在文档中的出现次数，另一个任务计数在词语文件夹下出现的次数，后者将 forks 子任务，然后将这些任务 join 起来，集合他们的结果。

依赖的任务关系很容易理解，如下图所示。fork/join 框架通过在等待一个任务执行文档或文件夹单词计数时可以通过 `join()` 同时执行一个文件夹任务，实现了并行最大化。

<img src="/img/article/java-multithreading/fork_and_join_2.png" style="display:block;margin:auto;"/>

让我们以 DocumentSearchTask 任务开始，它将计算一个文档中单词的出现次数：

```java
class DocumentSearchTask extends RecursiveTask<Long> {
    private final Document document;
    private final String searchedWord;

    DocumentSearchTask(Document document, String searchedWord) {
        super();
        this.document = document;
        this.searchedWord = searchedWord;
    }

    @Override
    protected Long compute() {
        return occurrencesCount(document, searchedWord);
    }
}
```

因为我们的任务需要返回值，因此它们扩展自 RecursiveTask 类，由于出现次数用长整型表示，所以用 Long 作为范型参数。`compute()` 方法是 RecursiveTask 的核心，这里的实现就简单的委派给上面的 `occurencesCount()` 方法。现在我们可以实现 FolderSearchTask，该任务将对树结构中的文件夹进行操作：

```java
class FolderSearchTask extends RecursiveTask<Long> {
    private final Folder folder;
    private final String searchedWord;
    
    FolderSearchTask(Folder folder, String searchedWord) {
        super();
        this.folder = folder;
        this.searchedWord = searchedWord;
    }
    
    @Override
    protected Long compute() {
        long count = 0L;
        List<RecursiveTask<Long>> forks = new LinkedList<>();
        for (Folder subFolder : folder.getSubFolders()) {
            FolderSearchTask task = new FolderSearchTask(subFolder, searchedWord);
            forks.add(task);
            task.fork();
        }
        for (Document document : folder.getDocuments()) {
            DocumentSearchTask task = new DocumentSearchTask(document, searchedWord);
            forks.add(task);
            task.fork();
        }
        for (RecursiveTask<Long> task : forks) {
            count = count + task.join();
        }
        return count;
    }
}
```

该任务的 `compute()` 方法的实现简单地对构造函数中传递的文件夹的每个元素 fork 出新的文档或文件夹任务，然后 join 所有的计算出的部分和并返回部分和。

对于 fork/join 框架，我们现在缺少一个方法来引导单词计数操作和一个 fork/join 池执行者:

```java
private final ForkJoinPool forkJoinPool = new ForkJoinPool();
Long countOccurrencesInParallel(Folder folder, String searchedWord) {
    return forkJoinPool.invoke(new FolderSearchTask(folder, searchedWord));
}
```

一个初始的 FolderSearchTask 引导了所有任务。上面的例子中使用了 ForkJoinPool 的空构造函数以自动匹配硬件可用的处理器单元数。

现在我们可以写 `main()` 方法，通过命令行参数来获得要操作的文件夹和搜索的单词：

```java
public static void main(String[] args) throws IOException {
    WordCounter wordCounter = new WordCounter();
    Folder folder = Folder.fromDirectory(new File(args[0]));
    System.out.println(wordCounter.countOccurrencesOnSingleThread(folder, args[1]));
}
```

## 参考

并发编程网[《Fork and Join: Java也可以轻松地编写并发程序》](http://ifeve.com/fork-and-join-java/)  
廖雪峰[《Java的Fork/Join任务，你写对了吗？》](https://www.liaoxuefeng.com/article/001493522711597674607c7f4f346628a76145477e2ff82000)

