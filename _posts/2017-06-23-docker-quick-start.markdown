---
layout: article
title: Docker 快速入门：第一个 Docker 化的 Java 应用
tags:
    - Docker
mathjax: false
---

> 本文参考慕课网[《第一个docker化的java应用》](http://www.imooc.com/learn/824)课程编写，感谢课程讲师 [刘果国](http://www.imooc.com/u/1212130/courses?sort=publish)。

### 1. 什么是 Docker

稍微有些经验的人都知道，在一台电脑上能跑起来的程序，拿到令一台电脑上未必就可以运行，因为两台电脑可能拥有完全不同的环境，而 Docker 就是为解决这种问题而生。简单来说 Docker 就是一个装应用的容器，我们可以将任何形式的程序放入其中，为这些应用创建一个轻量级的、可移植的、自给自足的容器。这样我们在自己笔记本上编译测试通过的容器就可以批量地在生产环境中部署，包括 VMs（虚拟机）、bare metal、OpenStack 集群和其他的基础应用平台。 

#### Docker 思想

![1](/img/article/docker-quick-start/docker.png)

Docker 的图标是一条运输集装箱的鲸鱼，这个图标很贴切地描述了 Docker 的核心思想：集装箱、标准化和隔离。

- **集装箱：**如果我们把程序看成是货物，那么程序的迁移过程就可以看成是货物的运输。过去我们迁移程序，需要将程序、数据以及各种依赖环境单独拷贝到新环境中并安装配置，这相当于零散地运输货物。而 Docker 将现实中集装箱运输的思想引入这一过程，将程序及其依赖的各种数据、运行环境等都装载到一个集装箱中整体进行迁移，避免了迁移后在新环境中可能产生的各种问题，大幅提高了程序迁移的效率。
- **标准化：**Docker 还对程序的迁移中的各个环节都进行了标准化，具体可以细分为运输方式、存储方式和 API 接口的标准化。Docker 存在一个超级码头，在迁移程序时，只需先将装载程序的集装箱由鲸鱼运输到超级码头，然后再由鲸鱼将这些集装箱从超级码头运输到各自的目的地，这就是运输方式的标准化。过去迁移程序时，需要记录下程序迁移到的位置（目录），以便后续对其进行改动，而 Docker 将存储方式进行了标准化，使得用户无需再关注这些细节问题，只需运行一条命令就可以了。Docker 还提供了一系列接口，对容器中程序的运行、控制、查看、删除等操作都进行了标准化。
- **隔离：**Docker 相当于为程序创建了一个轻量级的虚拟机，每个容器内的程序拥有独立的资源（CPU 和内存、磁盘等等），也可以进行独立的 IO。这就避免了因为某一个程序运行错误，例如陷入死循环，而大量占用 CPU、内存、磁盘，干扰到其他程序的正常运行。只需将程序运行在各自独立的容器中，即使一个容器中程序崩溃，其他容器中的程序依旧可以正常地运行。

#### 镜像、仓库和容器

Docker 中的装载程序的集装箱被称为镜像，每一个镜像都包含了程序及其运行所需要的各种依赖环境。运输过程中负责中转的超级码头称为仓库，仓库中存储了大量的 Docker 镜像，相当于堆积了大量集装箱的码头。而容器就是运行程序的地方，或者说运行的镜像就是容器。**通过 Docker 运行一个程序的过程就是：从仓库把镜像拉到本地，然后通过命令将镜像运行起来，变成容器。**

Docker 镜像是一个分层的文件系统，或者说就是一堆文件的集合，它包含了程序及其依赖环境的所有文件，结构如下图所示：

<img src="/img/article/docker-quick-start/docker_image.jpg" width="300px" style="display:block;margin:auto"/>

可以看到，最下层是操作系统的引导文件，上面一层 Base Image 是一个 Linux 操作系统，再上面就是与我们程序相关的文件，每一个程序都可以添加一层，存储与这个程序相关的文件，这些程序层是我们可以控制的。最上面一层是运行时的容器，这不属于 Docker 镜像，我们之后会讨论。Docker 镜像中所有的文件都是只读的，因而一个镜像是永久不会变的。

> 镜像是用来创建容器的，Docker 运行容器前需要本地存在对应的镜像，如果镜像不存在本地，Docker 会从镜像仓库下载（ 默认是 Docker Hub 公共注册服务器中的仓库）。

运行起来的镜像就是容器，可以把容器看成是一个虚拟机，只不过是采用分层的文件系统。如上图所示，容器包括下层只读的 Docker 镜像，以及最上层可以修改的镜像可写层，程度运行中所有对文件系统的修改都在这一层进行，这也是整个 Docker 容器中唯一可以修改的层。注意，在程序的运行过程中，难免要对 Docker 镜像中原有的文件做一些修改，而镜像是只读的，这怎么办呢？对于这种情况，Docker 会将这些文件拷贝到最上面的镜像可写层，然后再做修改。Docker 在访问一个文件时，会从顶层开始查找，未找到才会查找下一层。因为这些被修改的文件已经被拷贝到了最上层，所以会访问这些文件的最新版本。

> 因为容器是可以修改的，而镜像是不可以修改的，因而对于同一个镜像，我们可以生成多个不同的容器。它们独立运行，彼此没有干扰。

<img src="/img/article/docker-quick-start/container.jpg" style="display:block;margin:auto"/>

使用 Docker 的目的就是在不同的运行环境之间迁移程序，这个过程就需要使用 Docker 仓库：首先将镜像从源端传输到仓库，再从目标端将镜像拉取下来。Docker 其实就是一个存储管理镜像的服务器，Docker 官方就提供了 [Docker Hub](https://hub.docker.com/)，国内也有许多公司提供了 Docker 仓库，例如[网易蜂巢镜像中心](https://c.163.com/hub#/m/home/)。像 Ubuntu、CentOS、MySQL、tomcat 等常见的系统和软件，都有官方或者用户提供的 Docker 镜像。如果需要传输镜像是私有项目，或者需要保密，可以通过自己搭建镜像中心或者直接导入导出镜像来完成。

### 2. Docker 初体验

#### 第一个 docker 镜像

接下来我们通过运行 Docker 官方提供的 HelloWorld 程序镜像，来了解一下 Docker 的使用方式。Docker 常用的命令有：

- 拉取镜像

  ```shell
  docker pull [OPTIONS] NAME[:TAG]
  ```

  `NAME` 表示要拉取的镜像名称，这是必填项；`TAG` 表示要拉取的镜像版本（标签），如果缺省默认拉取 latest 最新版本；`OPTIONS` 表示拉取的一些参数，通常情况下不需要考虑。

- 查看本地镜像

  ```shell
  docker images [OPTIONS] [REPOSITORY[:TAG]]
  ```

  `OPTIONS` 表示可选参数，通常情况下也不需要考虑；`REPOSITORY` 和 `TAG` 分别表示镜像的名称和版本，通常只有在本地存在大量镜像的情况使用，一般也不需要考虑。

- 运行镜像

  ```
  docker run [OPTIONS] IMAGE[:TAG] [COMMAND] [ARG...]
  ```

  `IMAGE` 表示要运行的镜像名称，这是必填项；`TAG` 表示运行的镜像版本，本地可能存在一个镜像的多个版本；`COMMAND` 表示镜像运行起来之后要运行什么命令，`ARG...` 是这条命令对应的参数；`OPTIONS` 是可选的参数，后面会介绍一些常用的参数。

我们首先使用 `docker pull` 命令，将最新版的 HelloWorld 的镜像拉取到本地。拉取成功后控制台会输出摘要和状态：

```shell
$ docker pull hello-world
Using default tag: latest
latest: Pulling from library/hello-world
b04784fba78d: Pull complete 
Digest: sha256:f3b3b28a45160805bb16542c9531888519430e9e6d6ffc09d72261b0d26ff74f
Status: Downloaded newer image for hello-world:latest
```

> 这里没有提供仓库的地址，`docker pull` 命令默认从 Docker 官方提供的仓库 [Docker Hub](https://hub.docker.com/) 拉取镜像。

接下来可以通过 `docker images` 命令来查看刚刚爬取的 hello-world 镜像：

```shell
$ docker images
REPOSITORY                     TAG                 IMAGE ID            CREATED             SIZE
hello-world                    latest              1815c82652c0        8 days ago          1.84 kB
```

`REPOSITORY` 是镜像名称，`TAG` 是镜像版本，`IMAGE ID` 是镜像的 id（64 位，默认显示前 16 位），`CREATED` 表示镜像的创建时间，`SIZE` 表示镜像的大小。每一行显示一个本地镜像的信息，这里我们只拉取了一个镜像，所以只有一行。

最后，我们通过 `docker run` 命令将 hello-world 镜像运行起来，使它变成容器：

```shell
$ docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://cloud.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/engine/userguide/
```

如果你看到类似上面的信息，则代表 hello-world 镜像成功运行起来了。上面的信息还告诉我们，为了产生这条信息，Docker 做了以下的步骤：

1. Docker 客户端连接到 Docker 后台进程。
2. Docker 后台进程从 Docker Hub 拉取了 hello-world 镜像。
3. Docker 后台进程为 hello-world 镜像创建了一个新的容器，并且运行了一个可执行程度，产生我们看到的这些输出信息。
4. Docker 后台进程将输出信息变成输出流传送到 Docker 客户端，再由 Docker 客户端将信息发送到我们的终端。

下面我们通过一张图来更清晰的了解运行 Docker 镜像的整个过程：

![4](/img/article/docker-quick-start/docker_2.jpg)

左边是本地运行的 Docker 客户端，也就是我们平时执行命令的地方；中间是本地运行的 Docker 主机，由 Docker 后台进程管理；右边是 Docker 的远程仓库。在客户端运行 `docker pull` 命令后，后台进程首先会检查本地是否已存在我们指定版本的镜像，如果不存在就从远程仓库将镜像拉取到本地。同样地，在客户端运行 `docker run` 命令后，后台进程首先会检查本地是否已存在我们指定版本的镜像，如果不存在也会从远程仓库拉取，相当于执行了 `docker pull` 命令，然后将镜像运行起来变成容器。

#### 运行 nginx 镜像

接下来我们运行一个标准的 Docker 镜像，一个 nginx 服务器。不同于 hello-world 这样简单的镜像（它只是运行起来打印一条信息就结束了），nginx 是一个 web 服务器，因而它是一个需要持久运行的容器，而且 nginx 通常采用后台运行的方式。

同样地，我们先拉取 nginx 的镜像，这次我们从网易蜂巢的 Docker 仓库拉取镜像：

```shell
$ docker pull hub.c.163.com/library/nginx:latest
```

下面我们使用 `-d` 参数运行 nginx 镜像，使它在后台运行，运行后会返回容器的 id：

```shell
$ docker run -d hub.c.163.com/library/nginx
a2ca0f7186b5643abba02831a8d296427d3aedc145c47e7351310a04db335e49
```

接下来我们使用 `docker ps` 命令来查看当前运行的容器状态：

```shell
$ docker ps
CONTAINER ID        IMAGE                         COMMAND                  CREATED             STATUS              PORTS               NAMES
a2ca0f7186b5        hub.c.163.com/library/nginx   "nginx -g 'daemon ..."   2 minutes ago       Up 2 minutes        80/tcp              stupefied_sammet
```

可以看到，nginx 容器已经在后台运行了，显示的 id 也与之前返回的容器 id 一致。

容器运行起来之后，我们经常还需要进入容器去查看程序运行的情况。这时可以使用 `docker exec` 命令在容器中运行指定的命令，`docker exec` 的命令格式为：

```shell
docker exec [OPTIONS] CONTAINER COMMAND [ARG...]
```

`CONTAINER` 是容器的名称或 id，`COMMAND` 是要在容器中运行的命令。`OPTIONS` 是可选的参数，我们常用的参数有：`-i` 即使在未连接成功情况下保持输入有效、`-t` 给我们分配一个伪终端。

下面我们连接之前运行在后台的 nginx 容器：

```shell
$ docker exec -it a2 bash
root@a2ca0f7186b5:/#
```

因为此时只有一个容器在运行，所以我们只需输入很短的容器 id。可以看到，运行 bash 命令后，我们终端的命令行提示符发生了变化，仿佛是连接到了一个远程服务器一样。因为 nginx 容器环境为 Linux 系统，所以我们可以在终端内执行 `ls`、`pwd` 等常用的 Linux 命令：

```shell
root@a2ca0f7186b5:/# pwd  
/
root@a2ca0f7186b5:/# ls
bin  boot  dev	etc  home  lib	lib32  lib64  libx32  media  mnt  opt  proc  root  run	sbin  srv  sys	tmp  usr  var
root@a2ca0f7186b5:/# which nginx
/usr/sbin/nginx
root@a2ca0f7186b5:/# exit
exit
```

至此我们已经成功地运行了 nginx 容器，但是还无法通过浏览器访问。Docker 容器的网络分为三种：桥接(Bridge)、宿主机共享(Host)和无网络(None)：在 Bridge 模式下，Docker 会产生一个名为 docker0 的网桥与宿主机的网卡相连，然后每个容器再虚拟出自己的网卡与 docker0 相连，每个容器拥有自己的 IP 和端口；而在 Host 模式下，Docker 容器不会虚拟出自己的网卡，而是直接使用宿主机上的 IP 和端口，与宿主机共享网络环境。在 Bridge 模式下，Docker 可以在容器的端口与宿主机的端口之间建立映射，这样访问宿主机的端口就可以访问到容器的端口。

> Docker 容器默认使用 Bridge 的网络连接方式。

接下来，我们通过指定端口映射来重新启动 nginx 镜像：

```bash
$ docker stop a2    # 停止容器运行
$ docker run -d -p 8080:80 hub.c.163.com/library/nginx
e9981cdddcf799756209ec142de0386b25d1bca632fd004c86018f91ff6f9c20
$ docker ps    # 检查当前运行的容器
CONTAINER ID        IMAGE                         COMMAND                  CREATED             STATUS              PORTS                  NAMES
e9981cdddcf7        hub.c.163.com/library/nginx   "nginx -g 'daemon ..."   21 hours ago        Up 2 minutes        0.0.0.0:8080->80/tcp   sharp_dijkstra
$ netstat -na | grep 8080    # 检查主机8080端口是否处于监听状态
tcp6       0      0  ::1.8080               *.*                    LISTEN     
tcp4       0      0  *.8080                 *.*                    LISTEN
```

上面的命令将主机的 8080 端口与容器的 80 端口之间建立映射，下面我们就可以通过访问主机的 8080 端口来访问启动的 nginx 服务器了：

<img src="/img/article/docker-quick-start/nginx.png" style="display:block;margin:auto"/>

### 3. 制作自己的 Docker 镜像

下面我们以一个简单的 java web 网站 [Jpress](http://jpress.io/) 为例，了解一下 Docker 镜像的制作过程。

#### 制作镜像

Docker 通过 Dockerile 来描述镜像制作过程中的每一个步骤，Dockerfile 编写完成后再通过 `docker build` 命令来读取 Dockerfile 构建镜像。

我们先从 Github 上下载 Jpress 的最新 war 包：[https://github.com/JpressProjects/jpress/tree/master/wars](https://github.com/JpressProjects/jpress/tree/master/wars)。因为 Jpress 运行在 tomcat 之上，所以我们可以以 tomcat 的 Docker 镜像为基础，构建 Jpress 镜像。先从仓库下载 tomcat 镜像：

```shell
$ docker pull hub.c.163.com/library/tomcat:latest
```

因为 tomcat 的运行需要 Java 环境的支持，所以 tomcat 镜像中已经包含了 JDK，我们不必再去重复添加。下面我们切换到之前下载的 jpress war 包所在的目录，开始编写 Dockerfile：

```
from hub.c.163.com/library/tomcat

MAINTAINER xiaosheng jsksxs360@163.com

COPY jpress-web-newest.war /usr/local/tomcat/webapps
```

第一行我们通过 `from` 语句，指定了在 tomcat 镜像的基础上构建镜像。第二行我们添加了镜像制作者的名称和邮箱，这一行是可选的。第三行，我们通过 `COPY` 命令，将 Jpress 的 war 包拷贝到了 tomcat `CATALINA_HOME` 环境下的 webapps 文件夹，这样 Jpress 就能跟随 tomcat 启动了。

下面，我们通过 `docker build` 命令来构建镜像：

```shell
$ docker build -t jpress:latest .
Sending build context to Docker daemon 494.2 MB
Step 1/3 : FROM hub.c.163.com/library/tomcat
 ---> 3695a0fe8320
Step 2/3 : MAINTAINER xiaosheng jsksxs360@163.com
 ---> Running in 32f97f1c590c
 ---> a85aee7bd53c
Removing intermediate container 32f97f1c590c
Step 3/3 : COPY jpress-web-newest.war /usr/local/tomcat/webapps
 ---> 5383a0a3b9db
Removing intermediate container 316dec7cb915
Successfully built 5383a0a3b9db
$ docker images
REPOSITORY                     TAG                 IMAGE ID            CREATED             SIZE
jpress                         latest              5383a0a3b9db        21 hours ago        355 MB
hello-world                    latest              1815c82652c0        9 days ago          1.84 kB
hub.c.163.com/library/tomcat   latest              3695a0fe8320        4 weeks ago         334 MB
hub.c.163.com/library/nginx    latest              46102226f2fd        8 weeks ago         109 MB
```

通过 `-t` 选项，可以指定构建镜像的名称和版本，如果省略则都为 None。通过 `docker images` 可以看到，我们成功构建了 jpress 的镜像。

#### 运行 Jpress 镜像

因为 tomcat 也是一个网络服务器，所以与 nginx 类似，我们也需要指定端口映射：

```shell
$ docker run -d -p 8888:8080 jpress
d29a94798fb4ba24e1a4013dcd89e865b3cf78ace9b115087eee4bdf44f92551
$ netstat -na | grep 8888
tcp6       0      0  ::1.8888               *.*                    LISTEN     
tcp4       0      0  *.8888                 *.*                    LISTEN
```

tomcat 默认端口为 8080，我们这里将它映射到宿主机的 8888 端口。可以看到，本机的 8888 端口已经处于监听状态，可以使用浏览器访问：

![6](/img/article/docker-quick-start/tomcat.jpg)

下面我们通过 `http://127.0.0.1:8888/press-web-newest` 尝试访问 Jpress，会自动跳转到 Jpress 的安装界面：

![7](/img/article/docker-quick-start/jpress.jpg)

![8](/img/article/docker-quick-start/jpress_2.jpg)

可以看到，Jpress 镜像成功地运行起来了，但是 Jpress 需要 MySQL 数据库，而 MySQL 数据库的安装配置还是很麻烦的。于是，我们可以再次运用 Docker 快速地在本地运行起 MySQL 数据库：

```shell
$ docker pull hub.c.163.com/library/mysql:latest
$ docker run -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -e MYSQL_DATABASE=jpress hub.c.163.com/library/mysql
```

`docker run` 命令中，我们通过 `-e` 选项指定了 MySQL 的环境变量，以设置 root 用户的密码和创建的数据库名称。将 MySQL 镜像运行起来之后，我们就可以填写上面网页中的内容了：

![9](/img/article/docker-quick-start/jpress_3.jpg)

注意，数据库主机地址需要填写当前宿主机的实际 ip 地址，可以通过 `ifconfig` 命令查询得到。点击下一步，我们再设置 Jpress 网站的管理员和密码：

![10](/img/article/docker-quick-start/jpress_4.jpg)

这时 Jpress 提示我们重启 web 容器，我们可以使用 `docker restart` 命令重启指定的容器：

```shell
$ docker ps
CONTAINER ID        IMAGE                         COMMAND                  CREATED             STATUS              PORTS                    NAMES
564ae0e5811b        hub.c.163.com/library/mysql   "docker-entrypoint..."   21 hours ago        Up 18 minutes       0.0.0.0:3306->3306/tcp   keen_easley
d29a94798fb4        jpress                        "catalina.sh run"        22 hours ago        Up 44 minutes       0.0.0.0:8888->8080/tcp   distracted_goldstine
$ docker restart d29a94798fb4
```

再访问 `http://127.0.0.1:8888/jpress-web-newest/` 就可以正常访问我们的 Jpress 网站了。

> 本文参考慕课网[《第一个docker化的java应用》](http://www.imooc.com/learn/824)课程编写，感谢课程讲师 [刘果国](http://www.imooc.com/u/1212130/courses?sort=publish)。