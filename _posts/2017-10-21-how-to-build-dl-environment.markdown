---
layout: article
title: "Ubuntu深度学习环境搭建：安装Cuda、CuDNN、TensorFlow"
tags:
    - 机器学习
mathjax: true
---

## 前言

首先，确保你的机器上已经[下载](https://www.ubuntu.com/download/server)并安装好 Ubuntu 服务器系统 (Ubuntu Server) 了，如果是使用阿里云或者其他云服务主机，那么直接创建一个安装有 Ubuntu 的实例就可以了。

本文以目前最新的 Ubuntu Server 16.04.3 LTS 系统为例，介绍包括 CUDA、CudNN 软件以及 TensorFlow、Keras 等 Python 库的安装方法。

下面是我们需要安装的软件列表。

1. Cuda
2. CudNN
3. TensorFlow

### 确认 GPU 可用

首先我们需要检查 GPU 是否已正常安装并已处在运行状态。

```shell
lspci -nnk | grep -i nvidia
```

### 前期准备

在安装软件之前，首先确认 **apt-get** 已经更新到最新状态。

```shell
sudo apt-get update
```

因为深度学习需要大量科学计算相关的 python 库，因为我们直接安装 Python 科学计算发行版 **Anaconda**。这里我们使用对应 Python 3.6 版本的 Anaconda 5.0.0 (使用[清华大学镜像](https://mirrors.tuna.tsinghua.edu.cn/))。

```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.0-Linux-x86_64.sh
sudo chmod +x Anaconda3-5.0.0-Linux-x86_64.sh
./Anaconda3-5.0.0-Linux-x86_64.sh
```

注意，如果想用 Anaconda 代替系统自带的 Python，需要在 Anaconda 安装出现 `Anaconda3 will now be installed into this location:` 时，将安装位置修改为 `/usr/bin/anaconda3`。安装好后，将 Anaconda 的目录添加到 PATH 变量，在 `/etc/profile` 文件中添加：：

```shell
export PATH=/usr/bin/anaconda3/bin:$PATH
```

然后， `source /etc/profile` 使配置立刻生效。输入 `python3` 进入 Python 运行环境，如果见到 `Python 3.6.2 |Anaconda, Inc.|...` 的提示符，证明 Anaconda 安装成功。

## 安装 Cuda

### 下载并安装 Cuda

> 新版的 Cuda 已经包含了对应的 Nvidia 驱动，因而我们直接安装 Cuda 就可以了。这里我们示例安装最新的 Cuda 9.0。

首先打开 Cuda 下载界面：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)，我们依次选择 Linux → x86_64 → Ubuntu → 16.04 → deb [network]。

![1](/img/article/how-to-build-dl-environment/cuda.jpg)

下方会出现对应的下载链接和安装提示，按照提示一步一步安装即可：

![2](/img/article/how-to-build-dl-environment/cuda_2.jpg)

```shell
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

### 配置与测试

安装好之后，我们还需要将 Cuda 的路径添加至环境变量，在 `/etc/profile` 文件中添加：

```shell
export CUDA_HOME=/usr/local/cuda-9.0
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

> 上面示例安装的是 Cuda 9.0，所以目录为 cuda-9.0，需要根据实际情况修改。

之后 `source /etc/profile` 使配置立刻生效。

为了确认 Cuda 和驱动已得到正确安装，我们首先可以运行 `nvcc -V` 查看 nvcc 编译器的信息，如果看到类似下面所示的信息，则 Cuda 安装成功。

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
```

然后可以运行 `nivida-smi` 来查看 GPU 是否已被正确识别（如果你希望查看 GPU 的性能参数，这个命令也同样管用）。如果看到类似下图所示的信息，则 Nvidia 驱动已安装成功。

![3](/img/article/how-to-build-dl-environment/nivida_smi.jpg)

## 安装 cuDNN

cuDNN 是一个能够用来加速深度学习框架（比如 TensorFlow）的库，能够为深度神经网络提供原生的 GPU 加速。

要下载 CuDNN 需要[注册](https://developer.nvidia.com/developer-program)**英伟达开发者项目** (NVIDIA Developer Program)，通过邮件验证后，就可以从 [cuDNN 页面](https://developer.nvidia.com/cudnn) 登录并[下载 cuDNN](https://developer.nvidia.com/rdp/cudnn-download)。

![4](/img/article/how-to-build-dl-environment/cudnn.jpg)

我们之前已经安装了 CUDA 9.0 所以选择下载第一个 cuDNN v7.0.3 for CUDA 9.0 下的 **cuDNN v7.0 Library for Linux**。然后将下载得到的 zip 包通过 scp 传入服务器中。解压出的 cuda 文件夹里的 lib 目录和 cudnn.h 文件复制到 CUDA 目录覆盖对应的文件夹，在终端中输入：

```shell
cp cuda/lib64/* /usr/local/cuda/lib64/
cp cuda/include/cudnn.h /usr/local/cuda/include/
cd /usr/local/cuda/lib64
sudo ln -sf libcudnn.so.7.0.3 libcudnn.so.7
sudo ln -sf libcudnn.so.7 libcudnn.so
sudo ldconfig -v
```

注意，由于服务器权限问题，原来的链接会失效，所以需要重建链接。上面 `libcudnn.so.xxx` 版本号需要根据实际情况修改。

## 安装 TensorFlow

考虑到 pip 等提供的 Tensorflow 因为不同机型的兼容性问题，所以往往在配置上做了许多的限定，因而无法充分调用起机器的硬件（例如 CPU 的加速组件），而且还可能与安装的 Cuda 的版本不兼容。

所以最简单的方法就是原生编译安装 Tensorflow，具体的过程可以参考我的另一篇博客[《编译安装 TensorFlow》](/2017/09/19/article96/)，注意要按照其中的 GPU 版安装方法安装，否则 Tensorflow 只能使用 CPU 计算。

## 参考

Alexander Crosson[《Installing Nvidia, Cuda, CuDNN, TensorFlow and Keras》](https://medium.com/@acrosson/installing-nvidia-cuda-cudnn-tensorflow-and-keras-69bbf33dce8a)  
SCP-173[《Keras安装和配置指南(Linux)》](http://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/)