---
layout: article
title: Windows 10 系统上的 Python 开发环境配置
tags:
    - Python
    - Linux
mathjax: false
---

## 前言

在 Linux 环境上进行 Python 开发是很多用户的选择，但在很多情况下我们还是离不开 Windows 系统。因此过去只能通过安装双系统或者虚拟机的方式来完成，但无论哪一种方式都不是很方便。

幸运的是从 Win10 开始，系统已经自带了一个 Linux 子系统 (WSL)，并且可以与 Windows 环境无缝连接，这大大简化了在 Windows 上搭建 Linux 开发环境的难度。

## 准备工作

首先说明一下，因为 Windows 子系统 (WSL) 不支持 GUI 桌面或应用程序（如 PyGame、Gnome、KDE 等），因此本文仅适用于不需要图形化界面的 Python 开发。如果你需要开发包含图形化界面的 Python 程序，请直接在 Linux 系统上进行，或者在 Windows 上安装 Python（不推荐）。

在开始前，我们首先在 Win10 上启用 Linux 子系统（需要新版本的 Win10）。

> 通过 Windows 子系统，我们可以运行 GNU/Linux 环境（包括大多数命令行工具、实用工具和应用程序），直接在 Windows 上进行修改，并与 Windows 文件系统和常用工具（如 Visual Studio Code）完全集成。

首先在“开始”菜单边上的**搜索栏**中键入“启用或关闭 windows 功能”，下拉并勾选“适用于 Linux 的 Windows 子系统”，生效需要重启系统。

<img src="/img/article/configure-python-environment/windows_function.png" style="display: block; margin: auto;"/>

在 Windows 子系统上有多个 Linux 分发版可以运行，本文使用的是流行的 Ubuntu 系统，建议直接安装目前最新的是 [Ubuntu 18.04 LTS](https://www.microsoft.com/store/productId/9N9TNGVNDL3Q)。

下载完成后，直接在**搜索栏**中键入“bash”就能启动 Ubuntu 系统了。首次运行时需要创建帐户名称和密码，按照提示输入就行。以后默认情况下，我们都会以该用户的身份自动登录。

![pic2](/img/article/configure-python-environment/ubuntu_in_windows.png)

可以看到，默认情况下启动时的位置位于 `/mnt/c/Windows/System32`，也就是 Windows 下的 C 盘 Windows\System32 目录。所有 Windows 下的磁盘（C 盘、D 盘）都挂载在 Ubuntu 中的 `/mnt/` 位置，因此可以直接通过 bash 命令行切换磁盘位置，例如：`cd /mnt/d` 切换到 D 盘。

和标准的 Ubuntu 系统一样，我们可以通过 `lsb_release -d` 命令来检查当前使用的 Linux 版本，也可以通过 `sudo apt update && sudo apt upgrade` 命令来更新系统中的包。注意，Windows 系统不会自动更新子系统中的软件。

> 如果你在安装过程中有任何问题，可以参阅官方编写的[《适用于 windows 10 的适用于 Linux 的 Windows 子系统安装指南》](https://docs.microsoft.com/windows/wsl/install-win10)。

## 设置 Visual Studio Code

[VS Code](https://code.visualstudio.com/) 是微软提供的一款非常流行的代码编辑器，可以看作是超级 IDE Visual Studio 的轻量化版本。通过安装对应的插件，它足以满足绝大多数的开发需求。更重要的是 VS Code 与适用于 Linux 的 Windows 子系统完美集成，提供[内置终端](https://code.visualstudio.com/docs/editor/integrated-terminal)在代码编辑器和命令行之间建立无缝的工作流。

VS Code 也适用于 Linux，但适用于 Linux 的 Windows 子系统不支持 GUI 应用，因此我们需要在 Windows 上[下载](https://code.visualstudio.com/)并安装它。后面我们将通过远程 -WSL 扩展与 Linux 命令行和工具进行集成。

![pic3](/img/article/configure-python-environment/wsl.png)

如上图所示，我们首先通过 **Ctrl + Shift + X** 或菜单导航“**View > Extensions**”打开插件安装窗口，然后搜索“wsl”，安装 [Remote - WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) 插件。 这使我们可以将 WSL 用作集成开发环境，并自动处理兼容性和路径。

> 如果已安装 VS Code，请将版本升级到 1.35 及以上，以便安装远程 WSL 扩展。 

虽然 Ubuntu 附带了一个 Python，但是我们实际编程时还需要用到许多其他的 Python 模块。

- 一种解决方法是安装标准包管理器 **pip** 和创建管理轻型虚拟环境的标准模块 **venv**。
  - 通过 `sudo apt install python3-pip` 在 Ubuntu 上安装 pip，以后就可以通过 pip 来安装和管理不属于 Python 标准库的其他包。
  - 输入 `sudo apt install python3-venv` 安装 venv。
- 更推荐的方法是直接安装 **Anaconda**，它自带 Python 环境，并且封装了大量实用的科学计算包。
  - 首先通过 `wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh` 从清华大学镜像源下载 Anaconda，你也可以访问[这里](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)，修改成最新的版本号。
  - 下载完成后，输入 `./Anaconda3-5.2.0-Linux-x86_64.sh` 进行安装。安装时会提供一些配置选项，如果你不是很清楚，一路 Enter 就可以了。安装好后，通过 `source ~/.bashrc` 使得新配置立即生效。

## 创建新项目

在一切准备就绪后，我们进入正题——创建第一个 Python 项目。

首先在 Ubuntu 系统上创建一个新的项目目录：打开 bash，切换到要创建目录的位置，然后通过 `mkdir HelloWorld` 创建目录。

![pic4](/img/article/configure-python-environment/mkdir.png)

对于 Python 开发项目，我们一般使用虚拟环境将项目隔离开来，避免项目之间的 Python 包存在版本冲突。 例如，你可能需要维护一个需要 Django 1.2 的旧 web 项目，但新项目中又需要使用 Django 2.2，如果在虚拟环境外全局更新 Django 就可能会遇到一些版本控制问题。 

> 除了防止版本冲突以外，虚拟环境还允许我们在没有管理权限的情况下安装和管理包。

打开 bash 终端，在 HelloWorld 项目文件夹中使用 `python3 -m venv .venv` 命令创建名为 **.venv** 的虚拟环境。然后输入`source .venv/bin/activate` 激活虚拟环境，如果有效就会在命令提示符之前看到 **(. venv)** 。 

![pic5](/img/article/configure-python-environment/venv.png)

> Python 3.6 及以上的版本通过 venv 模块原生支持虚拟环境，可以代替之前的 virtualenv。

现在, 我们有了一个可供编写代码和安装包的独立环境。如果想要停用虚拟环境，直接输入 `deactivate` 就可以了。

建议大家在每个项目目录中创建自己单独的虚拟环境，这样虚拟环境命名上就不需要担心冲突。 遵循 Python 约定，我们通常将虚拟环境命名为 **.venv** 以创建一个隐藏的环境目录。

> 一般我们会将 **.venv** 添加到 .gitignore 文件中以防止被 git 跟踪。可以参考 GitHub 官方提供的适用于 Python 的默认 [.gitignore 模板](https://github.com/github/gitignore/blob/50e42aa1064d004a5c99eaa72a2d8054a0d8de55/Python.gitignore#L99-L106)。

## 打开 WSL-远程窗口

VS Code 通过我们前面安装的远程 WSL 插件将 Linux 子系统视为远程服务器，使得我们可以使用 WSL 作为集成开发环境。

在 bash 终端输入 `code .` 调用 VS Code 打开当前所在的项目文件夹。打开 VS Code 后，我们可以看到左下角的远程连接主机指示器显示“**WSL**”，这表明我们当前是在 WSL 上进行编辑。

![pic6](/img/article/configure-python-environment/wsl_2.png)

这时候可以关闭 Ubuntu 终端了，我们所有的后续操作都可以直接使用集成到 VS Code 中的 WSL 终端来进行。WSL 终端通过按 **Ctrl + '** 或选择“**View > Terminal**”来打开，命令行会直接定位到我们在 Ubuntu 终端中创建的项目文件夹。

![pic7](/img/article/configure-python-environment/vs_code.png)

为了让 VS Code 能够支持 Python 编码，我们还需要为 VS Code 安装 Python 插件。同样地，我们通过 **Ctrl + Shift + X** 或菜单导航“**视图 > 扩展**”打开 VS Code 插件窗口，搜索”**Python**“。找到微软官方提供的 Python 插件，点击绿色的 Install 按钮安装，安装后再点击 Reload 按钮重载 VS Code。

## 运行简单的 Python 程序

接下来我们创建并运行一个简单的 Python 程序进行测试。

首先通过 **Ctrl + Shift + E **或菜单导航“**View > Explorer**”打开文件资源管理器窗口。然后按 **Ctrl + Shift + '** 打开集成的 WSL 终端，并确保已位于 HelloWorld 项目文件夹，输入 `touch test.py` 创建 python 文件。这时应该可以在左侧的资源管理器窗口中看到我们创建的 test.py。

![pic8](/img/article/configure-python-environment/vs_code_2.png)

选择并打开 **test.py** 文件，Python 扩展会自动选择并加载一个 Python 解释器，显示在 VS Code 窗口的底部。

![pic9](/img/article/configure-python-environment/wsl_3.png)

将下面的代码粘贴到 test.py 文件中，然后保存该文件（Ctrl + S）：

```python
print("Hello World")
```

在资源管理器窗口中选择 **test.py** 文件，右键选择”**Run Python File in Terminal**“，或者直接在集成的 WSL 终端窗口中输入`python3 test.py` 运行 "Hello World" 程序。

如果终端窗口中成功打印出”Hello World“，那么恭喜你，Python 环境已经完全配置好了！

## 参考

[[1]](https://docs.microsoft.com/zh-cn/windows/python/get-started/python-for-web) 开始在 Windows 上使用 Python 进行 web 开发 

