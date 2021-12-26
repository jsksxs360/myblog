---
layout: article
title: "使用 Keras 搭建模型识别验证码：通过 Web API 提供识别服务"
tags:
    - Keras
    - 机器学习
    - Flask
mathjax: false
---

最近因为项目上的需求，需要开发一个识别 4 位数字字母验证码的接口。想到自己虽然在科研上使用 Keras 已经有很长时间，但从来没有真正地将这些模型运用起来，或者说以 Web API 的形式对外提供过服务。因此借着这次机会，我正好完整地进行了一次从训练模型到最终包装成网络服务的开发。

## 准备工作

### 获取标注数据

无论什么类型的任务，只要希望通过机器学习方法来解决，那么必不可少的就是数据，可以说数据是烹饪“机器学习模型”这道菜必不可少的原料。我们项目的实际需求是识别一个网站上查询数据时需要输入的验证码，因此任务本身非常地简单，只是爬虫的一个中间环节。

虽然每一次刷新网页都可以获得新的验证码，但这些获得的图片都是没有标注的数据，因此无法直接用于我们的监督学习任务，而如果人工标注，则需要付出大量的人力和时间成本。考虑到目前各大云平台基本都有验证码识别服务，而且我们可以通过网页的返回值来判断输入验证码的正确性，因此我们可以利用这些平台的识别接口来产生大量的数据样本。

最终我们将识别正确的验证码图片导出，并且将图片直接命名为识别结果（免去单独记录每张图片对应的验证码信息）。于是我们的标注数据就准备完成了，共包含 41 万张图片，如下图所示。如果需要可以点击[这里](https://pan.baidu.com/s/1N2d_JVfwPictjeV-2C9HbQ)下载（提取码: x6p8）：

<img src="/img/article/use-keras-to-identify-captcha/captcha.png" style="display: block; margin: auto;">

### 模型结构

接下来，就是考虑应该如何搭建模型来完成任务。因为验证码识别是一个图像任务，因此我们很直接地就想到使用卷积神经网络 (CNN) 来完成。考虑到如果将 4 位验证码拆开识别，那么每一位都是一个相同的 36 分类任务（26 个字母+10 个数字），因此我们在模型的最后使用 4 个独立的 36 类分类器来分别识别每一位，并且它们共享卷积核捕获到的图像特征。

## 模型实现

### 卷积模型的实现

模型结构想好后，实现就变得非常容易了。我们参考图片分类任务中经典的 VGG 模型，采用 3 个块 (Block) 共计 6 个卷积层来捕获图像特征，其中每个块中包含两个卷积层和一个池化层：

```python
pic_in = Input(shape=(32, 90, 1))
# Block 1
cnn_features = Conv2D(64, (3,3), activation='relu', padding='same')(pic_in)
cnn_features = Conv2D(64, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
# Block 2
cnn_features = Conv2D(128, (3,3), activation='relu',padding='same')(cnn_features)
cnn_features = Conv2D(128, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
# Block 3
cnn_features = Conv2D(256, (3,3), activation='relu',padding='same')(cnn_features)
cnn_features = Conv2D(256, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
cnn_features = Flatten()(cnn_features)
# classifier 1
output_l1 = Dense(256, activation='relu')(cnn_features)
output_l1 = Dropout(0.5)(output_l1)
output_l1 = Dense(36, activation='softmax')(output_l1)
# classifier 2
output_l2 = Dense(256, activation='relu')(cnn_features)
output_l2 = Dropout(0.5)(output_l2)
output_l2 = Dense(36, activation='softmax')(output_l2)
# classifier 3
output_l3 = Dense(256, activation='relu')(cnn_features)
output_l3 = Dropout(0.5)(output_l3)
output_l3 = Dense(36, activation='softmax')(output_l3)
# classifier 4
output_l4 = Dense(256, activation='relu')(cnn_features)
output_l4 = Dropout(0.5)(output_l4)
output_l4 = Dense(36, activation='softmax')(output_l4)

model = Model(inputs=pic_in, outputs=[output_l1, output_l2, output_l3, output_l4])
```

模型的输入就是验证码图像，考虑到这些图像都是黑白图片，因此我们只需要简单地以**灰度图**的方式来读取就可以了。这样最后每张图片都会转换为一个二维矩阵，并且每个位置上是一个 0~255 之间的灰度值（实际操作中会将值归一化到 0~1 之间）。因此最终我们模型的输入 `Input(shape=(32, 90, 1))` 只有一个通道（如果以 RGB 形式读取会有 3 个通道），其中 32 和 90 分别是图片的高度和宽度。

这里稍微介绍一下我们读取图片时使用的 `skimage.io.imread()` 函数。目前主流的读取图片的 Python 库有 `PIL`、`scipy`、`matplotlib`、`opencv`、`skimage` 等等，除了 `PIL.Image.ope()` 不直接返回 numpy 对象，其他绝大部分方法都会返回一个 numpy.ndarray 格式的数据，但是通道顺序可能有所不同。

本文首先使用 `skimage.io.imread()` 方法读取验证码 png 文件，它会读取图像并且返回 RBG 格式的 numpy 张量。因为验证码都是黑白图片，所以我们进一步通过 `skimage.color.rgb2gray()` 函数将 RGB 格式的图片转换为灰度图：

```python
im = io.imread(os.path.join(data_dir, file_name))
im3 = color.rgb2gray(im)
pic_data.append(np.expand_dims(im3, axis=2))
```

特别地，由于 Keras 需要输入的图像数据包含通道维，而 `skimage.color.rgb2gray` 函数转换后的灰度图是不包含通道维的，因此我们通过 `np.expand_dims()` 在最后添加一维。

### 训练模型

由于识别验证码模型具有多个输出（4 个），因此我们需要在编译模型时为每一个输出指定对应的损失函数。考虑到模型的四个分类器具有完全相同的结构，并且它们的输出都是一个长度为 36 的一维向量（每一个位置是一个 0~1 之间的值，对应字符的概率），因此我们简单地给每一个输出指定一个交叉熵损失就可以了：

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              loss_weights=[1., 1., 1., 1.], # 给四个分类器同样的权重
              metrics=['accuracy'])
```

这里因为四个输出都采用交叉熵损失，因此使用了简化写法，你也可以写成 `loss=[loss1, loss2, loss3, loss4]` 这样的列表。优化器这里就选择通用的 Adam，并且在计算总损失函数时给予四个分类器相同的权重。

> 事实上中间的两个字符相较边上的字符更难识别，因此可以稍微提高权重。

### 解码函数

模型的输出是一个 4 维的张量，其中每一维对应一个分类器的输出，而我们最终希望得到的是一个长度为 4 的字符串（例如“8a3t”这样的）。因此，我们还需要编写一个解码函数，以便将模型的输出转换为最终的结果字符串：

```python
def decode(y):
    y = np.argmax(np.array(y), axis=-1)
    return ''.join([id2label[x] for x in y])
```

正如前面所说，模型的输出尺寸为 `(4, 36)`，每一维分别对应一个分类器的输出。因此，我们首先通过 `np.argmax()` 函数将输出转换为 `(4)` 以便得到每一个分类器输出对应的字符编号，然后通过预处理中创建的 id2label 字典来获得编号对应的字符。

完整的模型训练代码可以参见  
[https://github.com/jsksxs360/captcha_identification/blob/master/src/train_model.py](https://github.com/jsksxs360/captcha_identification/blob/master/src/train_model.py)

## 将模型包装成 Web API

训练之后，我们通过简单的选择（挑出在验证集上准确率最高的模型），就得到了最终的模型。然后就是本文的最终环节——将模型包装成一个 Web 接口，以便其他用户通过网络来调用。

首先我们需要选择网络框架，Python 下有很多 Web 框架可以使用，本文选择的是非常流行的微服务框架 [Flask](http://flask.pocoo.org/)，它易于使用并且非常灵活，足够满足我们的需求。考虑到绝大部分的网络接口在处理图片时，都会将图片转换为 Base64 编码（产生一个非常长非常复杂的字符串），因此如果直接以 URL 参数的形式来传输是不合适的，我们需要将它装进数据包，以 POST 的形式发送给服务器。

下面我们实现一个地址为 `/captcha` 的 POST 数据接口，它首先将用户发来的 Base64 编码还原为图片，然后调用模型来识别验证码，并且将结果返回给用户（需要考虑异常处理）：

```python
app = Flask(__name__)

model = None
def loading():
    global model
    model = load_model('../model/model-45.hdf5')
    
def decode(y):
    y = np.argmax(np.array(y), axis=-1)
    return ''.join([id2label[x] for x in y])

@app.route('/captcha', methods = ['POST'])
def api_message():
    base64_str = ''
    if request.headers['Content-Type'] == 'text/plain':
        base64_str = request.data
    elif request.headers['Content-Type'] == 'application/json':
        base64_str = request.json.get('pic', '')
    else:
        return json.dumps({"status":"NO", "msg":"request head not support"})
    if base64_str:
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", base64_str, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            data = result.groupdict().get("data")
        else:
            data = src
        try:
            imgdata = base64.b64decode(data)
            tempfilename = str(uuid.uuid1()) + '.png'
            file = open(tempfilename,'wb')
            file.write(imgdata)
            file.close()
            im = io.imread(tempfilename)
            im3 = color.rgb2gray(im)
            pic = np.asarray([np.expand_dims(im3, axis=2)])
            result = np.squeeze(model.predict(pic))
            os.remove(tempfilename)
            return json.dumps({"status":"OK", "result": decode(result)})
        except:
            return json.dumps({"status":"NO", "msg":"error"})
    else:
        return json.dumps({"status":"NO", "msg":"input base64 is empty"})

if __name__ == '__main__':
    loading() # 加载模型
    http_server = WSGIServer(('127.0.0.1', 8090), app)
    http_server.serve_forever()
```

这里我们根据用户 POST 请求中的 Header 参数来决定读取 Base64 编码的方式。默认情况下我们采用 `Content-Type:application/json` 的方式，以 Json 格式传输数据。考虑到多用户可能并发调用，因此我们采用 UUID 来命名生成的临时图片文件。

额外说一下，在大多数 flask 的示例代码中，都是以下面的方式启动 Web 服务：

```python
from flask import Flask, url_for
app = Flask(__name__)
...

if __name__ == '__main__':
    app.run()
```

这样启动的其实是供测试使用的 Debug 模式，这时所有服务器的状态都将直接输出到屏幕上。但是测试模式有一个问题，此时 Flask 是单线程运行的，换句话说它不支持并发，当一个用户的请求没处理完之前，后面所有的用户都需要等待。在实际部署时，这肯定是不合适的。因此，我们需要使用 gevent 来非阻塞地运行服务器程序：

```python
from gevent import monkey
monkey.patch_all()  # 打上猴子补丁

from flask import flask
...

if __name__ == '__main__':
    from gevent import pywsgi
    app.debug = True
    server = pywsgi.WSGIServer( ('127.0.0.1', 5000 ), app)
    server.serve_forever()
```

这里在引入 gevent 前，我们还在程序最开始执行的位置引入了猴子补丁 `gevent.monkey`，这能修改 python 默认的 IO 行为，让标准库变成[协作式 (cooperative) 的 API](http://www.gevent.org/api/gevent.monkey.html)。

完整代码可以参见  
[https://github.com/jsksxs360/captcha_identification/blob/master/src/start_server.py](https://github.com/jsksxs360/captcha_identification/blob/master/src/start_server.py)

服务器在本地运行起来之后，可以通过 `curl` 命令进行测试：

```python
curl -H "Content-type: application/json" \
-X POST http://127.0.0.1:5000/captcha -d '{"pic":"base64XXXXX"}'
```

## 参考

[使用 Python & Flask 实现 RESTful Web API](https://www.cnblogs.com/luxiaoxun/p/7543143.html)  
[在 Flask 应用中使用 gevent](https://www.cnblogs.com/brifuture/p/10050946.html)  
[Keras 快速上手指南（上）：Keras 入门](https://xiaosheng.me/2017/07/04/article77/)