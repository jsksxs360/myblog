---
layout: article
title: "iOS 证书简介：快速了解 iOS 证书体系"
tags:
    - iOS
mathjax: false
---

在 iOS 开发的过程中，开发者通常将注意力都放在如何编写代码上，而对于 iOS 应用的证书签名部分却只是一知半解。到了真机调试、产品发布的过程，才胡乱折腾一通，最终解决问题的时候其实对证书的构成与机理还是一无所知，不知所以然。

本文首先从 iOS 证书体系入手，一步步解释各模块的内容与注意项，然后通过实例演示证书的实际申请过程。

## iOS 证书体系

iOS 的证书体系由以下四个基本构成模块组成：

- 证书 (Certificates)
- 标识 (Identifiers)
- 设备 (Device)
- 描述文件 (Provisioning Profile)

### 证书 (Certificates)

代码签名可以让系统确保你的应用来源，并确保你的应用不被修改（执行代码修改后，原签名将失效）。

首先你要有一个证书，通过 Keychain 的证书助理生成 Certificates Signing Request (CSR) 文件后，即可进一步得到最后的证书。完整的 iOS 证书包含公钥与私钥 (非对称加密)，公钥用于验证，私钥用于签名。

> **CSR** 是 **Cerificate Signing Request** 的英文缩写，即证书请求文件。证书申请者在申请数字证书时由 CSP (加密服务提供者)在生成私钥的同时也生成证书请求文件。证书申请者只要把 CSR 文件提交给证书颁发机构后，证书颁发机构使用其根证书私钥签名就生成了证书公钥文件，也就是颁发给用户的证书。

通常仅包含公钥的证书文件 (.cer) 将会被放置在开发帐号下提供开发团队人员下载共享使用。但是仅包含公钥的证书是不具备签名能力的，而私钥又保存在生成证书的机器 Keychain 内，所以当其他开发人员需要使用这份证书时候，我们需要将完整的公钥私钥导出生成个人信息交换文件 (.p12)，这样的证书环境才是完整可用的。

![1](/img/article/ios-certificate/certificates.png)

证书分为开发（Developerment）与发布（Distribution）两类，各自用途顾名思义，这里不再累述。

如果你的应用需要包含推送功能，那么还需要申请推送开发证书（APNs Development）和推送生产证书（Apple Push Services）。

> 有时开发人员当下的开发环境没有包含可用的私钥，会通过 revoke 操作重新申请证书。暴力操作过后其他开发人员的旧证书（包含私钥）将不可用，需要 revoke 的开发人员将最新的证书信息同步出来。

### 标识 (Identifiers)

注册一个 AppID 用于唯一标识一个 App 或一组 App，这里的应用程序 AppID 和 BundleID 是相对应的。为了确保 AppID 的唯一性，它的命名必须严格按照规范：

1. App（主程序、插件）BundleID：
   com.company.appname
   com.company.appname.extensionname
2. AppGroupsID：
   group.com.company.appname
3. Pass TypeID：
   pass.com.company.appname
4. Website PushID：
   web.com.company.appname
5. iCloud Containers ID：
   cloud.com.company.appname
6. Merchant ID：
   merchant.com.company.merchantname

每个 AppID 可以设置对应的服务开关（如 APNs、Game Center、iCloud 等等），生成时按照实际需要对应配置即可。

### 设备 (Device)

设备指的就是可调试的 iOS 设备，可以是 iPhone、iPad、iPod、Apple Watch 甚至是 Apple TV。新增一个设备到帐号下可以进行设备调试，仅需要提供对应名称与 UDID，但是，一个萝卜一个坑，一个帐号最多仅支持加入 100 个设备，即便你后续删除设备，用掉的名额也不会立刻恢复，直到来年开发者帐号的 membership year 开始时，才能选择删掉一些设备来恢复名额，或清空所有设备恢复到最多 100 个名额。

### 描述文件 (Provisioning Profile)

Provisioning Profile 文件将上文提及的相关信息 (Certificates、Identifiers、Device) 都打包在内，用于开发人员真机调试或者用于发布应用。Provision Profile 本质上是一个 plist 文件，以 development 为例，它一般包含但并不只以下内容：

- AppIDName
- ApplicationIdentifierPrefix
- CreationDate
- DeveloperCertificates
- Entitlements
- ExpirationDate
- ProvisionedDevice
- UUID

![2](/img/article/ios-certificate/provisioning_profile.png)

系统根据描述文件提供的信息进行一一校验，从 AppID 到 Entitenments，从数字证书到设备 UDID等，最后运行应用程序到设备。如下所示：

![3](/img/article/ios-certificate/provisioning_profile_2.jpg)

Distribution Provisioning Profile 与 Development Provisioning Profile 类似，不过它没有 Device 信息。

## 上手制作证书

通过上文了解了 iOS 证书体系后，我们实际上手为一个 iOS 应用申请证书。

> 实际操作前，请确保你已经申请了苹果开发者账号（个人、公司账号 99 美元，企业账号 299 美元）。
>
> 申请个人、公司账号地址：[https://developer.apple.com/programs/](https://developer.apple.com/programs/)  
> 申请企业账号地址：[https://developer.apple.com/programs/enterprise/](https://developer.apple.com/programs/enterprise/)

### 创建 App ID

申请证书前首先要为应用创建 App ID，App ID 用于唯一标识一个 App 或一组 App。登录Apple Member Center ：[https://developer.apple.com/membercenter](https://link.jianshu.com/?t=https://developer.apple.com/membercenter)，选择 **certificates，identifiers & profiles ** 选项，如下图所示：

![4](/img/article/ios-certificate/app_id_1.jpg)

在左侧菜单选择 App IDs，然后点击右上角的添加图标，在接下来的页面里面填写 App ID 描述。

- Name 填写应用的名称，
- App ID Suffix 栏选择 Explicit App ID，设置一个 Bundle ID，这个要与 APP 的 BundleID 一致，必须严格按照规范填写为 com.company.appname，
- 在 App Services 中选择服务功能，勾选上 Push Notifications 项，来开通 Push 功能。

点击 Continue 进入下一步。

![6](/img/article/ios-certificate/app_id_2.jpg)

在新页面中点击 register 按钮，然后点击 Done，创建 App ID 成功。

### 生成 Certificates Signing Request (CSR) 文件 

创建好 App ID 后，我们正式开始创建证书。如前文所说，我们首先需要通过 Keychain 的证书助理生成 Certificates Signing Request (CSR) 文件，然后可以进一步得到最后的证书。如图，应用程序->实用工具->钥匙串访问，打开 Keychain：

![7](/img/article/ios-certificate/csr_1.png)

打开“证书助理”，选择从证书颁发机构请求证书。

![8](/img/article/ios-certificate/csr_2.png)

接下来填写邮件地址，选择存储到磁盘，点击继续。

![9](/img/article/ios-certificate/csr_3.png)

如图，我们将 CSR 文件保存文件到桌面。

![10](/img/article/ios-certificate/csr_4.png)

### 制作 p12 证书

上文讲过，完整的 iOS 证书包含公钥与私钥，公钥用于验证，私钥用于签名。证书文件 (.cer) 仅包含公钥，不具备签名能力，而私钥又保存在生成证书的机器 Keychain 内，所以当其他开发人员需要使用这份证书时候，我们需要将完整的公钥私钥导出生成个人信息交换文件 (.p12)，这样的证书环境才是完整可用的。

下面我们以制作发布 (Distribution) 证书为例，演示 p12 证书的创建过程。如图，点击左边的 Production，在右边出来的页面的右上角选择添加：

![11](/img/article/ios-certificate/distribution_1.png)

如果是个人或公司账号，选择 App Store and Ad Hoc，如果是企业账号，则选择 In-House and Ad Hoc，点击 Continue 进入下一步，在下一页中点击 Continue。

![12](/img/article/ios-certificate/distribution_2.png)

如图，选择 Choose File 选择之前生成的 Cert Signing Request 文件，点击 Generate

![13](/img/article/ios-certificate/distribution_3.png)

如图所示，cer 证书创建成功，点击 Download 将证书下载到本地，然后双击打开证书，将其安装到钥匙串 Keychain。

![14](/img/article/ios-certificate/distribution_4.png)

在钥匙串中找到安装的证书，若提示此证书是由未知颁发机构签名的，请下载 Apple Worldwide Developer Relations Certification Authority 证书进行安装，地址 [http://developer.apple.com/certificationauthority/AppleWWDRCA.cer](http://developer.apple.com/certificationauthority/AppleWWDRCA.cer)，在左边选择“登录”和“我的证书”，找到证书，在证书上面点击鼠标右键，然后在菜单中选择导出证书，如图：

![15](/img/article/ios-certificate/distribution_5.png)

在弹出页面中指定证书名，点击存储，然后输入证书密码，点击好，生成 p12 格式证书。

![16](/img/article/ios-certificate/distribution_6.png)

> 收到 p12 文件后，**双击**该文件，会出现使用文件的密码输入框，**输入该 .p12 文件的使用密码**（就是生成的时候设的，不知道就问给你该文件的人），从而配置该文件到本地。然后开发者账号对应的开发证书和发布证书都可以 download 使用了。

### 制作描述文件 (Provisioning Profile) 

#### provision发布描述文件制作

下面以个人、公司账号创建 App Store 类型发布证书为例，企业账号创建 In House 类型发布描述文件类似。如图，点击左侧菜单 Distribution，然后点击右侧页面右上角的添加图标，最后选择 App Store，点击 Continue 进入下一步：

![17](/img/article/ios-certificate/provision_1.png)

如图，选择上面创建的 App ID，点击 Continue 进入下一步。

![18](/img/article/ios-certificate/provision_2.png)

如图，选择 certificates，点击 Continue 进入下一步。

![19](/img/article/ios-certificate/provision_3.png)

输入描述文件名称，点击 Generate，进入下一步完成创建。

![20](/img/article/ios-certificate/provision_4.png)

#### provision开发描述文件制作

个人或公司账号生成的 App Store 类型 mobileprovision 描述文件，应用在没有发布到 App Store 之前只能在越狱设备上安装，若要在非越狱手机上面安装，则需要把设备 UDID 添加到测试设备列表 Devices 里，并且生成 Ad Hoc 类型 mobileprovision 描述文件。

**添加测试设备**

首先获取设备的 UDID，打开 iTunes，连接设备，如图，找到序列号，然后点击序列号，该栏会变成 UDID，点击鼠标右键，拷贝 UDID。

![21](/img/article/ios-certificate/provision_5.png)

![22](/img/article/ios-certificate/provision_6.png)

回到网站页面，如图选择左侧菜单 Devices 下面的 All，在右侧页面点击右上角添加图标，进入下图所示页面：

![23](/img/article/ios-certificate/provision_7.png)

输入 Name 和获取的 UDID，点击 Continue 进入下一页，下一页中点击 Register，最后点击 Done，添加设备完成。

**Ad Hoc类型描述文件**

对于个人和公司账号，Ad Hoc 类型描述文件可以安装到指定的测试设备上面调试。如图，选择 Ad Hoc，点击 Continue 进入下一步。

![24](/img/article/ios-certificate/provision_8.png)

如图，选择 App ID，点击 Continue 进入下一步。

![25](/img/article/ios-certificate/provision_9.png)

如图，选择 certificates，点击 Continue 进入下一步。

![26](/img/article/ios-certificate/provision_10.png)

选择设备，然后点击 Continue。

![27](/img/article/ios-certificate/provision_11.png)

输入证书名称，点击 Generate，进入下一步完成创建。

![28](/img/article/ios-certificate/provision_12.png)

### 制作推送证书 

如果你的 App 具有推送功能，那么推送证书是必须的。推送证书同样也分为开发版和发布版：

**开发推送证书**

选择 **Apple Push Notification service SSL（Sandbox）**

![29](/img/article/ios-certificate/push_notification_1.jpeg)

选中 APP ID 后，点击 **continue**，直到上传 CSR 文件，如下图所示：

![30](/img/article/ios-certificate/push_notification_2.jpeg)

![31](/img/article/ios-certificate/push_notification_3.jpeg)

![32](/img/article/ios-certificate/push_notification_4.jpeg)

![33](/img/article/ios-certificate/push_notification_5.jpeg)

> 推送证书是对应 APP ID 的。这与开发证书是有区别的。

点击 Download，将证书下载到本地后，**双击下载的开发环境推送证书**，就可以在钥匙串访问中的我的证书中找到。与前文类似，如果是团队合作，可以将证书导出 .p12文件，供其他开发者在各自电脑上安装。

**发布推送证书**

如果需要生成生产环境的推送证书请选择： Apple Push Notification service SSL （Sandbox&Production），如下图所示：

![34](/img/article/ios-certificate/push_notification_6.jpg)

## 证书管理

开发团队的人员越多，合理的证书管理愈显重要。流程上，为保证日常开发应用程序的安全与效率，无论是大到公司的发布证书抑或是小到项目组的团队个人开发证书，我们都尽量做到：

1. 帐号密码统一由一个关键接口人维护 (证书管理员)；
2. 开发人员统一到证书管理员领取 .p12 文件与 Provision Profile 文件进行应用开发或发布；
3. 新增设备，提供设备名与 UDID 到证书管理员添加。证书管理员更新后，周知并同步新的ProvisioningProfile 到团队开发人员。
4. 开发描述文件可以由开发者自己管理，发布描述文件由证书管理员管理。

一个开发者账户只能申请 **3 个发布证书，2 个开发证书**，一般在我们的证书界面中应该只有一个开发证书，一个发布证书，所有 App 和所有开发者共用一个证书即可，证书一般在过期之后才会重新添加。

开发描述文件应该是团队里的每一个开发者都有权去管理的，实际上当开发类型的描述文件出现问题的时候，开发者可以对此描述文件重新编辑一下进行使用，这样是不会影响其他开发者的，甚至你可以自己重新制作一个描述文件也没什么问题。

## 参考

陈泽滨[《关于 iOS 证书，你必须了解的知识》](https://cloud.tencent.com/developer/article/1004883)  
APICloud[《iOS证书及描述文件制作流程》](https://docs.apicloud.com/Dev-Guide/iOS-License-Application-Guidance)   
Alfred_xyz[《iOS证书配置实践》](https://www.jianshu.com/p/fce459fbd10f)