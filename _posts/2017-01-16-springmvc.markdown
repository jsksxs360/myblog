---
layout: article
title: Spring MVC 快速入门：快速开发一个 Java 网站
tags:
    - Java
mathjax: true
---

### 1. 前言

Spring Web MVC 是一种轻量级的 Web 框架，它实现了 Web MVC 设计模式，能够简化日常的 Web 开发。本文将通过一个简单的示例，让大家对 Spring MVC 框架有一个大概的认识。(最下方可以直接下载源码)

### 2. 新建项目

我们使用 Eclipse 进行 Spring MVC 的开发，选择新建一个动态 Web 项目(Dynamic Web Project)。

![pic](/img/article/springmvc/eclipse.png)

项目建好之后，目录结构如下：

![pic](/img/article/springmvc/dir.png)

简单说明一下各文件夹的作用：

- **src** : 存放 Java 项目的源代码(与普通项目相同)，包括后面编写的 Web 请求响应逻辑，都存放在这里。
- **WebContent** : 存放网页、css、js 等网站内容，可以把 WebContent 目录看成是服务器的根目录，所有的网站内容都存放在这里。
  * **WEB-INF** : Java Web 项目的安全目录，只有服务端可以访问，因而实际跳转到的 jsp 页面，包括项目引用的 jar 包都存放在这里。

开发 Spring MVC 项目首先需要导入 Spring 框架的 jar 包，可以到[官方 maven 仓库](http://maven.springframework.org/release/org/springframework/spring/)下载。Spring 框架包含大量的 jar 包，在一般的 Web 开发中，我们只需引入下面这些就足够了：

- spring-aop-x.x.x.RELEASE.jar
- spring-beans-x.x.x.RELEASE.jar
- spring-context-x.x.x.RELEASE.jar
- spring-core-x.x.x.RELEASE.jar
- spring-expression-x.x.x.RELEASE.jar
- spring-web-x.x.x.RELEASE.jar
- spring-webmvc-x.x.x.RELEASE.jar

为了能打印 log 信息，通常还会引入 **commons-logging-x.x.x.jar**，可以到 [apache 官网](http://commons.apache.org/proper/commons-logging/download_logging.cgi)下载。

引入上面所列的 jar 包后，如下图所示：

![pic](/img/article/springmvc/dir_2.png)

### 3. 配置文件

#### 3.1 编写 Web 项目配置文件

首先我们在 **WEB-INF** 目录下，编写 web.xml 文件，作为整个 Java Web 项目的配置文件。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns="http://java.sun.com/xml/ns/javaee"
    xsi:schemaLocation="http://java.sun.com/xml/ns/javaee http://java.sun.com/xml/ns/javaee/web-app_2_5.xsd"
    id="WebApp_ID" version="2.5">
    
    <!-- 配置DispatchcerServlet -->
    <servlet>
        <servlet-name>springDispatcherServlet</servlet-name>
        <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
        <!-- 配置Spring mvc下的配置文件的位置和名称 -->
        <init-param>
            <param-name>contextConfigLocation</param-name>
            <param-value>classpath:springmvc.xml</param-value>
        </init-param>
        <load-on-startup>1</load-on-startup>
    </servlet>
    
    <servlet-mapping>
        <servlet-name>springDispatcherServlet</servlet-name>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
```

在 `servlet` 中，我们定义了 Spring MVC 配置文件的名称为 **springmvc.xml**，路径为项目 classpath (即 src 目录)下。

> 我们也可以不新建 springmvc.xml，直接使用默认的配置文件，格式为 **/WEB-INF/[servlet-name]-servlet.xml**，对应这里的就是 springDispatcherServlet-servlet.xml

`servlet-mapping` 表示拦截的模式，这里是 `/`，表示拦截所有的请求，包括静态资源(如 html、js、jpg 等)。这时候直接访问静态资源会报 404 错误，可以通过在 Spring MVC 配置文件中添加配置项解决。

#### 3.2 Spring MVC 配置文件

接下来我们在 **src** 目录下编写 Spring MVC 配置文件 Springmvc.xml。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans:beans xmlns="http://www.springframework.org/schema/mvc"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:beans="http://www.springframework.org/schema/beans"
    xmlns:context="http://www.springframework.org/schema/context"
    xsi:schemaLocation="http://www.springframework.org/schema/mvc http://www.springframework.org/schema/mvc/spring-mvc.xsd
        http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
        http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context.xsd">
        
        <annotation-driven />
        <!-- 配置静态资源路径 -->
        <resources mapping="/css/**" location="/css/" />
    	<resources mapping="/img/**" location="/img/" />
    	<resources mapping="/js/**" location="/js/" />
        
        <!-- 配置自动扫描的包 -->
        <context:component-scan base-package="me.xiaosheng.handlers"></context:component-scan>
        
        <!-- 配置视图解析器 如何把handler 方法返回值解析为实际的物理视图 -->
        <beans:bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
            <beans:property name = "prefix" value="/WEB-INF/views/"/>
            <beans:property name = "suffix" value = ".jsp"/>
        </beans:bean>
</beans:beans>
```

`resources` 表示对静态资源路径的映射， 这里我们配置了 **css**、**img**、**js** 目录的地址映射。例如对 `/css/` 路径下任何资源的访问请求，都会映射 **WebContent/css/** 目录下。

`context:component-scan` 表示 Spring 监听的范围，也就是我们编写的控制器(Controller)代码所在的包名，大家需要根据实际情况做相应的改变。

`beans:bean` 下我们添加了一个视图解析器，用于把在控制器中 handler 的结构解析为实际的物理视图。`prefix` 定义了物理视图的路径，也就是最终跳转到的 jsp 页面的位置，`suffix` 定义了物理视图的后缀。

> 当用户向服务器发起请求时，Spring MVC 首先在控制器(Controller)类里，查找有没有能够处理这个请求的方法。如果有，则转入相应的方法处理，处理完的结果为 handler 结构，还需要进一步解析为实际的物理视图，也就是某个 jsp 页面。最后将这个动态生成的 jsp 页面返回给用户。

### 4. 响应请求的逻辑代码

正如上面所说，Spring MVC 通过编写控制器(Controller)类来处理用户发起的请求，我们通过编写一个 HelloWorld 类来展示控制器类的编写方法。

```java
package me.xiaosheng.handlers;

import java.util.Arrays;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class HelloWorld {

    /**
     * 1. 使用RequestMapping注解来映射请求的URL
     * 2. 返回值会通过视图解析器解析为实际的物理视图, 对于InternalResourceViewResolver视图解析器，会做如下解析
     * 通过prefix+returnVal+suffix 这样的方式得到实际的物理视图，然后会转发操作
     * "/WEB-INF/views/success.jsp"
     * @return
     */
    @RequestMapping("/helloworld")
    public String hello() {
        return "success";
    }
    
    @RequestMapping("/aboutme")
    public String myIntroduction(String version, HttpServletRequest request) {
    	HttpSession session = request.getSession();
    	//通过session传递参数
    	if (version.equals("chinese")) { //中文版
    		session.setAttribute("version", "chinese");
    		session.setAttribute("main", "一个电脑爱好者");
    		List<String> hobbies = Arrays.asList("看书","玩游戏","睡觉");
    		session.setAttribute("hobbies", hobbies);
    	} else { //英文版
    		session.setAttribute("version", "english");
    		session.setAttribute("main", "a computer fan");
    		List<String> hobbies = Arrays.asList("reading","playing computer games","sleeping");
    		session.setAttribute("hobbies", hobbies);
    	}
    	return "me";
    }
}
```

要创建一个控制器类，只需在普通类的前面添加一个 `Controller` 的注解，就表示这是一个 Spring 的控制器。这里我们写了两个方法 **hello()** 和 **myIntroduction()**，分别用来处理用户发起的 `/helloworld` 和 `/aboutme` 请求。例如当用户在浏览器输入 *http://localhost:8080/springTest/helloworld* 时，就会跳转到 **hello()** 方法来处理。

控制器方法返回一个字符串，代表最终解析出的物理视图的名称，也就是我们要返回的 jsp 页面的名称。之前我们已经在 Spring MVC 配置文件 **springmvc.xml** 中声明了 `prefix` 和 `suffix`，而夹在这两者之间的就是这里返回的字符串。所以执行完 **hello()** 方法后，就会得到请求资源路径 */WEB-INF/views/success.jsp*。当然，这个 success.jsp 是需要我们新建的。

#### 4.1 带参数的请求

用户发起的请求也可以带参数，例如这里的 **myIntroduction()** 方法，就需要接收一个名为 version 的参数，用户可以通过 `/aboutme?version=chinese` 这样的方式向控制器传递参数。

> 如果用户没带参数，那么方法得到的对应参数值为 null。

#### 4.2 向物理视图传递参数

同样地，当控制器中的方法处理完，要跳转到 jsp 页面时，也可能需要向 jsp 页面传递参数。这里简单地直接通过 session 来传递，所以 **myIntroduction()** 方法还需要接受一个 HttpServletRequest 参数，用来获取 session 对象。

在控制器方法中我们通过 `session.setAttribute(名称, 对象)` 语句，将要传递的对象放入 session 中，在对应的 jsp 页面中，则通过 `session.getAttribute(名称)` 语句来获取 session 中的对象。

### 5. 物理视图

当控制器中方法处理完用户的请求后，需要将 handler 对象解析为实际的物理视图，也就是需要跳转到相应的 jsp 页面。我们之前编写的 **hello()** 和 **myIntroduction()** 方法，分别需要跳转到 success.jsp 和 me.jsp 页面，我们现在来编写他们。

根据 Spring MVC 配置文件 **springmvc.xml** 中的配置，我们在 **WEB-INF** 目录下新建 **views** 目录，然后在其中编写 success.jsp 和 me.jsp 页面。

#### success.jsp：

```html
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
<title>Insert title here</title>
</head>
<body>
<h4>Success Page</h4>
</body>
</html>
```

#### me.jsp

```html
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ page import="java.util.*" %>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html lang="zh-cn">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>关于小昇</title>
<link rel="stylesheet" type="text/css" href="css/cayman.css">
</head>
<body>
<section class="page-header">
<% String version = (String) session.getAttribute("version"); %>
<% String main = (String) session.getAttribute("main"); %>
<h1><% if (version.equals("chinese")) { %>关于小昇<% } else { %>About Xiaosheng<% } %></h1>
<h2><%= main %></h2>
<a href="http://github.com/jsksxs360" class="btn">GitHub</a>
<a href="http://xiaosheng.me" class="btn">Website</a>
</section>
<section class="main-content">
<h3 id="small-image"><% if (version.equals("chinese")) { %>关于我<% } else { %>About Me<% } %></h3>
<p><img src="img/avatar-xs.jpg"/></p>
<hr />
<h3><% if (version.equals("chinese")) { %>爱好<% } else { %>Hobbies<% } %></h3>
<ul>
<% List<String> hobbies = (List<String>) session.getAttribute("hobbies"); %>
<% for (String hobby : hobbies) { %>
<li><%= hobby %></li>
<% } %>
</ul>
</section>
</body>
</html>
```

可以看到 me.jsp 页面引用了 css 文件 **css/cayman.css** 和图片 **img/avatar-xs.jpg**，因为我们已经在 Springmvc.xml 文件中做了静态资源的映射，所以服务器会尝试加载目录 **WebContent/css/** 和 **WebContent/img/** 下的对应文件。

cayman.css 可以到[这里](/css/cayman.css)下载，vatar-xs.jpg 可以到[这里](/img/avatar-xs.jpg)下载。下载后分别放置在 **WebContent/css/** 和 **WebContent/img/** 下。

#### index.jsp

启动服务器后，默认会访问服务器器的根路径 `/`，也就是访问根目录下的 index.jsp 页面。

我们在根目录，即 **WebContent** 目录下创建 index.jsp，作为网站的入口。包含三个链接，分别是发起 *helloworld* 请求和带参数的 *aboutme* 请求。

```html
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>SpringMVC 测试网站</title>
</head>
<body>
<h1>SpringMVC 测试网站</h1>
<p><a href="helloworld">Hello World</a></p>
<p><a href="aboutme?version=chinese">关于小昇(中文版)</a></p>
<p><a href="aboutme?version=english">关于小昇(英文版)</a></p>
</body>
</html>
```

> 也可以不写 index.jsp 而是在控制器中，指定某一个方法处理对 `/` 路径的请求。只需在方法前面加上 **@RequestMapping(value={" ","/"})** 注解。

### 6. 运行网站

按照之前的步骤编写好所有的文件后，整个项目的结构应该如下图所示：

![pic](/img/article/springmvc/dir_3.png)

在 Tomcat 服务器上运行前面编写好的网站，首先会访问根路径 `/`，即访问 index.jsp 页面。

![pic](/img/article/springmvc/website.png)

点击 Hello World 发起对 **/helloworld** 的请求，即访问 success.jsp 页面。

![pic](/img/article/springmvc/website_2.png)

点击 关于小昇(中文/英文版)，发起带参数的对 **/aboutme** 的请求，即访问 me.jsp 页面。

![pic](/img/article/springmvc/website_3.png)

可以看到一个基本的 Spring MVC 网站就开发完成了。

### 源码

- [SpringMVCTest](https://github.com/jsksxs360/SpringMVCTest)

### 参考

- [《学习SpringMVC——从HelloWorld开始》](http://www.cnblogs.com/bigdataZJ/p/springmvc1.html)
