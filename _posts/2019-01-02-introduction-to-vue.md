---
layout: article
title: Vue.js 快速入门
tags:
    - Vue
    - 前端
mathjax: false
---

> 该文为[慕课网](https://www.imooc.com/)[《3小时速成 Vue2.x 核心技术》](https://www.imooc.com/learn/1091)的学习笔记，讲师 [wayearn](https://www.imooc.com/u/2432190/courses?sort=publish)

## 前言

[**Vue.js**](https://cn.vuejs.org/) 是一个用于创建用户界面的开源 JavaScript 框架，也是一个创建单页面应用的Web应用框架。Vue 所关注的核心是[ MVC 模式](https://zh.wikipedia.org/wiki/MVC)中的视图层，同时，它也能方便地获取数据更新，并通过组件内部特定的方法实现视图与模型的交互。

俗话说“工欲善其事，必先利其器”，我们首先配置一下 Vue 的开发环境：

- 开发环境 (IDE)：[WebStorm](https://www.jetbrains.com/webstorm/) 或 [VS Code](https://code.visualstudio.com/)

- Node 开发环境：[Node.js](https://nodejs.org/) 和 [包管理工具npm](https://www.npmjs.com/)

  推荐通过 [nvm](https://github.com/creationix/nvm) 来安装和管理 Node 环境。安装好 NVM 后，通过 `nvm ls` 和 `nvm ls-remote` 可以查看本地和远程的 Node 环境版本，然后通过 `nvm install vX.X.X` 来安装指定的版本，npm 也会自动安装。本地多个 node 版本之间可以通过 `nvm use vX.X.X` 来切换。

- 调试环境：[Chrome浏览器](https://www.google.com/chrome/) **+** [Vue.js devtools 插件](https://chrome.google.com/webstore/detail/vuejs-devtools/nhdogjmejiglipccpnnnanhbledajbpd?hl=zh-CN)

- 工程环境：[Vue CLI](https://cli.vuejs.org/)，通过 `npm install -g vue-cli` 安装，速度特别慢可以考虑[淘宝 NPM 镜像](https://npm.taobao.org/)。

## Vue 框架常用知识点

### 第一个 Vue 应用

这里为了方便，我们在一个 html 页面里直接通过 CDN 方式引入 vue.js：

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Vue 测试实例</title>
    <script src="https://cdn.bootcss.com/vue/2.5.21/vue.min.js"></script>
</head>
<body>
<div id="app">
  <p>{% raw %}{{ message }}{% endraw %}</p>
</div>

<script>
new Vue({
  el: '#app',
  data: {
    message: 'Hello Vue.js!'
  }
})
</script>
</body>
</html>
```

[点击查看代码效果](http://www.runoob.com/try/try.php?filename=vue2-hw)。

我们首先 new 一个 Vue 对象，然后通过 `el: '#app'` 语句将它与上面 `id="app"` 的 div 绑定。在 `data` 中我们定义页面上使用到的数据 `message`，然后通过双大括号的方式 `{% raw %}{{ message }}{% endraw %}` 将它显示到模板中。

可以看到在页面中集成 Vue 框架非常方便，并且语法清晰，使用便捷。

### 模板语法

如上面例子中所示，数据绑定最常见的形式就是使用 {% raw %}{{...}}{% endraw %} 双大括号将变量显示到模板中。事实上，我们可以在模板中插入 JS 运算表达式，例如：

```html
<div id="app">
    <p>{% raw %}{{ (count + 1)*10 }}{% endraw %}</p>
</div>
```

还可以通过 v-html 直接在模板中输出 html 代码，[试一试](http://www.runoob.com/try/try.php?filename=vue2-v-html)：

```html
<div id="app">
    <div v-html="message"></div>
</div>
    
<script>
new Vue({
  el: '#app',
  data: {
    message: '<h1>hello template</h1>'
  }
})
</script>
```

有的时候，我们需要对一些属性进行修改，可以通过 v-bind 语法完成属性数据的绑定：

```html
<div id="app">
    <a v-bind:href="url">百度</a>
</div>
    
<script>
new Vue({
  el: '#app',
  data: {
    url: 'https://www.baidu.com/'
  }
})
</script>
```

这样页面中所有的属性都可以包含到 vue 对象的 data 中来，方便对属性进行管理和修改。

另外一个常用的语法是 v-on，用于监听 DOM 事件：

```html
<div id="app">
    <p>{% raw %}{{count}}{% endraw %}</p>
    <button type="button" v-on:click="submit()">加数字</button>
</div>
    
<script>
new Vue({
  el: '#app',
  data: {
    count: 0
  },
  methods: {
    submit: function() {
      this.count++
    }
  },
})
</script>
```

上面我们对按钮的 click 事件绑定了一个 `submit()` 函数，用于将 count 变量的值加一，vue 中的函数包含在对象的 methods 中。点击按钮，就会触发 methods 中的 `submit()` 方法改变 count 的值，并实时地刷新到页面中。

> **小技巧：** `v-on:` 语法可以简写为 `@`，例如 `v-on:click="submit()"` 可以简写为 `@click="submit()"`。`v-bind:` 语法可以简写为 `:`，例如 `v-bind:href="url"` 可以简写为 `:href="url"`。

在网页中处理用户的输入数据也是常见的操作，vue 可以使用 v-model 语法来实现双向数据绑定，[试一试](http://www.runoob.com/vue2/vue-template-syntax.html)：

```html
<div id="app">
    <input v-model="message">
    <p>{% raw %}{{ message }}{% endraw %}</p>
</div>

<script>
new Vue({
  el: '#app',
  data: {
    message: ''
  }
})
</script>
```

### 计算属性与侦听器

vue 中计算属性和侦听器分别对应于对应于对象中的 `computed` 和 `watch` 属性。

例如，我们要监听 message 变量的变动，就在 watch 属性下为 message 变量定义对应的监听方法，方法有两个参数 `newval` 和 `oldval`，分别对应变量的新值和旧值：

```html
<div id="app">
    <p>{% raw %}{{ message }}{% endraw %}</p>
</div>

<script>
var vue = new Vue({
  el: '#app',
  data: {
    message: 'Hello Vue'
  },
  watch: {
    message: function(newval, oldval) {
      console.log('new val is:' + newval)
      console.log('old val is:' + oldval)
    }
  }
})
</script>
```

为了方便调试，我们给 vue 对象一个变量名，然后在控制台直接通过 `vue.message = 'new message'` 修改 message 变量的值，控制台就会输出：

```bash
new val is:new message
old val is:Hello Vue
```

计算属性可以看成是一种特殊的变量，它的值由函数来定义。例如我们定义一个 `msg` 变量：

```html
<div id="app">
    <p>{% raw %}{{ message1 }}{% endraw %}</p>
    <p>{% raw %}{{ message2 }}{% endraw %}</p>
    <p>{% raw %}{{ msg }}{% endraw %}</p>
</div>

<script>
var vue = new Vue({
  el: '#app',
  data: {
    message1: 'Hello',
    message2: 'Vue'
  },
  computed: {
    msg: function() {
      return 'computed:' + this.message1 + this.message2
    }
  }
})
</script>
```

`msg` 变量的值通过对 `message1` 和 `message2` 变量值的计算得到，如果我们修改 `message1` 或者 `message2` 的值，那么 `msg` 的值也会发生变化。即对于计算属性而言，函数中包含的任意一个变量的值发生变化，计算属性的值就会发生变化，相当于同时监听了很多个变量。

### 条件渲染、列表渲染

很多时候我们需要在满足某些条件时对页面进行渲染，这时就可以使用条件渲染；而列表渲染则非常常见，比如新闻网站，海量的新闻在展示时不可能手工编写页面，都是通过列表渲染自动生成的。

条件渲染对应语法为 `v-if`，还可以伴随有 `v-else`、`v-else-if` 等语法：

```html
<div id="app">
    <div v-if="count > 0">
      count大于0，值为:{% raw %}{{count}}{% endraw %}
    </div>
    <div v-else-if="count == 0">
      count等于0
    </div>
    <div v-else>
      count小于0，值为:{% raw %}{{count}}{% endraw %}
    </div>
</div>

<script>
var vue = new Vue({
  el: '#app',
  data: {
    count: -5
  }
})
</script>
```

```
count小于0，值为:-5
```

列表渲染的语法为 `v-for`，类似于编程中的循环语句，通过绑定数组来渲染一个列表。v-for 指令需要以 `item in list` 形式的特殊语法， list 是源数据数组，item 是数组元素迭代的别名。例如：

```html
<body>
<div id="app">
  <ul>
    <li v-for="site in sites">
      {% raw %}{{ site.name }}{% endraw %}
    </li>
  </ul>
</div>
  
<script>
new Vue({
  el: '#app',
  data: {
    sites: [
      { name: 'Google' },
      { name: 'Taobao' },
      { name: 'Tencent'}
    ]
  }
})
</script>
```

```
Google
Taobao
Tencent
```

### Class 和 Style 的绑定

我们都知道 class 与 style 是 HTML 元素的属性，用于设置元素的样式。前面已经介绍过 vue 使用 v-bind 来设置属性，而样式也是属性，因此也使用 v-bind 语法。并且 Vue 在处理 class 和 style 时，专门增强了 v-bind，表达式的结果类型除了字符串之外，还可以是对象或数组。例如：

```html
<div id="app">
  <div v-bind:style="red">红色的字</div>
</div>
  
<script>
new Vue({
  el: '#app',
  data: {
    red: {
      'color': 'red',
      'text-shadow': '0 0 5px yellow'
    }
  }
})
</script>
```

我们还可以为 v-bind:class 设置一个对象，从而动态的切换 class：

```html
<div v-bind:class="{ active: isActive }"></div>
```

 这里，如果 `isActive` 设置为 true 则该 div 的 class 为 active，如果设置为 false 则 class 为空。也可以在对象中传入更多属性用来动态切换多个 class 。

```html
<div id="app">
  <div class="static"
     v-bind:class="{ active: isActive, 'text-danger': hasError }">
  </div>
</div>

<script>
new Vue({
  el: '#app',
  data: {
    isActive: true,
	hasError: true
  }
})
</script>
```

其结果为：

```html
<div class="static active text-danger"></div>
```

我们还可以直接把一个数组传给 **v-bind:class**，例如：

```html
<div v-bind:class="['active', 'text-danger', {'anothor': true}]"></div>
```

其结果为：

```html
<div class="active text-danger anothor"></div>
```

注意，这里为了演示的简单，所以直接赋值了 true 或 false，在通常情况下这里是一个逻辑判断表达式。例如：

```html
<div id="app">
  <div v-for="num in numbers" v-bind:class="{red: num % 2 != 0}">
    {% raw %}{{num}}{% endraw %}
  </div>
</div>
  
<script>
new Vue({
  el: '#app',
  data: {
    red: {
      'color': 'red',
      'text-shadow': '0 0 5px yellow'
    },
    numbers: [1,2,3,4,5]
  }
})
</script>
```

可以让奇数的 class 为 red。

## Vue 核心技术

### Vue CLI

Vue CLI 是官方提供的 Vue 命令行工具，可用于快速搭建大型单页应用。在第一节中，我们已经通过 `npm install -g vue-cli` 安装好了 Vue CLI。接下来我们通过 Vue CLI 来创建标准的 Vue 项目。

首先切换到要创建项目的目录，然后通过 `vue create 项目名` 来创建项目，例如：

```bash
$ vue create hello-world
```

切换到手动选择模式 (Manually select)：

```bash
? Please pick a preset: 
  default (babel, eslint) 
❯ Manually select features 
```

接下来选择安装到组件（按空格选中），这里我们选择一些常用的组件（路由组件 Router、状态控制组件 Vuex、CSS 预编译组件等）：

```bash
? Please pick a preset: Manually select features
? Check the features needed for your project: 
 ◉ Babel
 ◯ TypeScript
 ◯ Progressive Web App (PWA) Support
 ◉ Router
 ◉ Vuex
❯◉ CSS Pre-processors
 ◉ Linter / Formatter
 ◯ Unit Testing
 ◯ E2E Testing
```

代码规范建议选择 ESLint + Airbnb config 或者 ESLint + Standard config，并且选择在保存时就检查代码（Lint on save）：

```bash
? Use history mode for router? (Requires proper server setup for index fallback 
in production) Yes
? Pick a CSS pre-processor (PostCSS, Autoprefixer and CSS Modules are supported 
by default): Sass/SCSS
? Pick a linter / formatter config: Airbnb
? Pick additional lint features: (Press <space> to select, <a> to toggle all, <i
> to invert selection)Lint on save
? Where do you prefer placing config for Babel, PostCSS, ESLint, etc.? In dedica
ted config files
? Save this as a preset for future projects? (y/N) n
```

接下来等待 Vue CLI 根据我们的配置自动下载好插件和依赖后，Vue 项目就创建完成了。接下来，使用

```bash
$ cd hello-world
$ npm run serve
```

就可以运行起我们的 Vue 项目了，默认网页服务端口为 8080，可以直接通过浏览器访问：

![1](/img/article/introduction-to-vue/vue_1.png)

打开我们创建的 Vue 项目，可以看到以下目录和文件：

- node_modules：前端依赖，
- public：存放公共资源，
- src：存放源文件，
- package.json：项目配置文件

其中 public/index.html 是项目的入口，src/main.js 是项目的主 JS 文件。

在 main.js 中我们可以看到，创建项目时选择的路由组件 Router 和状态控制组件 Vuex（store）都被引入并绑定到 Vue 对象中，然后这个 Vue 对象再被绑定到 index.html 中的 `<div id="app"></div>` 上：

```js
import Vue from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';

Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: h => h(App),
}).$mount('#app');
```

> 注：这里的 `$mount` 挂载与前面单 html 中 vue 对象的 `el` 属性是一样的。

我们编写的代码都存放在 src 目录下，src 目录下又包含了以下文件夹：

- views：视图目录，下面的 Home.vue 和 About.vue 分别是首页和关系页，
- components: 组件目录，其中的 HelloWord.vue 组件被引入到 views/Home.vue 视图中。

### 组件化思想

Vue 中的组件是独立的、可复用的、整体化的：一个组件相当于是一个模块，独立存在于 Vue 的应用中；组件是可复用的，一个组件可以被用在多个页面中；组件内部包含整个组件所需要用到的业务逻辑和样式。组件化的最主要的目的是实现功能模块的复用，并且能够提高执行效率（高效的 DOM 渲染），还可以方便地拆分复杂的业务逻辑从而开发单页面复杂应用。

业务逻辑拆分原则：

- 300 行原则：每一个组件的代码控制规模，方便阅读和维护
- 复用原则：组件应该经常被使用，例如头部的导航、底部的版权信息、侧边栏等等
- 业务复杂性原则

> 组件化也会带来一些问题，例如组件状态的管理 (vuex)、多组件的混合使用，多页面，复杂业务 (vue-router) 以及组件间的传参、消息、事件管理 (props, emit/on, bus)。

### Vue-router

Vue-router 是官方的路由管理工具，可以非常方便地实现单页应用。我们通过 Vue CLI 创建项目时选择了 Router 组件，所以项目中已经包含了 router.js 文件。

打开 src/router.js 文件，里面默认定义好了两个路由：

```js
import Vue from 'vue';
import Router from 'vue-router';
import Home from './views/Home.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import(/* webpackChunkName: "about" */ './views/About.vue'),
    },
  ],
});
```

根目录 `/` 使用到的是引入自 `'./views/Home.vue'` 的 Home 组件，`\about` 使用到的是 `'./views/About.vue'` 组件。

接下来我们自己创建一个路由 Info，首先新建 views/Info.vue 组件。标准的组件由三部分组成：template 模板、script 组件执行的代码、style 样式。

```vue
<template>
    <div>
        Hello Info Component
    </div>
</template>

<script>
export default {
    
}
</script>

<style scoped>

</style>
```

接下来我们将这个组件添加到路由中，修改 src/router.js 为：

```js
import Vue from 'vue';
import Router from 'vue-router';
import Home from './views/Home.vue';
import Info from './views/Info.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
    },
    {
      path: '/info',
      name: 'info',
      component: Info,
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import(/* webpackChunkName: "about" */ './views/About.vue'),
    },
  ],
});
```

并且在页面中添加 router-link，修改 src/App.vue 中的 template 为：

```vue
<template>
  <div id="app">
    <div id="nav">
      <router-link to="/">Home</router-link> |
      <router-link to="/about">About</router-link> | 
      <router-link to="/info">Info</router-link>
    </div>
    <router-view/>
  </div>
</template>
```

打开浏览器，访问 `http://localhost:8080` 就可以点击切换到 Info 页面了：

![2](/img/article/introduction-to-vue/vue_2.png)

### Vuex

单向数据流如下图所示：

![3](/img/article/introduction-to-vue/data_flow.png)

我们的页面是由多个视图 (View) 组成的，用户的操作 (Actions) 会带来视图上一些状态 (State) 的变化，而状态的变化又会驱动视图的更新。

一些情况下我们对状态的管理会很复杂：

- 多个状态依赖于同一状态，比如菜单导航有很多 TAB，点击某一个 TAB，其他 TAB 的状态需要变成未激活的状态。
- 来自不同视图的行为需要变更同一状态，例如评论弹幕

Vuex 应运而生，它是为 Vue.js 开发的状态管理模式，实现组件状态的集中管理，组件状态改变遵循统一的规则。

我们通过 Vue CLI 创建项目时选择了 Vuex 组件，所以项目中已经包含了 store.js 文件。打开 src/store.js 文件：

```js
import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {

  },
  mutations: {

  },
  actions: {

  },
});
```

`state` 是组件的状态，这里对状态进行集中管理。`mutations` 是唯一可以改变 Vuex 中状态的方法集，所有方法在这里定义。

接下来我们通过一个实例来演示多页面状态的共享。首先在 src/store.js 中创建一个状态 count，并创建对应的计数方法：

```js
import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    count: 0
  },
  mutations: {
    increase() {
      this.state.count++
    }
  },
  actions: {

  },
});
```

然后在前面创建的 views/Info.vue 组件中添加一个按钮，用于调用计数方法改变 count 状态的值：

```vue
<template>
    <div>
        Hello Info Component
        <button type="button" @click="add()">添加</button>
    </div>
</template>

<script>
import store from '@/store.js' // 
export default {
    name: 'Info',
    store, // 引入 store
    methods: {
        add() {
            store.commit('increase')
        }
    }
}
</script>

<style scoped>

</style>
```

这里在 script 中通过 `import store from '@/store.js'` 引入 Vuex 组件，这里通过 config文件将 @ 指向 src 目录。在按钮点击事件绑定的函数 add() 中，我们通过 `store.commit('increase')` 调用前面 store 中定义的 increase 方法。

打开浏览器，点击页面上的“添加”按钮，chrome 调试窗口切换到我们安装好的  [Vue.js devtools 插件](https://chrome.google.com/webstore/detail/vuejs-devtools/nhdogjmejiglipccpnnnanhbledajbpd?hl=zh-CN)，就可以看到 Vuex 中 count 状态的改变：

![4](/img/article/introduction-to-vue/chrome_vue.png)

接下来我们在 About 组件中引入并显示这个 count 状态，修改 views/About.vue 为：

```vue
<template>
  <div class="about">
    <h1>This is an about page</h1>
    <p>{% raw %}{{ msg }}{% endraw %}</p>
  </div>
</template>

<script>
import store from '@/store.js'
export default {
  name: 'About',
  store,
  data() {
    return {
      msg: store.state.count
    }
  }
}
</script>
```

与 Info.vue 类似，我们通过 `import store from '@/store.js'` 引入 Vuex，然后通过 `store.state.count` 来访问 store 中的状态。打开浏览器，切换到 About 页面就可以看到 store 中的 count 状态的值：

![5](/img/article/introduction-to-vue/chrome_vue_2.png)

所以 Vuex 的使用可以总结为：

1. 创建引用 Vuex 组件的 store.js 文件，然后定义 state 和 mutations，分别对应组件公用的状态和改变状态的方法。
2. 在 Vue 组件中通过 `import store from '@/store.js'` 引入 store 文件，然后在组件的 default 对象中引入 store。之后，通过 `store.commit()` 方法调用 store 中的方法修改状态的值。

## 参考

[Vue.js 教程](http://www.runoob.com/vue2/vue-tutorial.html)
[3小时速成 Vue2.x 核心技术](https://www.imooc.com/learn/1091)
[Vue.js维基百科](https://zh.wikipedia.org/wiki/Vue.js)

