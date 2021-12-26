---
layout: article
title: C# 使用 Json.NET 解析 Json：Json.NET 简易指南
tags:
    - C#
    - Json
mathjax: false
---

>使用 [Json.NET](http://www.newtonsoft.com/json) 完成 .NET 对象的序列化和反序列化，以及对复杂 Json 数据的解析。

## 前言

最近在 C# 项目中需要使用到 Json 格式的数据，我简单上网搜索了一下，基本上有两种操作 Json 数据的方法：

- 使用 Windows 系统自带的类
- 使用第三方的包

本着**“第三方包一定有比系统自带类优秀地方，否则就不会存在”**的原则，再加上 [JavaScriptSerializer](https://msdn.microsoft.com/zh-cn/library/system.web.script.serialization.javascriptserializer.aspx)、[DataContractJsonSerializer](https://msdn.microsoft.com/zh-cn/library/system.runtime.serialization.json.datacontractjsonserializer.aspx) 等这些自带类库使用起来很麻烦，我毫不犹豫地就选择了在 Json 操作方面小有名气的 [Json.NET](http://www.newtonsoft.com/json)。Json.NET 自己也做了与自带类库的比较，详情可以见 [Json.NET vs .NET Serializers](http://www.newtonsoft.com/json/help/html/JsonNetVsDotNetSerializers.htm) 和 [Json.NET vs Windows.Data.Json](http://www.newtonsoft.com/json/help/html/JsonNetVsWindowsDataJson.htm)。

Json.NET 是一个 Newtonsoft 编写的开源类库包，你可以直接到 Github 上查看[项目的源码和说明](https://github.com/JamesNK/Newtonsoft.Json)。Json.NET 提供了大量对于 Json 数据的操作方法，他们使用起来非常简单，而且执行效率很高。

## .NET 对象的序列化和反序列化

### 普通对象的序列化和反序列化

[JsonConvert](http://www.newtonsoft.com/json/help/html/T_Newtonsoft_Json_JsonConvert.htm) 是 Json.NET 中的一个数据转换类，提供了用于 .NET 对象序列化和反序列化的方法 **SerializeObject()** 和 **DeserializeObject()**。在通常情况下，我们也只需要使用这两个方法就足以完成任务了。

比如说，我们现在定义了一个学生类 Student：

```csharp
class Student //学生类
{
	public int Id { get; set;} //学号
	public string Name { get; set;} //姓名
	public double[] Scores { get; set;} //成绩
}
```

现在我们创建一个学生类对象，并使用 **JsonConvert** 类提供的 **SerializeObject()** 方法将它转换到 Json 字符串（需要引入命名空间 using Newtonsoft.Json）：

```csharp
Student student = new Student
{
	Id = 12883,
	Name = "Jim David",
	Scores = new double[] { 87.5, 92, 76.2 }
};

string jsonStudent = JsonConvert.SerializeObject(student);
//{"Id":12883,"Name":"Jim David","Scores":[87.5,92.0,76.2]}
```

可以看到在序列化的过程中，**JsonConvert** 会将 .NET 对象中的变量名转换为 Json 中的*“属性”*，同时将变量的值复制为 Json 的*“属性值”*。接下来，我们尝试将 Json 字符串转换为 **Student** 对象，使用 **JsonConvert** 提供的 **DeserializeObject()** 方法：

```csharp
Student deserializedStudent = JsonConvert.DeserializeObject<Student>(jsonStudent);
Console.WriteLine("student.Id = " + deserializedStudent.Id);
//student.Id = 12883
Console.WriteLine("student.Name = " + deserializedStudent.Name);
//student.Name = Jim David
```

可以看到，创建的学生对象 **student** 的 Json 字符串被顺利地解析成了 Student 对象。在调用 **DeserializeObject()** 方法进行反序列化时，最好使用带泛型参数的重载方法。

> 如果在调用 **DeserializeObject()** 时不指定对象类型，**JsonConvert** 会默认转换为 Object 对象。

### 集合的序列化和反序列化

上面我们已经简单测试了 JsonConvert 提供的 **SerializeObject()** 和 **DeserializeObject()** 方法，完成了 .NET 对象的序列化和反序列化。

C# 项目中，除了自定义的类型外，集合（Collections）也是经常会使用的数据类型，包括列表、数组、字典或者我们自定义的集合类型。我们同样可以使用之前使用的 **SerializeObject()** 和 **DeserializeObject()** 方法来完成集合的序列化和反序列化。

> 为了使转换后的结果更加易读，我指定转换后的 Json 字符串带缩进。通过向 **SerializeObject()** 方法传递进第二个参数 Formatting 实现。

```csharp
Student student1 = new Student
{
	Id = 12883,
	Name = "Jim David",
	Scores = new double[] { 87.5, 92, 76.2 }
};
Student student2 = new Student
{
	Id = 35228,
	Name = "Milly Smith",
	Scores = new double[] { 92.5, 88, 85.7 }
};
List<Student> students = new List<Student>();
students.Add(student1);
students.Add(student2);
string jsonStudents = JsonConvert.SerializeObject(students, Formatting.Indented);
//[
//  {
//    "Id": 12883,
//    "Name": "Jim David",
//    "Scores": [
//      87.5,
//      92.0,
//      76.2
//    ]
//  },
//  {
//    "Id": 35228,
//    "Name": "Milly Smith",
//    "Scores": [
//      92.5,
//      88.0,
//      85.7
//    ]
//  }
//]
```

接下来我们对上面生成的 Json 字符串进行反序列化，解析出原有的 Student 类型列表。同样，我们需要使用带泛型参数的 **DeserializeObject()** 方法，指定 JsonConvert 解析的目标类型。

```csharp
string jsonStudentList = @"[
  {
    'Id': 12883,
    'Name': 'Jim David',
    'Scores': [
      87.5,
      92.0,
      76.2
    ]
  },
  {
    'Id': 35228,
    'Name': 'Milly Smith',
    'Scores': [
      92.5,
      88.0,
      85.7
    ]
  }
]";

List<Student> studentsList = JsonConvert.DeserializeObject<List<Student>>(jsonStudentList);
Console.WriteLine(studentsList.Count);
//2
Student s = studentsList[0];
Console.WriteLine(s.Name);
//Jim David
```

如果 Json 对象拥有统一类型的属性和属性值，我们还可以把 Json 字符串反序列化为 .NET 中的字典，Json 对象的“属性”和“属性值”会依次赋值给字典中的 Key 和 Value。下面我举一个简单的例子：

```csharp
string json = @"{""English"":88.2,""Math"":96.9}";
Dictionary<string, double> values = JsonConvert.DeserializeObject<Dictionary<string, double>>(json);
Console.WriteLine(values.Count);
//2
Console.WriteLine(values["Math"]);
//96.9
```

## 解析复杂的 Json 字符串

如今大量的 Web API 为我们编写复杂程序提供了极大的方便，例如[百度地图 API](http://lbsyun.baidu.com/index.php?title=webapi)、[图灵机器人 API](http://www.tuling123.com/) 等等，利用这些 Web 应用程序我们可以充分发挥云服务的优势，开发出大量有趣的应用。

Web API 通常返回 Json 或 XML 格式的检索数据，由于 Json 数据量更小，所以目前大多数情况下我们都选择返回 Json 格式的数据。

如果返回的 Json 文档很大，而我们仅仅需要其中的一小部分数据。按照之前的方法，我们必须首先定义一个与 Json 对象对应的 .NET 对象，然后反序列化，最后才能从对象中取出我们需要的数据。而有了 Json.NET，这个任务就很容易实现了，我们可以局部地解析一个 Json 对象。

下面以获取 Google 搜索结果为例，简单演示一下对复杂结构 Json 文档的解析。

```csharp
string googleSearchText = @"{
  'responseData': {
    'results': [
      {
        'GsearchResultClass': 'GwebSearch',
        'unescapedUrl': 'http://en.wikipedia.org/wiki/Paris_Hilton',
        'url': 'http://en.wikipedia.org/wiki/Paris_Hilton',
        'visibleUrl': 'en.wikipedia.org',
        'cacheUrl': 'http://www.google.com/search?q=cache:TwrPfhd22hYJ:en.wikipedia.org',
        'title': '<b>Paris Hilton</b> - Wikipedia, the free encyclopedia',
        'titleNoFormatting': 'Paris Hilton - Wikipedia, the free encyclopedia',
        'content': '[1] In 2006, she released her debut album...'
      },
      {
        'GsearchResultClass': 'GwebSearch',
        'unescapedUrl': 'http://www.imdb.com/name/nm0385296/',
        'url': 'http://www.imdb.com/name/nm0385296/',
        'visibleUrl': 'www.imdb.com',
        'cacheUrl': 'http://www.google.com/search?q=cache:1i34KkqnsooJ:www.imdb.com',
        'title': '<b>Paris Hilton</b>',
        'titleNoFormatting': 'Paris Hilton',
        'content': 'Self: Zoolander. Socialite <b>Paris Hilton</b>...'
      }
    ],
    'cursor': {
      'pages': [
        {
          'start': '0',
          'label': 1
        },
        {
          'start': '4',
          'label': 2
        },
        {
          'start': '8',
          'label': 3
        },
        {
          'start': '12',
          'label': 4
        }
      ],
      'estimatedResultCount': '59600000',
      'currentPageIndex': 0,
      'moreResultsUrl': 'http://www.google.com/search?oe=utf8&ie=utf8...'
    }
  },
  'responseDetails': null,
  'responseStatus': 200
}";
```

上面就是从 Google 搜索返回的 Json 数据，我们现在需要的是 **responseData** 属性下的 **results** 列表中结果，而且仅仅需要结果中的 `title`、`content` 和 `url` 属性值。

```csharp
public class SearchResult
{
	public string Title { get; set; }
	public string Content { get; set; }
	public string Url { get; set; }
}
```

```csharp
//将 Json 文档解析为 JObject
JObject googleSearch = JObject.Parse(googleSearchText);
//将获得的 Json 结果转换为列表
IList<JToken> results = googleSearch["responseData"]["results"].Children().ToList();
//将 Json 结果反序列化为 .NET 对象
IList<SearchResult> searchResults = new List<SearchResult>();
foreach(JToken result in results)
{
	SearchResult searchResult = JsonConvert.DeserializeObject<SearchResult>(result.ToString());
	searchResults.Add(searchResult);
}
// Title = <b>Paris Hilton</b> - Wikipedia, the free encyclopedia
// Content = [1] In 2006, she released her debut album...
// Url = http://en.wikipedia.org/wiki/Paris_Hilton

// Title = <b>Paris Hilton</b>
// Content = Self: Zoolander. Socialite <b>Paris Hilton</b>...
// Url = http://www.imdb.com/name/nm0385296/
```

可以看到，对 Json 文档的解析基本分为以下几步：

1. 将 Json 文档转换为 JObject 对象
2. 使用`JObject[属性]`获取 JObject 对象中某个属性的值（JToken 格式）  
   可以继续通过 `JToken[属性]` 获取属性内部的属性值（依然为 JToken 格式）
3. 将 JToken 格式的属性值反序列化为 .NET 对象

如果属性值为我们需要的数据对象，那么直接反序列化该对象就可以了；如果属性值为列表（比如上面 **results** 属性的值），那么就可以调用 JToken 类的 **Children()** 函数，获得一个可迭代的 JEnumerable\<JToken> 对象（用于迭代访问列表中的每一个对象），最后再依次反序列化列表中的对象。

## 参考

- [Json.NET documentation](http://www.newtonsoft.com/json/help)
- [c# 解析 JSON 的几种办法](http://www.cnblogs.com/ambar/archive/2010/07/13/parse-json-via-csharp.html)