---
layout: article
title: MongoDB 权限管理：裸奔很快乐，后果很严重
tags:
    - MongoDB
    - 数据库
mathjax: false
---

## 前言

最近部署在阿里云上的 MongoDB 数据库被人黑了，因为这个项目只是一个 Demo，所以当时就偷懒直接使用了 MongoDB 默认的配置，既没有设置数据库的访问权限，也没有修改访问的端口，处于门户大开的全裸奔状态。虽然从 2016 年底到 2017 年初，有大量 MongoDB 数据库都因为未配置安全权限被黑，但抱着自己这个小项目应该没人感兴趣的侥幸心理，一直都没有补上这个漏洞，没想到在项目演示时才发现数据被清空了……

攻击者删除了所有的数据，只留下了下面勒索比特币的信息：

```json
{ 
    "_id" : ObjectId("594658f3c108c36b5cecf868"), 
    "email" : "request@tfwno.gf", 
    "btc_wallet" : "1JjNwP7kXyvP38xbFhxbmebHAZFvnfXHt8", 
    "note" : "Your DB is in safety and backed up (check logs). To restore send 0.15 BTC and email with your server ip or domain name. Each 24 hours we erase all data."
}
```

不过幸好数据量不大，而且我之前做了备份，否则这些数据估计就找不回来了。这件事也给了我一个教训，在信息安全上永远不要抱侥幸的心理，毕竟网络中有太多别有居心的人了。

## 用户角色

### 创建用户管理员账户

要能够管理数据库中的用户，需要一个具有 grant 权限的账户，即：账号管理的授权权限。我们先以不开启权限认证的方式启动数据库（auth=false），以创建这个账户：

```shell
> use admin
switched to db admin
> db.createUser(
...   {
...     user: "dba",
...     pwd: "dba",
...     roles: [ { role: "userAdminAnyDatabase", db: "admin" } ]
...   }
... )
Successfully added user: {
    "user" : "dba",
    "roles" : [
        {
            "role" : "userAdminAnyDatabase",
            "db" : "admin"
        }
    ]
}
```

MongoDB 中使用 `db.createUser` 命令添加用户，其中 `user` 表示用户名，`pwd` 表示密码，`roles` 表示角色。注意，MongoDB 中帐号是跟着库走的，所以在指定库里授权，必须也在指定库里验证(auth)，后面会进一步解释。`roles` 参数是一个数组，可以指定内置角色和用户定义的角色，也可以用一个空数组给新用户设定空角色。`roles` 中可以指定的角色有：

- **数据库用户角色：**read、readWrite
- **数据库管理角色：**dbAdmin、dbOwner、userAdmin
- **集群管理角色：**clusterAdmin、clusterManager、clusterMonitor、hostManager
- **备份恢复角色：**backup、restore
- **所有数据库角色：**readAnyDatabase、readWriteAnyDatabase、userAdminAnyDatabase、dbAdminAnyDatabase
- **超级用户角色：**root
- **内部角色：**__system

> 还有几个角色间接或直接提供了系统超级用户的访问 dbOwner 、userAdmin、userAdminAnyDatabase

具体角色的说明为：

- read：允许用户读取指定数据库
- readWrite：允许用户读写指定数据库
- dbAdmin：允许用户在指定数据库中执行管理函数，如索引创建、删除，查看统计或访问 system.profile
- userAdmin：允许用户向 system.users 集合写入，可以找指定数据库里创建、删除和管理用户
- clusterAdmin：只在 admin 数据库中可用，赋予用户所有分片和复制集相关函数的管理权限
- readAnyDatabase：只在 admin 数据库中可用，赋予用户所有数据库的读权限
- readWriteAnyDatabase：只在 admin 数据库中可用，赋予用户所有数据库的读写权限
- userAdminAnyDatabase：只在 admin 数据库中可用，赋予用户所有数据库的 userAdmin 权限
- dbAdminAnyDatabase：只在 admin 数据库中可用，赋予用户所有数据库的 dbAdmin 权限
- root：只在 admin 数据库中可用。超级账号，超级权限

我们通过上面的命令，创建了一个名称为 dba 的 userAdminAnyDatabase 用户，他可以在所有的数据中创建和删除用户。接下来，我们重新启动数据库服务，并且将 auth 参数设置为 true 以启动权限认证。

```shell
> show dbs;    #没有验证，导致没权限。
2015-06-29T10:02:16.634-0400 E QUERY    Error: listDatabases failed:{
    "ok" : 0,
    "errmsg" : "not authorized on admin to execute command { listDatabases: 1.0 }",
    "code" : 13
}
    at Error (<anonymous>)
    at Mongo.getDBs (src/mongo/shell/mongo.js:47:15)
    at shellHelper.show (src/mongo/shell/utils.js:630:33)
    at shellHelper (src/mongo/shell/utils.js:524:36)
    at (shellhelp2):1:1 at src/mongo/shell/mongo.js:47
> use admin    #验证，因为在 admin 下面添加的帐号，所以要到 admin 下面验证。
switched to db admin
> db.auth('dba','dba')
1
> show dbs;
admin  0.078GB
local  0.078GB
```

可以看到，启动权限认证后的数据库不再能够随意的访问了，甚至连 `show dbs` 查看数据库列表都无法执行。而切换到之前创建的 dba 用户后，命令就可以顺利执行了。因为 dba 是在 admin 数据库中建立的，所以也必须到 admin 数据库中认证，这就是 MongoDB 中“数据库帐号跟着数据库走，哪里创建哪里认证”的机制。

### 创建指定数据库的角色

现在我们需要给 test 数据库创建两个角色，一个拥有 read 权限，只能读取数据库中数据；另一个拥有 readWrite 权限，负责维护数据库。一般情况，要创建用户的角色在哪个数据库上生效，就在该数据库上创建用户：

```shell
> use admin
switched to db admin
> db.auth('dba','dba')
1
> use test
switched to db test
> db.createUser(
...     {
...       user: "testr",
...       pwd: "testr",
...       roles: [
...          { role: "read", db: "test" }    #只读帐号
...       ]
...     }
... )
Successfully added user: {
    "user" : "testr",
    "roles" : [
        {
            "role" : "read",
            "db" : "test"
        }
    ]
}
> db.createUser(
...     {
...       user: "testrw",
...       pwd: "testrw",
...       roles: [
...          { role: "readWrite", db: "test" }   #读写帐号
...       ]
...     }
... )
Successfully added user: {
    "user" : "testrw",
    "roles" : [
        {
            "role" : "readWrite",                #读写账号
            "db" : "test"
        }
    ]
}
> show users    #查看当前库下的用户
{
    "_id" : "test.testr",
    "user" : "testr",
    "db" : "test",
    "roles" : [
        {
            "role" : "read",
            "db" : "test"
        }
    ]
}
{
    "_id" : "test.testrw",
    "user" : "testrw",
    "db" : "test",
    "roles" : [
        {
            "role" : "readWrite",
            "db" : "test"
        }
    ]
}
```

通过上面的命令，我们为 test 数据库创建了两个账号：拥有 read 只读权限的 testr 用户和拥有 readWrite 读写权限的 testrw 用户。接下来，我们在 test 数据库上测试一下这两个用户：

```shell
> use test
switched to db test
> db.abc.insert({"a":1,"b":2})    #插入失败，没有权限，userAdminAnyDatabase 权限只是针对用户管理的，对其他是没有权限的。
WriteResult({
    "writeError" : {
        "code" : 13,
        "errmsg" : "not authorized on test to execute command { insert: \"abc\", documents: [ { _id: ObjectId('55915185d629831d887ce2cb'), a: 1.0, b: 2.0 } ], ordered: true }"
    }
})
> db.auth('testrw','testrw')    #用创建的 readWrite 帐号进行写入
1
> db.abc.insert({"a":1,"b":2})
> db.abc.find()
{ "_id" : ObjectId("559151a1b78649ebd8316853"), "a" : 1, "b" : 2 }
> db.auth('testr','testr')    #切换到只有 read 权限的帐号
1
> db.abc.insert({"a":11,"b":22})    #不能写入
WriteResult({
    "writeError" : {
        "code" : 13,
        "errmsg" : "not authorized on test to execute command { insert: \"abc\", documents: [ { _id: ObjectId('559151ebb78649ebd8316854'), a: 11.0, b: 22.0 } ], ordered: true }"
    }
})
> db.abc.find()    #可以查看
{ "_id" : ObjectId("559151a1b78649ebd8316853"), "a" : 1, "b" : 2 }
```

可以看到，拥有读写权限的 testrw 用户，可以往表中添加数据；而只拥有读权限的 testr 用户，则只能读取数据，无法写入数据。

### root 超级用户

通过上面的例子，我们一定会想有没有一个拥有超级权限的用户？他不仅可以授权，而且也可以对任意数据库下的集合进行任意操作？答案是肯定的，但是不建议使用。那就是将 role 角色设置成 root：

```shell
> use admin
switched to db admin
> db.auth('dba','dba')
1
> db.createUser(
...  {
...    user: "super",
...    pwd: "super",
...    roles: [
...       { role: "root", db: "admin" }    #超级root帐号
...    ]
...  }
... )
Successfully added user: {
    "user" : "super",
    "roles" : [
        {
            "role" : "root",
            "db" : "admin"
        }
    ]
}
> show users    #查看当前库下的用户
{
    "_id" : "admin.dba",
    "user" : "dba",
    "db" : "admin",
    "roles" : [
        {
            "role" : "userAdminAnyDatabase",
            "db" : "admin"
        }
    ]
}
{
    "_id" : "admin.super",
    "user" : "super",
    "db" : "admin",
    "roles" : [
        {
            "role" : "root",
            "db" : "admin"
        }
    ]
}
> db.auth('super','super')
1
> use test
switched to db test
> db.abc.insert({"a":11,"b":22})
WriteResult({ "nInserted" : 1 })
> db.abc.find()
{ "_id" : ObjectId("559151a1b78649ebd8316853"), "a" : 1, "b" : 2 }
{ "_id" : ObjectId("559153a0b78649ebd8316854"), "a" : 11, "b" : 22 }
> db.abc.remove({})
WriteResult({ "nRemoved" : 2 })
```

可以看到，拥有 root 角色的用户可以对任意数据库中的集合进行修改，拥有非常大的权限。因而 root 角色不建议使用，在实际的操作中很少被使用。

## MongoDB 账户

### 哪里创建哪里认证

正如我们之前所说，MongoDB 中帐号是跟着库走的，所以在指定库里授权，必须也在指定库里验证(auth)。一般情况，要创建用户的角色在哪个数据库上生效，就在该数据库上创建用户。不过我们也可以在一个数据库下创建对应其他数据库中角色的用户：

```shell
> db
admin
> db.createUser(
...  {
...    user: "haha",
...    pwd: "haha",
...    roles: [
...       { role: "readWrite", db: "test" },    #在 admin 库下创建 test、abc 库的帐号
...       { role: "readWrite", db: "abc" }         
...    ]
...  }
... )
Successfully added user: {
    "user" : "haha",
    "roles" : [
        {
            "role" : "readWrite",
            "db" : "test"
        },
        {
            "role" : "readWrite",
            "db" : "abc"
        }
    ]
}
> show users
{
    "_id" : "admin.dba",
    "user" : "dba",
    "db" : "admin",
    "roles" : [
        {
            "role" : "userAdminAnyDatabase",
            "db" : "admin"
        }
    ]
}
{
    "_id" : "admin.super",
    "user" : "super",
    "db" : "admin",
    "roles" : [
        {
            "role" : "root",
            "db" : "admin"
        }
    ]
}
{
    "_id" : "admin.haha",
    "user" : "haha",
    "db" : "admin",
    "roles" : [
        {
            "role" : "readWrite",
            "db" : "test"
        },
        {
            "role" : "readWrite",
            "db" : "abc"
        }
    ]
}
> use test
switched to db test
> db.auth('haha','haha')    #在 admin 下创建的帐号，不能直接在其他库验证，
Error: 18 Authentication failed.
0
> use admin
switched to db admin    #只能在帐号创建库下认证，再去其他库进行操作。
> db.auth('haha','haha')
1
> use test
switched to db test
> db.abc.insert({"a":1111,"b":2222})
WriteResult({ "nInserted" : 1 })
> use abc
switched to db abc
> db.abc.insert({"a":1111,"b":2222})
WriteResult({ "nInserted" : 1 })
```

可以看到，在 admin 数据库中创建的用户 haha，虽然设置的权限是对应 test 和 abc 库，但也必须先在 admin 库中认证后，才能在角色对应的库中操作。这更加进一步说明了数据库帐号是跟着数据库走的，哪里创建哪里认证。

### 查看所有帐号

之前的操作中，我们都只能查看当前数据库下的用户，如果想要查看当前所有数据库下的用户，可以使用 `db.system.users.find().pretty()` 命令，当然执行这条命令的用户需要具有用户管理员权限：

```shell
>  use admin
switched to db admin
> db.auth('dba','dba')
1
> db.system.users.find().pretty()
{
    "_id" : "admin.dba",
    "user" : "dba",
    "db" : "admin",
    "credentials" : {
        "SCRAM-SHA-1" : {
            "iterationCount" : 10000,
            "salt" : "KfDUzCOIUo7WVjFr64ZOcQ==",
            "storedKey" : "t4sPsKG2dXnZztVYj5EgdUzT9sc=",
            "serverKey" : "2vCGiq9NIc1zKqeEL6VvO4rP26A="
        }
    },
    "roles" : [
        {
            "role" : "userAdminAnyDatabase",
            "db" : "admin"
        }
    ]
}
{
    "_id" : "test.testr",
    "user" : "testr",
    "db" : "test",
    "credentials" : {
        "SCRAM-SHA-1" : {
            "iterationCount" : 10000,
            "salt" : "h1gOW3J7wzJuTqgmmQgJKQ==",
            "storedKey" : "7lkoANdxM2py0qiDBzFaZYPp1cM=",
            "serverKey" : "Qyu6IRNyaKLUvqJ2CAa/tQYY36c="
        }
    },
    "roles" : [
        {
            "role" : "read",
            "db" : "test"
        }
    ]
}
{
    "_id" : "test.testrw",
    "user" : "testrw",
    "db" : "test",
    "credentials" : {
        "SCRAM-SHA-1" : {
            "iterationCount" : 10000,
            "salt" : "afwaKuTYPWwbDBduQ4Hm7g==",
            "storedKey" : "ebb2LYLn4hiOVlZqgrAKBdStfn8=",
            "serverKey" : "LG2qWwuuV+FNMmr9lWs+Rb3DIhQ="
        }
    },
    "roles" : [
        {
            "role" : "readWrite",
            "db" : "test"
        }
    ]
}
{
    "_id" : "admin.super",
    "user" : "super",
    "db" : "admin",
    "credentials" : {
        "SCRAM-SHA-1" : {
            "iterationCount" : 10000,
            "salt" : "pE2cSOYtBOYevk8tqrwbSQ==",
            "storedKey" : "TwMxdnlB5Eiaqg4tNh9ByNuUp9A=",
            "serverKey" : "Mofr9ohVlFfR6/md4LMRkOhXouc="
        }
    },
    "roles" : [
        {
            "role" : "root",
            "db" : "admin"
        }
    ]
}
{
    "_id" : "admin.haha",
    "user" : "haha",
    "db" : "admin",
    "credentials" : {
        "SCRAM-SHA-1" : {
            "iterationCount" : 10000,
            "salt" : "XD6smcWX4tdg/ZJPoLxxRg==",
            "storedKey" : "F4uiayykHDp/r9krAKZjdr+gqjM=",
            "serverKey" : "Kf51IU9J3RIrB8CFn5Z5hEKMSkw="
        }
    },
    "roles" : [
        {
            "role" : "readWrite",
            "db" : "test"
        },
        {
            "role" : "readWrite",
            "db" : "abc"
        }
    ]
}
> db.system.users.find().count()
5
```

### 备份还原权限

在 MongoDB 中，数据的备份与还原是常用的操作，那么执行这些操作需要怎样的用户权限呢？

```shell
root@zhoujinyi:~# mongodump --port=27020 -uzjyr -pzjyr --db=test -o backup    #只要读权限就可以备份
2015-06-29T11:20:04.864-0400    writing test.abc to backup/test/abc.bson
2015-06-29T11:20:04.865-0400    writing test.abc metadata to backup/test/abc.metadata.json
2015-06-29T11:20:04.866-0400    done dumping test.abc
2015-06-29T11:20:04.867-0400    writing test.system.indexes to backup/test/system.indexes.bson


root@zhoujinyi:~# mongorestore --port=27020 -uzjy -pzjy --db=test backup/test/    #读写权限可以进行还原
2015-06-29T11:20:26.607-0400    building a list of collections to restore from backup/test/ dir
2015-06-29T11:20:26.609-0400    reading metadata file from backup/test/abc.metadata.json
2015-06-29T11:20:26.609-0400    restoring test.abc from file backup/test/abc.bson
2015-06-29T11:20:26.611-0400    error: E11000 duplicate key error index: test.abc.$_id_ dup key: { : ObjectId('559154efb78649ebd831685a') }
2015-06-29T11:20:26.611-0400    restoring indexes for collection test.abc from metadata
2015-06-29T11:20:26.612-0400    finished restoring test.abc
2015-06-29T11:20:26.612-0400    done
```

可以看到，对于备份只需要读权限就可以了，而还原自然是需要读写权限的。

## 使用用户名密码连接 MongoDB

在大部分的情况下，我们都不会直接使用 MongoDB 的终端去连接数据库，而是在程序中使用 MongoDB 的驱动与数据库进行交互。下面就是 Java 和 Python 与数据库建立带权限认证的连接的方式：

### Java

```java
MongoClient mongoClient;
MongoCredential credential = MongoCredential.createCredential("user", "database", password);
mongoClient = new MongoClient(new ServerAddress(ip, port), Arrays.asList(credential));
MongoDatabase database = mongoClient.getDatabase("test");
MongoCollection<Document> collection = database.getCollection("abc");
Document myDoc = collection.find().first();
System.out.println(myDoc.toJson());
```

### Python

```python
from pymongo import MongoClient

client = MongoClient("ip",port)
client.database.authenticate("user","password")
db = client.database
collection = db.collection
```

## 参考

[《MongoDB 3.X 用户权限控制》](http://www.cnblogs.com/shiyiwen/p/5552750.html)  

[《用户名、密码 连接mongodb数据库》](http://blog.chinaunix.net/uid-13869856-id-5074868.html) 