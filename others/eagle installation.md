# Apache eagle installing

## 介绍

Apache eagle 目前0.4.0孵化版本已经可用，而新的0.5.0预计在2016年11月份发布。它具备如下几个功能：
- 监控Hadoop中的数据访问流量
- 检测非法入侵和违反安全规则的行为
- 检测并防止敏感数据丢失和访问
- 实现基于策略的实时检测和预警
- 实现基于用户行为模式的异常数据行为检测

个人使用下来，主要有如下用处：
- 监控敏感数据或者目录，监控操作次数
- 监控hive表字段操作
- 监控oozie coodinator任务情况
- 监控警告可以是eagle自身存储、mail、kafka

而对于个人行为模式异常检测，即userpofile功能，建议忽略，且0.5.0版本将剔除此模块。

由于是开源的孵化版本，搭建起来会遇到很多问题，如下在安装过程中我会将遇到的问题一一列举并解决。

如果你需要安装，建议通读整个文章后再进行编译安装。

## 搭建

#### 准备


参见[官方网站](http://eagle.apache.org/docs/deployment-env.html)，
For streaming platform dependencies

- Storm: 0.9.3 or later
- Kafka: 0.8.x or later
- Java: 1.7.x
- NPM (On MAC OS try “brew install node”)
- logstash(我使用2.3.4版本)
- maven

For database dependencies (Choose one of them)

- HBase: 0.98 or later (Hadoop5: 2.6.x is required)
- Mysql (Installation is required)
- Derby (No installation)

注意：
- 建议使用mysql做元数据管理，0.4.0版本使用hbase会有异常，如下会有阐述。
- 官方eagle有大量篇幅使用了hontornworks的Ambari和HDP，依赖很大，其实cloudera CDH版本hadoop也是可以的。
- 建议安装logstash来采集数据写入至kafka，通过修改hdfs audit log4j KAFKA_HDFS_AUDIT方式，一对hadoop有倾入性，二我测试没通过，KafkaLog4jAppender一直找不到类。
``` java
java.lang.NoClassDefFoundError: com/yammer/metrics/Metrics
        at kafka.metrics.KafkaMetricsGroup$class.newMeter(KafkaMetricsGroup.scala:79)
        at kafka.producer.ProducerStats.newMeter(ProducerStats.scala:23)
        at kafka.producer.ProducerStats.<init>(ProducerStats.scala:25)
        at kafka.producer.ProducerStatsRegistry$$anonfun$1.apply(ProducerStats.scala:34)
        at kafka.producer.ProducerStatsRegistry$$anonfun$1.apply(ProducerStats.scala:34)
        at kafka.utils.Pool.getAndMaybePut(Pool.scala:61)
        at kafka.producer.ProducerStatsRegistry$.getProducerStats(ProducerStats.scala:38)
        at kafka.producer.async.DefaultEventHandler.<init>(DefaultEventHandler.scala:48)
        at kafka.producer.Producer.<init>(Producer.scala:60)
        at org.apache.eagle.log4j.kafka.KafkaLog4jAppender.activateOptions(KafkaLog4jAppender.scala:113)
        at org.apache.log4j.config.PropertySetter.activate(PropertySetter.java:307)
```

#### 安装

参见[官方网站](http://eagle.apache.org/docs/quick-start.html)。
###### 编译
编译时如果遇到：
```
[INFO] eagle-webservice ................................... FAILURE [03:03 min]
Failed to execute goal org.codehaus.mojo:exec-maven-plugin:1.5.0:exec (exec-ui-install) on project eagle-webservice: Command execution failed. Process exited with an error: 1 (Exit value: 1)
```
表示NPM没有安装：
```
sudo apt-get install NPM
#或者
sudo yum install NPM
```

###### 运行&配置

安装官方文档运行examples/eagle-sandbox-starter.sh，一般你会得到大堆错误的。
修改元数据存储，eagle-service.conf，mysql：
```
eagle {
        service {
                storage-type="jdbc"
                storage-adapter="mysql"
                storage-username="eagle"
                storage-password=eagle
                storage-database=eagle
                storage-connection-url="jdbc:mysql://cs01/eagle"
                storage-connection-props="encoding=UTF-8"
                storage-driver-class="com.mysql.jdbc.Driver"
                storage-connection-max=8
        }
}
```
hbase：
```
eagle {
        service {
                storage-type="hbase"
                hbase-zookeeper-quorum="cs02,cs03,cs04"
                hbase-zookeeper-property-clientPort=2181
                zookeeper-znode-parent="/hbase"
        }
}
```
执行bin/eagle-topology-init.sh。

当配置mysql时，eagle能正常启动bin/eagle-service.sh start，但在运行hdfsaudit时报错，缺少一些表，即使按照github上创建所有mysql表也会缺表，在元数据中增加如下表：
``` sql
create table alertnotifications_alertnotifications(
`uuid` varchar(254) COLLATE utf8_bin NOT NULL,
`timestamp` bigint(20) DEFAULT NULL,
`notificationType` varchar(30000),
`enabled` tinyint(1) DEFAULT NULL,
`description` mediumtext,
`className` mediumtext,
`fields` mediumtext,
PRIMARY KEY (`uuid`),
UNIQUE KEY `uuid_UNIQUE` (`uuid`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

create table eagle_metadata_topologydescription(
`uuid` varchar(254) COLLATE utf8_bin NOT NULL,
`timestamp` bigint(20) DEFAULT NULL,
`topology` varchar(30000),
`description` mediumtext,
`exeClass` mediumtext,
`type` mediumtext,
`version` mediumtext,
PRIMARY KEY (`uuid`),
UNIQUE KEY `uuid_UNIQUE` (`uuid`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

create table eagle_metadata_topologyexecution(
`uuid` varchar(254) COLLATE utf8_bin NOT NULL,
`timestamp` bigint(20) DEFAULT NULL,
site varchar(1024),
application varchar(1024),
topology varchar(1024),
environment varchar(1024),
status varchar(1024),
description varchar(1024),
lastmodifieddate bigint(20),
fullname varchar(1024),
url varchar(1024),
mode varchar(1024),
PRIMARY KEY (`uuid`),
UNIQUE KEY `uuid_UNIQUE` (`uuid`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

create table eagle_metric_dmeta(
`uuid` varchar(254) COLLATE utf8_bin NOT NULL,
`timestamp` bigint(20) DEFAULT NULL,
drillDownPaths mediumtext,
aggFunctions mediumtext,
defaultDownSamplingFunction mediumtext,
defaultAggregateFunction mediumtext,
resolutions mediumtext,
downSamplingFunctions mediumtext,
storeType mediumtext,
displayName mediumtext,
PRIMARY KEY (`uuid`),
UNIQUE KEY `uuid_UNIQUE` (`uuid`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
```

当配置为hbase时，eagle也能正常启动，但运行过程中会报如下异常(未解决)：
```
Exception:
java.io.IOException: java.io.IOException: java.lang.NoSuchMethodError: com.google.protobuf.LazyStringList.getUnmodifiableView()Lcom/google/protobuf/LazyStringList;
org.apache.eagle.storage.hbase.HBaseStorage.query(HBaseStorage.java:210)
org.apache.eagle.storage.operation.QueryStatement.execute(QueryStatement.java:47)
org.apache.eagle.service.generic.GenericEntityServiceResource.search(GenericEntityServiceResource.java:443)
sun.reflect.GeneratedMethodAccessor59.invoke(Unknown Source)
sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
java.lang.reflect.Method.invoke(Method.java:606)
com.sun.jersey.spi.container.JavaMethodInvokerFactory$1.invoke(JavaMethodInvokerFactory.java:60)
com.sun.jersey.server.impl.model.method.dispatch.AbstractResourceMethodDispatchProvider$TypeOutInvoker._dispatch(AbstractResourceMethodDispatchProvider.java:185)
com.sun.jersey.server.impl.model.method.dispatch.ResourceJavaMethodDispatcher.dispatch(ResourceJavaMethodDispatcher.java:75)
com.sun.jersey.server.impl.uri.rules.HttpMethodRule.accept(HttpMethodRule.java:288)


2016-08-24 18:57:44,378 WARN [http-bio-9099-exec-10] client.HTable[1661]: Error calling coprocessor service org.apache.eagle.storage.hbase.query.coprocessor.generated.AggregateProtos$AggregateProtocol for row \x8F\xE4h\xBB\x7F\xFF\xFE\xA9A\x00\xF7\xFF\xFF\xFF\xFF\xFF\xFF
java.util.concurrent.ExecutionException: java.lang.NoSuchMethodError: com.google.protobuf.LazyStringList.getUnmodifiableView()Lcom/google/protobuf/LazyStringList;
        at java.util.concurrent.FutureTask.report(FutureTask.java:122)
        at java.util.concurrent.FutureTask.get(FutureTask.java:188)
        at org.apache.hadoop.hbase.client.HTable.coprocessorService(HTable.java:1659)
        at org.apache.hadoop.hbase.client.HTablePool$PooledHTable.coprocessorService(HTablePool.java:533)
```
所以建议安装mysql元数据服务。

#### 配置hdfsaudit

hdfsaudit应用默认使用KafkaLog4jAppender读取hdfs-audit日志数据，并写入kafka的sanbox_hdfs_audit_log主题，然后strom消费这个topic数据，根据用户配置的监控策略，进行警告。

###### 安装logstash
安装logstash，运行命令：./logstash -f ../conf/hdfs-audit.conf ，hdfs-audit.conf(可以参考[tream hdfs log data into Kafka](http://eagle.apache.org/docs/import-hdfs-auditLog.html)，logstash方式)：
```
input {
        file {
                type => "cdh-nn-audit"
                path => "/var/log/hadoop-hdfs/hdfs-audit.log"
                start_position => end
                sincedb_path => "/var/log/logstash"
         }
}

filter{
        if [type] == "cdh-nn-audit" {
           grok {
                   match => ["message", "ugi=(?<user>([\w\d\-]+))@|ugi=(?<user>([\w\d\-]+))/[\w\d\-.]+@|ugi=(?<user>([\w\d.\-_]+))[\s(]+"]
           }
        }
}

output {
        if [type] == "cdh-nn-audit" {
                kafka {
                        codec => plain {
                                format => "%{message}"
                        }
                        bootstrap_servers => "cs03:9092"
                        topic_id => "hdfs_audit_log"
                        timeout_ms => 10000
                        retries => 3
                        client_id => "cdh-nn-audit"
                }
                # stdout { codec => rubydebug }
        }
}
```

###### 安装storm

strom 0.9.3，安装步骤省略。
storm.yaml 配置：
```
storm.zookeeper.servers:
     - "cs02"
     - "cs03"
     - "cs04"

nimbus.thrift.max_buffer_size: 20480000
nimbus.host: "cs01"
```
启动niumbis ui和supervisor。

###### 安装kafka
安装步骤省略，我使用cloudera manager安装的1.2.1版本。
kafka相关命令：

kafka test topic数据：
```
kafka-console-consumer --zookeeper cs03:2181 --topic hdfs_audit_log
```
查看kafka offset ：
```
zookeeper-client -server cs02:2181
get /consumers/console-consumer-51536/offsets/hdfs_audit_log/0
```

###### 配置
配置apache-eagle-0.4.0-incubating/bin/eagle-env.sh:
```
export JAVA_HOME=/usr/java/jdk1.7.0_67-cloudera

# nimbus.host, default is localhost
export EAGLE_NIMBUS_HOST=cs01

# EAGLE_SERVICE_HOST, default is `hostname -f`
export EAGLE_SERVICE_HOST=cs01

# EAGLE_SERVICE_PORT, default is 9099
export EAGLE_SERVICE_PORT=9099

# EAGLE_SERVICE_USER
export EAGLE_SERVICE_USER=admin

# EAGLE_SERVICE_PASSWORD
export EAGLE_SERVICE_PASSWD=secret
```
配置apache-eagle-0.4.0-incubating/conf/eagle-scheduler.conf：
```
appCommandLoaderEnabled = false
appCommandLoaderIntervalSecs = 1
appHealthCheckIntervalSecs = 5

### execution platform properties
envContextConfig.env = "storm"
envContextConfig.url = "http://cs01:8080"
envContextConfig.nimbusHost = "cs01"
envContextConfig.nimbusThriftPort = 6627
envContextConfig.jarFile = "/data1/apache-eagle-0.4.0-incubating/lib/topology/eagle-topology-0.4.0-incubating-assembly.jar"

### default topology properties
eagleProps.mailHost = "mailHost.com"
eagleProps.mailSmtpPort = "25"
eagleProps.mailDebug = "true"
eagleProps.eagleService.host = "cs01"
eagleProps.eagleService.port = 9099
eagleProps.eagleService.username = "admin"
eagleProps.eagleService.password = "secret"
eagleProps.dataJoinPollIntervalSec = 30

dynamicConfigSource.enabled = true
dynamicConfigSource.initDelayMillis = 0
dynamicConfigSource.delayMillis = 30000
```
配置： apache-eagle-0.4.0-incubating/conf/sandbox-hdfsAuditLog-application.conf
```
{
  "envContextConfig" : {
    "env" : "storm",
    "mode" : "cluster",
    "topologyName" : "sandbox-hdfsAuditLog-topology",
    "stormConfigFile" : "security-auditlog-storm.yaml",
    "parallelismConfig" : {
      "kafkaMsgConsumer" : 1,
      "hdfsAuditLogAlertExecutor*" : 1
    }
  },
  "dataSourceConfig": {
    "topic" : "hdfs_audit_log",
    "zkConnection" : "cs02:2181,cs03:2181,cs04:2181",
    "brokerZkPath" : "/brokers",
    "zkConnectionTimeoutMS" : 15000,
    "fetchSize" : 1048586,
    "deserializerClass" : "org.apache.eagle.security.auditlog.HdfsAuditLogKafkaDeserializer",
    "transactionZKServers" : "cs02,cs03,cs04",
    "transactionZKPort" : 2181,
    "transactionZKRoot" : "/consumers",
    "consumerGroupId" : "eagle.hdfsaudit.consumer",
    "transactionStateUpdateMS" : 2000
  },
  ...
}
```
重点是topic要与logstash写入的topic一致。

###### 启动hdfsaudit
之前已经执行eagle-topology-init.sh初始化了，则先启动eagle：
```
bin/eagle-service.sh start
```
然后启动storm topology：
```
bin/eagle-topology.sh start
```
默认启动的就是org.apache.eagle.security.auditlog.HdfsAuditLogProcessorMain。storm ui界面应该能看到提交的topology。

###### 遇到问题
1. 如果你的hadoop namenode启用了balance，两个节点都需要安装logstash并运行。且得在eagle的admin->指定stie->hdfsAuditLog->Configuration加入：
```
classification.fs.defaultFS=hdfs://nameservice1
classification.dfs.nameservices=nameservice1
classification.dfs.ha.namenodes.nameservice1=namenode32,namenode70
classification.dfs.namenode.rpc-address.nameservice1.namenode32=cs01:8020
classification.dfs.namenode.rpc-address.nameservice1.namenode70=cs02:8020
classification.dfs.client.failover.proxy.provider.nameservice1=org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider
```
界面验证不通过，报：Invalid Properties format 
修lib/tomcat/webapps/eagle-service/ui/public/js/doc.js里的common.properties.check 方法
```
var regex = /^\s*[\w\.]+\s*=(.*)$/; 
##修改成
var regex = /^\s*[\w\.\-]+\s*=(.*)$/; 
```
然后点击保存。

2. storm报错，查看响应storm节点
```
java.lang.RuntimeException: java.util.regex.PatternSyntaxException: Illegal repetition near index 1 ./{} ^ 
at backtype.storm.utils.DisruptorQueue.consumeBatchToCursor(DisruptorQueue.java:128) 
at backtype.storm.utils.DisruptorQueue.consumeBatchWhenAvailable(DisruptorQueue.java:99) 
at backtype.storm.disruptor$consume_batch_when_available.invoke(disruptor.clj:80) 
at backtype.storm.daemon.executor$fn__3441$fn__3453$fn__3500.invoke(executor.clj:748) 
at backtype.storm.util$async_loop$fn__464.invoke(util.clj:463) 
at clojure.lang.AFn.run(AFn.java:24) 
at java.lang.Thread.run(Thread.java:745) Caused by: java.util.regex.PatternSyntaxException: Illegal repetition near index 1 ./{} ^ 
at java.util.regex.Pattern.error(Pattern.java:1924) 
at java.util.regex.Pattern.closure(Pattern.java:3104) 
at java.util.regex.Pattern.sequence(Pattern.java:2101) 
at java.util.regex.Pattern.expr(Pattern.java:1964) 
at java.util.regex.Pattern.compile(Pattern.java:1665) 
at java.util.regex.Pattern.<init>(Pattern.java:1337) 
at java.util.regex.Pattern.compile(Pattern.java:1047) 
at org.apache.eagle.security.auditlog.FileSensitivityDataJoinExecutor.flatMap(FileSensitivityDataJoinExecutor.java:71) 
at org.apache.eagle.datastream.storm.JavaStormBoltWrapper.execute(JavaStormBoltWrapper.scala:38) 
at backtype.storm.daemon.executor$fn__3441$tuple_action_fn__3443.invoke(executor.clj:633) 
at backtype.storm.daemon.executor$mk_task_receiver$fn__3364.invoke(executor.clj:401) 
at backtype.storm.disruptor$clojure_handler$reify__1447.onEvent(disruptor.clj:58) 
at backtype.storm.utils.DisruptorQueue.consumeBatchToCursor(DisruptorQueue.java:125) ... 6 more 
```
经过分析，hdfs audit日志中含有"{}"：
```
2016-08-26 09:42:03,923 INFO FSNamesystem.audit: allowed=true   ugi=hdfs (auth:SIMPLE)  ip=/172.18.2.177        cmd=listCacheDirectives src={}  dst=null        perm=null       proto=rpc

2016-08-26 10:09:58,107 INFO FSNamesystem.audit: allowed=true   ugi=hdfs (auth:SIMPLE)  ip=/172.18.2.177        cmd=addCachePool        src={poolName:four_gig_pool1, ownerName:impala, groupName:hdfs, mode:0755, limit:40, maxRelativeExpiryMs:2305843009213693951}      dst=null        perm=null       proto=rpc
```
org.apache.eagle.security.auditlog.FileSensitivityDataJoinExecutor的71行报错。此时我做了个简单处理，在68行进行过滤：
```
if(map != null && src != null && !src.contains("{")) {
```
重新编译打包。

###### 运行example
创建/tmp/private文件，并执行hadoop fs -cat /tmp/private。可以看到storm topology有数据输出。

问题：
Visualization Statistics Alerts都看不到数据，可能原因：
1. 创建policy最后一步时指定Notification，我一般选择"eagleStore"
2. 通过查看storm的worker日志可以看到异常，mysql 元数据表alertdetail_hadoop的alertcontext字段设置的长度太短（默认100），我调整成2048。
3. Visualization Statistics两个界面看不到notification信息，未解决。


#### 其他appliation

###### hiveQueryLog

hiveQueryLog可以监控敏感字段的使用情况。
配置eagle的admin->指定stie->hdfsAuditLog->Configuration：
```
classification.accessType=metastoredb_jdbc
classification.password=admin
classification.user=admin
classification.jdbcDriverClassName=com.mysql.jdbc.Driver
classification.jdbcUrl=jdbc:mysql://cs01/hive?createDatabaseIfNotExist=true
```
修改apache-eagle-0.4.0-incubating/conf/sandbox-hiveQueryLog-application.conf:
```
{
  "envContextConfig" : {
    "env" : "storm",
    "mode" : "cluster",
    "topologyName" : "sandbox-hiveQueryRunning-topology",
    "stormConfigFile" : "hive.storm.yaml",
    "parallelismConfig" : {
      "msgConsumer" : 2
    }
  },
  "dataSourceConfig": {
    "flavor" : "stormrunning",
    "zkQuorum" : "cs02:2181,cs03:2181,cs04:2181",
    "zkRoot" : "/jobrunning",
    "zkSessionTimeoutMs" : 15000,
    "zkRetryTimes" : 3,
    "zkRetryInterval" : 2000,
    "RMEndPoints" : "http://cs03:8088/",
    "HSEndPoint" : "http://cs04:19888/",
    "partitionerCls" : "org.apache.eagle.job.DefaultJobPartitionerImpl"
  },
  ...
}  
```
主要是修改RMEndPoints和HSEndPoint。
启动：
```
bin/eagle-topology.sh --main org.apache.eagle.security.hive.jobrunning.HiveJobRunningMonitoringMain  start
```

###### oozieAuditLog

oozieAuditLog监控coodinator的运行情况。
修改eagle的admin->指定stie->oozieAuditLog->Configuration：
```
classification.accessType=oozie_api
classification.oozieUrl=http://cs01:11000/oozie
classification.filter=status=RUNNING
classification.authType=SIMPLE
```