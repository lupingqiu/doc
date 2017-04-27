

## 介绍

Kudu集HDFS的顺序读和HBASE的随机读于一身，同时具备高性能的随机写，以及很强大的可用性（单行事务，一致性协议），支持Impala spark计算引擎。

## 什么时候使用kudu

1. 大规模数据复杂的实时分析，例如大数据量的join。
2. 数据有更新
3. 查询准实时

## 存储

Kudu的存储是不基于HDFS的，构建集群时，kudu很有可能和HDFS共同占用物理磁盘或者云磁盘，理想情况是独立空间。

> 正式环境中，占用磁盘空间，比数据真实空间要大，比HDFS相同数据占用空间也大。kudu存储了每次更新的数据，即数据变化频率越大，存储空间占用也越大，在1.3.0版本中有[改进](http://blog.csdn.net/lookqlp/article/details/68927640)（新特性第四点）。
>
> 出现过清理表后，依然占用空间的情况。（待进一步验证）
> 建议使用SSD硬盘。

## 配置参考

低配
24-core(48 logical cores),96GB of RAM,12*2TB disk driver
高配
32-core(64 logical cores),128GB of RAM,12*2TB disk driver

> 更低配置也能满足相对小数据量的查询，例如8核16G，但性能和并发量不能保证。

## 表设计

10+G a tablet
10-100 tablets individual node

> 在配置32C，64G机器上，跑过1000个tablet情况，能正常写入，但在大量查询情况下，出现tablet time out，rpc满了的情况，


## 分区设计

hash
range（动态增加，删除range IMPALA-2890开发中）

## 参数

- Maximum Process File Descriptors 这个参数决定了Kudu能够同时打开的操作系统文件数。不设置则使用系统的ulimits值，设置后会覆盖系统的设置。需要根据集群的规模及并发处理能力，非常谨慎的设置这个值。

- Default Number of Replicas 这个参数设置了每个Tablet的默认复制因子，默认值为3，表示每个表的数据会在Kudu中存储3份副本。我们可以根据需要修改这个全局默认值，也可以在建表语句中通过'kudu.num_tablet_replicas'属性来设置每个表的副本数，比如：'kudu.num_tablet_replicas' = '1'。

- Kudu Tablet Server Maintenance Threads 这个参数决定了Kudu后台对数据进行维护操作，如写入数据时的并发线程数。并发数越大，吞吐量越高，但对集群计算能力的要求也越高。默认值为1，表示Kudu会采用单线程操作；对于需要大量数据进行快速写入/删除的集群，可以设置更大的值。该值可以设置跟计算节点的数据磁盘数量和CPU核数有关，一般来说，建议设置为4以获取比较均衡的性能，最大不超过8。

- Kudu Tablet Server Block Cache Capacity Tablet的Block buffer cache，根据集群内存配置和数据量规模设置。一般建议至少2GB～4GB。

- Kudu Tablet Server Hard Memory Limit Kudu的Tablet Server能使用的最大内存。Tablet Server在批量写入数据时并非实时写入磁盘，而是先Cache在内存中，在flush到磁盘。这个值设置过小时，会造成Kudu数据写入性能显著下降。对于写入性能要求比较高的集群，建议设置更大的值，比如32GB。

## 局限

- 官方没有测过大于200columns的情形，建议每个表50columns左右

- 官方没有测试过大于10KB的rows，建议列大小在1KB左右

- 每列的大小没有硬性的要求，但大的列可能导致行超出建议大小范围

- 表创建好后，列的类型不可alter

## 痛点

- 增加节点时，tablet不会自动均衡，需要通过运维手段，将tablet迁移到新节点。

  > 迁移时，不能迁移leader tablet。

- 节点丢失后（默认5分钟），tablet副本会在其他节点生成，节点再恢复时，该节点没有任何tablet。

  > 连接不上节点后尽量在5分钟内解决和恢复节点。

- 当节点含有大量tablet时，重启tablet server，加载tablet非常耗时，800 tablets 耗时近30分钟，期间tabletserver日志没有任何信息。

- tablet_history_max_age_sec 超过参数时间的历史数据会被清理，如果是base数据不会被清理。而真实运行时数据大小持续累加，没有被清理。

  > 需要进一步验证。

## 设计

 - [Master设计](http://blog.csdn.net/lookqlp/article/details/51355195)
 - [Tabelt设计](http://blog.csdn.net/lookqlp/article/details/51416829)
