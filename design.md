# design

## kudu

#### 环境配置
低配
24-core(48 logical cores),96GB of RAM,12*2TB disk driver
高配
32-core(64 logical cores),128GB of RAM,12*2TB disk driver

#### 表设计

10+G a tablet
10-100 tablets individual node


#### 分区设计
hash
range（动态增加，删除rangeIMPALA-2890开发中）

#### 参数

- Maximum Process File Descriptors 这个参数决定了Kudu能够同时打开的操作系统文件数。不设置则使用系统的ulimits值，设置后会覆盖系统的设置。需要根据集群的规模及并发处理能力，非常谨慎的设置这个值。
- Default Number of Replicas 这个参数设置了每个Tablet的默认复制因子，默认值为3，表示每个表的数据会在Kudu中存储3份副本。我们可以根据需要修改这个全局默认值，也可以在建表语句中通过'kudu.num_tablet_replicas'属性来设置每个表的副本数，比如：'kudu.num_tablet_replicas' = '1'。
- Kudu Tablet Server Maintenance Threads 这个参数决定了Kudu后台对数据进行维护操作，如写入数据时的并发线程数。并发数越大，吞吐量越高，但对集群计算能力的要求也越高。默认值为1，表示Kudu会采用单线程操作；对于需要大量数据进行快速写入/删除的集群，可以设置更大的值。该值可以设置跟计算节点的数据磁盘数量和CPU核数有关，一般来说，建议设置为4以获取比较均衡的性能，最大不超过8。
- Kudu Tablet Server Block Cache Capacity Tablet的Block buffer cache，根据集群内存配置和数据量规模设置。一般建议至少2GB～4GB。
- Kudu Tablet Server Hard Memory Limit Kudu的Tablet Server能使用的最大内存。Tablet Server在批量写入数据时并非实时写入磁盘，而是先Cache在内存中，在flush到磁盘。这个值设置过小时，会造成Kudu数据写入性能显著下降。对于写入性能要求比较高的集群，建议设置更大的值，比如32GB。

#### kudu局限

- 官方没有测过大于200columns的情形，建议每个表50columns左右
- 官方没有测试过大于10KB的rows，建议列大小在1KB左右
- 每列的大小没有硬性的要求，但大的列可能导致行超出建议大小范围
- 表创建好后，列的类型不可alter


## impala

#### 环境配置

cpu SSSE3 指令集
mem 128GB 理想的256GB
12*nT disk

#### 优化点

partition，物理上隔绝数据，让大部分查询跳过不必要的数据
join，COMPUTE STATS、COMPUTE INCREMENTAL STATS（增加部分分区时使用）；大小表顺序是：大->中->小
COMPUTE STATS
