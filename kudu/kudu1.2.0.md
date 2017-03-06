## 新特性
1. kudu clients和servers可以编辑用户的数据，例如log信息，java的异常信息和状态信息。但用户的元数据是不可编辑的，例如表名，字段名，分区边界。默认是可编辑的，但可以通过设置log_redact_user_data=false关闭。
2. kudu一致性保证的能力大幅度提升了：
3. 每个副本都会track它们的 safe timestamp，这个时间戳是最大时间戳，在这个时间戳上读是可重复的。
4. SCAN_AT_SNAPSHOT模式的scan，要不等待本副本的snapshot safe后再查，要不路由到一个safe的副本查。如此保证数据scan是可重复的。
5. Kudu会保留以往所有历史数据，无论是插入，还是删除，还是相同key插入一条新的数据。以往版本的kudu不会保留历史数据在这种情况下。如此tablet server可以得到历史某一时间段点的准确的快照，即便是重新插入的情形。
6. kudu client会自动记住它们最近成功读取或者写入操作的时间戳。在使用READ_AT_SNAPSHOT模式，且没有指定时间戳的情况下，scan会自动指定一个比最近写入时间戳大的时间戳。写入同样会传播timestamp，确认一系列的有因果关系的操作，能被指定有序的时间戳。总的来说，这个改变，可以保证数据的读写一致性，同样保证了在其他clients上的快照查询能等到一致的结果。
7. kuduserver自动限制了log文件数量，默认是10个，通过参数max_log_files设置。

## 优化和改进
1. java和c++client的日志将会变得更平和，不在记录正常操作的日志，而记录error日志。
2. c++client提供一个KuduSession::SetErrorBufferSpace API，通过它可以限制同步操作异常的buffer大小。
3. java client可以获取tablet地址信息1000个一个批次（原先是10）。如此可以提升spark或者impala查询具有大量tablets的表性能。
4. kudu master表元数据信息的锁竞争大幅度缓解。如此提升了在大集群环境下寻址（tablet）的高并发度。
5. tablet server端的高并发写的锁竞争同样被缓解了。
6. 写日志的锁竞争也被缓解。

## 修复的bug
1. KUDU-1508，ext4file的文件系统损坏。
2. KUDU-1399，实现LRU cache解决长时间运行的kudu机器openfiles不够的问题。默认kudu会使用ulimit的一半的量。
省略

## 兼容性
1. 1.2.0与历史版本兼容
2. 1.2client可以了解1.0server，只是有些没有的功能不可用。
3. 1.0cleint可以连接1.2，没有任何限制。
4. 滚动升级从1.0到1.1是可能的，但是没有完整的测试。建议安装关闭所有节点，更新版本，启动更新的节点的步骤来升级。

## 不可兼容变化
1. 副本因子最大值改成7，并且副本不能是偶数。
2. 不提供GROUP_VARINT无损压缩算法。

## 约束性
1. 列数，建议不超过300列，建议列数越少越好。
2. cell大小，不能大于64KB，不然写入时client有error信息。
3. 有效标识符，表名列名严格要求是UTF-8，且不能超过256个字符。

## 引用
https://github.com/cloudera/kudu/blob/master/docs/release_notes.adoc
