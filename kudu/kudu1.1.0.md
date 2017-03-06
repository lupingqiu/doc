
## 新特性
1. python API升级，具备JAVA C++client一样的功能（从0.3版本直接升级到1.1），主要的点如下：
  1.1. 改进了Parial Row的语义
  1.2. 增加了range partition支持
  1.3. 提供了 scan api
  1.4. 增强predicate支持
  1.5. 支持所有kudu的类型，包括datetime.datetime转化成UNIXTIME_MICROS
  1.6. 支持修改表（alter）
  1.7. scanner可以读取快照数据
  1.8. 支持scanner副本选择
  1.9. python 
  1.10. 一些bug的修复
2. 为了优化过滤，增加了IN LIST predicate	pushdown的支持（即匹配一列中一个集合的数据，predicate push down类似于hbase的协处理器coprocessors），有些过滤在后台执行。spark、mr和impala query的此项功能还没有完成。
为了便于查错，Java client增加了client-side请求追踪。原先只有最终的错误日志，没有导致最终错误其他日志信息。

## 优化和改进

1. kudu发布了spark2.0（编译于scala2.11）的JAR。
2. 通过配置java client可以从最近的副本读取数据，而不是原先的从leader副本读取数据。不过默认还是后者，可以通过设置scanner builder replicaSelection参数来调整。
3. Tablet server采用一种新策略来维护write-ahead log（WAL）。原先使用log_min_segments_to_retain=2的策略，这种策略会过于频繁flush内存数据，限制了写入的性能。新策略引入了一个新的参数log_target_replay_size_mb，它决定了flush内存数据的阀值，而且这个参数值已经经过实验验证，用户不需要去修改它。这个新策略在某些写入用例下，提升了相对2x倍的写入性能。
4. kudu Raft consensus algorithm 算法加入了一个新的阶段pre-election，它可以在高负荷的情况下，更稳定的进行leader选举，特别在一个tabletserver含有大量的tablet的情况下。
5. 提升了在tabletserver含有大量的tombstoned tablet时，tabletserver的启动的速度。

## 工具

1. kudu tablet leader_step_down，step down一个leader tablet。
2. kudu remote_replica copy拷贝tablet从一个running tabletserver。
3. kudu local_replica delet删除tablet。
4. kudu test loadgen

## 兼容性

1. 1.1的client可以连接到1.0的kudu服务。
2. 1.0的client可以无限制的连接到1.1kudu服务。
3. 滚动升级从1.0到1.1是可能的，但是没有完整的测试。建议安装关闭所有节点，更新版本，启动更新的节点的步骤来升级。

## 参考

https://github.com/cloudera/kudu/blob/master/docs/prior_release_notes.adoc
