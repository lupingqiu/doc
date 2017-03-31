## 新特性

1. 增加了kerberos安全认证，可以通过kerberos tickets或者keytabs文件认证。此新特性是个可选项，推荐在部署环境中增加安全机制。

2. 增加了Transport Layer Security（TLS）网络安全传输协议，kudu将会对任意client和server间的信息通信进行加密。默认加密是开启的，无论client或者server端都可以决定是否启用加密。

3. 增加了粗粒度服务级别的授权。细粒度的授权，例如表级别、字段级别，暂不支持。

4. 增加了清理过期历史版本数据（超过保留时间）的后台任务。减少磁盘空间的使用，特别是有频繁更新的数据。

5. 便于诊断错误，集成了Google Breakpad，它产生的reports可以在配置的日志文件夹中看到。

## 优化

1. 修改了数据目录和数据文件的权限，可以通过--umask配置。升级之后文件权限会更加严格。

2. Web UI 去除了一些用户的敏感信息，例如查询时的predicate values。

3. 默认kudu对配置的磁盘预留1%空间，当磁盘空闲空间不足1%时，为避免完全写满磁盘，会停止写入数据。

4. 数字列（int float double）默认编码BIT_SHUFFLE，binary string类型的变成DICT_ENCODING。此类编码存储机制类似于parquet。

5. WAL使用LZ4压缩，提升写入性能和稳定性。

6. Delta file使用LZ4压缩，可以提高读和写，特别是频繁更新的可压缩数据。

7. Kudu API在查询时支持IS NULL 和 IS NOT NULL（KuduPredicate.newIsNotNullPredicate）的pridicate，spark datasource集成可以利用这些新的predicate。

8. C++ 和 Java client "in partitions"的查询有优化。

9. Java client的异常信息被截断成最大的32KB。

## 兼容性

1. Kudu 1.3 可连接kudu1.0 server，调用新特性时会报错。

2. kudu 1.1 可连接kudu1.3 server，但当集群配置了安全认证，将会报错。

3. 从1.2滚动升级到1.3没有被充分验证。建议使用者关闭整个集群，升级版本，然后重启新版本，通过这种方式来升级。

4. 升级后，如果1.3版本设置了安全认证（authentication or encryption set to "required"），老版本的client将不能连接服务。

5. 升级后，如果1.3版本没有设置安全认证（set to "optional" or "disabled"），老版本的client还能继续连接server。

## 不可兼容变化

1. 因为存储格式变化，1.3版本将不能降级到老版本。

2. 为了在配置了安全的集群上跑mr或者spark任务，需要提供认证凭证。
