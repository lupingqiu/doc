- sql中含有distinct  不必要的
CREATE TABLE temp_node_analysis_order_download_data_3_245021 STORED AS PARQUET AS
SELECT
 distinct
 t0.dp_id AS normal_2102,
 t0.tid AS normal_2101,
 t0.oid AS normal_2100,
 t0.customerno AS normal_2103,
 t0.created AS normal_2112,
 t0.pay_time AS normal_2116,
 t0.consign_time AS normal_2117,
 t0.type AS normal_2115,
 t0.status AS normal_2108,
 t0.num_iid AS normal_2111,
 t0.outer_iid AS normal_2130,
 t0.title AS normal_2121,
 t0.num AS normal_2110,
 t0.total_fee AS normal_2104,
 t0.refund_status AS normal_2118,
 t0.refund_fee AS normal_2119,
 t0.payment AS normal_2107
FROM plt_taobao_order_item t0
INNER JOIN rube_test_tmp t1
ON t0.tid = t1.normal_2000

- 用户执行- cancel 重复提交 impala如何cancel任务 --- cloudera-manger-api解决

- 任务只在08节点 ha无用   --- ha的默认根据ip hash规则，有利于cache，可修改成随机模式。

- 大量的order by offset limit   ----rownumber

- 大量任务累加时，执行很慢  --- 业务方执行有问题，队列也有问题,减少队列配额？  default_pool_max_requests最大执行的请求数，200个无效，后台显示的而是-1，可在impala daemon设置有效，后续压力测试

- kudu森马 order大表 数据写入很慢，qianzhihe没有问题   ---AUTO_FLUSH_BACKGROUND模式解决


- kudu 丢数问题  --backgroud方式没有丢数据，但如果有丢数据定位不到具体哪些数据丢掉了，后续考虑manual flush方式

- es trade upsert 越来越慢问题
