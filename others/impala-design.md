# design

## impala

#### 环境配置

cpu SSSE3 指令集
mem 128GB 理想的256GB
12*nT disk

#### 优化点

partition，物理上隔绝数据，让大部分查询跳过不必要的数据
join，COMPUTE STATS、COMPUTE INCREMENTAL STATS（增加部分分区时使用）；大小表顺序是：大->中->小
COMPUTE STATS
