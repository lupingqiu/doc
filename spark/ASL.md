# 协同过滤，交替最小二乘

  ALS（alternating least squares）

  根据两个用户的年龄来判断他们可能有相似的偏好，不叫协同过滤。根据两个用户播放许多相同歌曲来判断他们可能都喜欢某一些歌，才叫协同过滤。

## 申明

  一般我们使用大写字母表示矩阵，例如 $A$ ，对应小写字母表示该矩阵某一行的向量（列向量），例如 $a_{i}$表示矩阵 $A$ 的第`i`行的转置。

## 通俗的理解

  A是一个`m*n`的矩阵，每一行代表一个用户，m表示用户的个数；每一列表示一个商品，n表示商品的个数。A可以用两个小矩阵 $U$`(m*k)`和 $V$`(n*k)`的乘积来近似，即$A=U{V}^{T}$，其中`k<m,k<n`，复杂度由`o(m*n)`降到`o((m+n)*k)`。

  这是一种矩阵分解算法，也叫作矩阵补全。因为 $U$ 和 $V$ 都是未知数，A已知（稀疏），我们给 $V$ 随机生成一个初始矩阵，然后通过线性代数并行化计算 $U$ 的每一行最优解，即 $u_{i}=a_{i}V(V^{T}V)^{-1}$ ，其中 $u_{i} = (u_{i0},u_{i1}...,u_{ik}$)，$a_{i}$ 表示 $A$ 的第`i`行，`i`表示第`i`个用户， 最小化两者的平方误差，这也是“最小二乘”的由来。完事后，又可以通过$U$，用同样的方法计算 $V_{j}$，`j`表示第`j`个商品 ，如此反复（交替），$U$ 和 $V$ 会收敛到一个合适的结果。

  $A=U{V}^{T}$ 如何得到 $u_{i}=a_{i}V(V^{T}V)^{-1}$ 呢，即已知 $V$ 和 $A$，求 $U$。$U$ 右侧乘以 $V$ ，使得 $U$ 右侧变成一个方阵$V^{T}V$，再乘以这个方阵的逆（只有方阵才有逆，且假设是可逆的，或者说非奇异的），那么 $U$ 的右侧变成了一个单位矩阵 $E$ 。等式左侧 $A$做同样的处理，即得到结果。

  我们再用维度验证一下，$u_{i}$是`1*k`向量，$a_{i}V(V^{T}V)^{-1}$ 是`(1*n)*(n*k)*(k*n)*(n*k)=(1*k)`向量，没毛病！

## 反馈类型

#### 显式反馈

  矩阵中的元素 $a_{ij}$ 就是user对item的显性偏好，例如打分值。

  通过 $a_{ij}=u_{i}^{T}v_{j}$ 来预测。目标函数是：
  $$
    min_{u,v}\sum_{a_{ij}}(a_{ij}-u_{i}^{T}v_{j})^2 + \lambda (\left \| u_{i} \right \|^2+\left \| v_{j} \right \|^2)
  $$

#### 隐式反馈

  真实案例中一般是隐式反馈，$a_{ij}$ 例如查看数、点击数、是否喜欢、是否分享、购买次数。

  将$a_{ij}$ 转化成两个不同量来衡量：偏好 $p_{ij}$ 和信任度 $c_{ij}$。
  $$
  p_{ij}=\begin{cases}
  1 & \text{ if } a_{ij}>0 \\
  0 & \text{ if } a_{ij}=0
  \end{cases}
  $$
  $$
  c_{ij} = 1 + \alpha a_{ij}
  $$

  简单举例，$p_{ij}$ 表示了用户是否有偏好，$c_{ij}$ 表示通过查看次数来衡量偏好程度。

  目标函数$J$：
  $$
  min_{u,v}\sum_{i,j}c_{ij}(p_{ij}-u_{i}^{T}v_{j})^2 + \lambda (\sum_{i}\left \| u_{i} \right \|^2+\sum_{j}\left \| v_{j} \right \|^2)
  $$

  推导：

  $$
  U = \begin{bmatrix}
  u_{0}^{T}\\
  u_{1}^{T}\\
  ...\\
  u_{m}^{T}\\
  \end{bmatrix}
  \,
  V = \begin{bmatrix}
  v_{0}^{T}\\
  v_{1}^{T}\\
  ...\\
  v_{n}^{T}\\
  \end{bmatrix}
  $$
  目标函数对 $u_{i}$ 求导，$u_{i}$ 是一个`k*1`的向量，值对向量的求导也是向量，且有“前导不变，后导转置”的原则，得到:
  $$
  \begin{aligned}
  \frac{\partial J}{\partial u_{i}} &= -2\sum_{j}c_{ij}(p_{ij}-u_{i}^{T}v_{j})v_{j} + 2\lambda u_{i} \\
  
  &=-2\sum_{j}c_{ij}(p_{ij}-v_{j}^{T}u_{i})v_{j} + 2\lambda u_{i} \\
  
  &=-2\sum_{j}(c_{ij}p_{ij}v_{j}-c_{ij}v_{j}^{T}u_{i}v_{j}) + 2\lambda u_{i} \\
  
  &=-2V^{T}C^{i}p_{i} + V^{T}C^{i}Vu_{i} + 2\lambda u_{i}
  \end{aligned}
  $$
  其中 $V$ 是`n*k`矩阵，$C^{i}$ 是`n*n`对角矩阵， $p_{i}$ 是`n*1`列向量。令导数=0：
  $$
  \begin{aligned}
  V^{T}C^{i}p_{i}&=(V^{T}C^{i}V+\lambda I)u_{i}\\
  
  =>u_{i}&=(V^{T}C^{i}V+\lambda I)^{-1}V^{T}C^{i}p_{i} \\
  
  &=(V^{T}V+V^{T}(C^{i}-I)V+\lambda I)^{T}V^{T}C^{i}p_{i}
  \end{aligned}
  $$
  得到 $u_{i}$ 向量，其中 $V^{T}V$ 与`i`无关，可以提前计算，而 $C^{i}-I$ 只有 $n_{i}$ 非零元素，$n_{i} < n$，它表示用户`i`对`n`个商品 $a_{ij}>0$的个数，也即是 $p_{ij}=1$ 的个数。

#### 两者区别

  显性反馈矩阵分解优化时，对于missing data未知评分数据，不会当作训练数据输入到模型中，优化时也只是对已知评分的数据优化。
  隐性反馈，考虑信任度 $c_{ij}$ ，会利用所有可能的m，n键值对，没有所谓的missing data，因为即使 $p_{ij}=0$ ， 信任度也是有值的，只是比较小而已。

## 相关基础

  SVD：http://blog.csdn.net/zhongkejingwang/article/details/43053513

## spark2.x样例

```scala
package com.rube.spark.example.als

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}

/**
  * Created by rube on 17-4-30.
  */
class ALSExample {

	def runExample: Unit ={
		val conf = new SparkConf().setAppName("ALS example").setMaster("local")
		val sc = new SparkContext(conf)
		//加载文件
		val data = sc.textFile("/home/rube/workspace/spark-2.1.0-bin-hadoop2.7/data/mllib/als/test.data")

		val ratings = data.map(_.split(",") match {case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble) })
		// 模型训练
		val rank = 10
		val numIterations = 10
		val model = ALS.train(ratings, rank, numIterations, 0.01)
		//取出用户和商品组
		val userProducts = ratings.map{case Rating(user, product, rate) => (user, product)}
		//根据模型预测评分
		val predictions = model.predict(userProducts).map{ case Rating(user,product,rate) => ((user,product), rate)}
		//原始数据与预测数据全关联
		val rateAndPreds = ratings.map{ case Rating(user,product,rate) => ((user,product), rate)}.join(predictions)
		//计算误差的平方
		val mse = rateAndPreds.map{
			case ((user,product),(r1,r2)) =>
				val err = (r1-r2)
				err*err
		}.mean()
		println("mean squared error: "+ mse)

		sc.stop()
	}
}
object ALSExample{
	def main(args: Array[String]): Unit = {
		val als = new ALSExample()
		als.runExample
	}
}
// spark-submit --class com.rube.spark.example.als.ALSExample --master local --deploy-mode client spark2example.jar
```

## 源码解析
