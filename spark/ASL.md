# 协同过滤，交替最小二乘

  ALS（alternating least squares）

## 通俗的理解

  根据两个用户的年龄来判断他们可能有相似的偏好，不叫协同过滤。根据两个用户播放许多相同歌曲来判断他们可能都喜欢某一些歌，才叫协同过滤。

  使用的是一种矩阵分解算法，也叫作矩阵补全，矩阵A的第i行第j列表示，用户i使用商品j的次数或者是否使用过。将A分解成两个小矩阵X和Y的乘积。X和Y行很多，但列很少，A=XY(T)。因为X和Y都是未知数，A已知（稀疏），我们给Y随机生成一个初始矩阵，然后通过线性代数并行化计算X的每一行最优解，即最小化|A(i)Y(Y(T)Y(-1))-X(i)|，或者最小化两者的平方差，这也是“最小二乘”的由来。完事后，又可以通过X，用同样的方法计算Y（j），如此反复（交替），X和Y会收敛到一个合适的结果。

## 反馈类型

  显式反馈：矩阵中的元素就是user对item的显性偏好，例如打分值。
  隐式反馈：真实案例中一般是隐式反馈，例如查看数、点击数、是否喜欢、是否分享。

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
