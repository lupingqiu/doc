
1. spark各种运行模式

  --master 可以是本地local（所有都在本地），yarn，meos://host:port,standalone的spark://host:port

  --deploy-mode 默认client，cluster，区别在于SparkContext是否在集群内部创建。

  参考
  - [url1](http://blog.csdn.net/lovehuangjiaju/article/details/48634607)

2. spark基本术语

  - RDD 适合数据挖掘，机器学习，图计算等具有大量的迭代计算任务；不适于例如分布式爬虫需要频繁更新共享状态的任务。RDD在Spark中是一个只读的（val类型）、经过分区的记录集合。RDD在Spark中只有两种创建方式：（1）从存储系统中创建；（2）从其它RDD中创建。从存储中创建有多种方式，可以是本地文件系统，也可以是分布式文件系统，还可以是内存中的数据。 Transformations（转换)、Actions两种。transformations操作会将一个RDD转换成一个新的RDD，需要特别注意的是所有的transformation都是lazy的，如果对scala中的lazy了解的人都知道，transformation之后它不会立马执行，而只是会记住对相应数据集的transformation，而到真正被使用的时候才会执行，例如distData.filter(e=>e>2) transformation后，它不会立即执行，而是等到distDataFiltered.collect方法执行时才被执行

  - DataFrame DataFrame是一种以RDD为基础的分布式数据集，与传统RDBMS的表结构类似。与一般的RDD不同的是，DataFrame带有schema元信息，即DataFrame所表示的表数据集的每一列都带有名称和类型，它对于数据的内部结构具有很强的描述能力。因此Spark SQL可以对藏于DataFrame背后的数据源以及作用于DataFrame之上的变换进行了针对性的优化，最终达到大幅提升运行时效率。http://blog.csdn.net/lovehuangjiaju/article/details/48661847

  - [url2](http://blog.csdn.net/book_mmicky/article/details/25714419)

3. sparkSQL

  http://blog.csdn.net/book_mmicky/article/details/39956809 spark sql运行原理

4. 源码

  http://blog.csdn.net/lovehuangjiaju/article/details/49123975

5. idea 远程调试

submit-class 加入： $JAVA_OPTS
done < <("$RUNNER" -cp "$LAUNCH_CLASSPATH" org.apache.spark.launcher.Main $JAVA_OPTS "$@")

执行
export JAVA_OPTS="$JAVA_OPTS -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=5005"

idea中创建remote run，端口与5005一致

后台执行：
spark-submit --master spark://rube-ubuntu:7077 --class SparkWordCount --executor-memory 1g /home/rube/cloudera/spark_demo/out/artifacts/spark_demo_jar/spark_demo.jar hdfs://ns1/README.md hdfs://ns1/SparkWordCountResult

indea中运行debug，然后调试
