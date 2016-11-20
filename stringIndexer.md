# spark StringIndexer and IndexToString usage

## StringIndexer

StringIndexer将一列labels转译成[0,labels基数)的index，labels基数即为labels的去重后总量，index的顺序为labels频次升序，因此出现最多次labels的index为0。如果输入的列时数字类型，我们会把它转化成string，并且使用string转译成index。当pipeline的下游组件例如Estimator或者Transformer使用生成的index时，需要将该组件的输入列名称设置为index的列名。在多数情况下，你可以使用setInputCol设置列名。

另外，当StringIndexer fit了一个dataset后，transfomer一个dataset遇到没见过的labels时，有两种处理策略：
- 抛出异常（默认）
- 跳过整行数据，setHandleInvalid("skip")

### exmaple
``` scala
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)
indexed.show()
```

## IndexToString

与StringIndexer对称的，IndexToString将index映射回原先的labels。通常我们使用StringIndexer产生index，然后使用模型训练数据，最后使用IndexToString找回原先的labels。

### example
``` scala
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)
val indexed = indexer.transform(df)

val converter = new IndexToString()
  .setInputCol("categoryIndex")
  .setOutputCol("originalCategory")

val converted = converter.transform(indexed)
converted.select("id", "originalCategory").show()
```
官方的例子并不是很好，稍微修改一下或许你能更容易明白：
``` scala
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(df)
val indexed = indexer.transform(df)

//设置indexer的labels
val converter = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory").setLabels(indexer.labels)


val df1 = spark.createDataFrame(Seq(
  (10, 2.0),
  (11, 2.0),
  (12, 0.0),
  (13, 0.0),
  (14, 1.0),
  (15, 1.0)
)).toDF("id", "categoryIndex")

val converted = converter.transform(df1)
converted.select("id","categoryIndex"， "originalCategory").show()
```

## 引用

[官方文档](http://spark.apache.org/docs/latest/ml-features.html#indextostring)