
lr.setMaxIter(10).setRegParam(0.01)

val paramMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter, 30).put(lr.regParam -> 0.1, lr.threshold -> 0.55)

model2.transform(test2).select("features", "label", "myProbability", "prediction").collect().foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) => println(s"($features, $label) -> prob=$prob, prediction=$prediction")}

val test2 = spark.createDataFrame(Seq(
  (0.0, Vectors.dense(-1.0, 1.5, 1.3)),
  (1.0, Vectors.dense(3.0, 2.0, -0.1)),
  (0.0, Vectors.dense(0.0, 2.2, -1.5))
)).toDF("label", "features")

val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10)

Array[org.apache.spark.sql.Row] = Array(
[0,Hi I heard about Spark,WrappedArray(hi, i, heard, about, spark),(20,[0,5,9,17],[1.0,1.0,1.0,2.0])], 
[0,I wish Java could use case classes,WrappedArray(i, wish, java, could, use, case, classes),(20,[2,7,9,13,15],[1.0,1.0,3.0,1.0,1.0])], 
[1,Logistic regression models are neat,WrappedArray(logistic, regression, models, are, neat),(20,[4,6,13,15,18],[1.0,1.0,1.0,1.0,1.0])])

val sentenceData = spark.createDataFrame(Seq(
  (0, "0 1 2 3"),
  (0, "1 4 5 6"),
  (1, "1 7 8 9")
)).toDF("label", "sentence")


import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

val sentenceData = spark.createDataFrame(Seq(
  (0, "a a a a"),
  (0, "a a a a a b"),
  (1, "b b b b")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val wordsData = tokenizer.transform(sentenceData)
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(3)
val featurizedData = hashingTF.transform(wordsData)

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val rescaledData = idfModel.transform(featurizedData)
rescaledData.select("features", "label").take(3).foreach(println)


(hi, i, heard, about, spark),
24417,49304,73197,91137,234657
0.0,0.28768207245178085,0.28768207245178085,0.0,0.6931471805599453

i, wish, java, could, use, case, classes, heard, about, hi),
20719,24417,49304,55551,73197,91137,116873,147765,162369,192310
0.6931471805599453,0.0,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453

i, logistic, regression, models, are, neat, java, about
13671,24417,55551,91006,91137,132713,167122,190884
0.6931471805599453,0.0,0.28768207245178085,0.6931471805599453,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453


import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val df = spark.createDataFrame(Seq(
  (0, Array("a", "b", "c","b","c","c")),
  (1, Array("a", "b", "b", "c","e"))
)).toDF("id", "words")

// fit a CountVectorizerModel from the corpus
val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(5).setMinDF(0.5).fit(df)

// alternatively, define CountVectorizerModel with a-priori vocabulary
val cvm = new CountVectorizerModel(Array("a", "b", "c","e")).setInputCol("words").setOutputCol("features")

cvm.transform(df).select("features").take(5).foreach(println)

cvModel.transform(df).select("features").take(5).foreach(println)

http://stackoverflow.com/questions/35205865/what-is-the-difference-between-hashingtf-and-countvectorizer-in-spark


import org.apache.spark.ml.feature.Word2Vec

// Input data: Each row is a bag of words from a sentence or document.
val documentDF = spark.createDataFrame(Seq(
  "a b c".split(" "),
  "a c b".split(" "),
  "a b c d a a".split(" ")
).map(Tuple1.apply)).toDF("text")

// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(2).setMinCount(0)
val model = word2Vec.fit(documentDF)
val result = model.transform(documentDF)
result.select("text","result").take(3).foreach(println)


import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

val input = sc.textFile("/home/rube/apache/xaa").map(line => line.split(" ").toSeq)

val word2vec = new Word2Vec()

val model = word2vec.fit(input)

val synonyms = model.findSynonyms("1", 5)

for((synonym, cosineSimilarity) <- synonyms) {
  println(s"$synonym $cosineSimilarity")
}

// Save and load model
model.save(sc, "myModelPath")
val sameModel = Word2VecModel.load(sc, "myModelPath")


./spark-submit --master local --class org.apache.spark.examples.mllib.MovieLensALS \
/home/rube/apache/spark-2.0.0-bin-hadoop2.6/examples/jars/spark-examples_2.11-2.0.0.jar \
--rank 5 --numIterations 5 --lambda 1.0 --   /home/rube/apache/sample_movielens_data.txt    --jars  ../jars/scala-library.jar

172.18.60.106  255.255.255.0  172.18.60.1  114.114.114.114


import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

val sentenceDataFrame = spark.createDataFrame(Seq(
  (0, "Hi I heard about Spark"),
  (1, "I wish Java could use case classes"),
  (2, "Logistic,regression,models,are,neat")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("words", "label").take(3).foreach(println)
val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("words", "label").take(3).foreach(println)


import org.apache.spark.ml.feature.NGram

val wordDataFrame = spark.createDataFrame(Seq(
  (0, Array("Hi", "I", "heard", "about", "Spark")),
  (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
  (2, Array("Logistic", "regression", "models", "are", "neat"))
)).toDF("label", "words")

val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")
val ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.take(3).map(_.getAs[Stream[String]]("ngrams").toList).foreach(println)

import org.apache.spark.ml.feature.Binarizer

val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
val dataFrame = spark.createDataFrame(data).toDF("label", "feature")

val binarizer: Binarizer = new Binarizer().setInputCol("feature").setOutputCol("binarized_feature").setThreshold(0.5)

val binarizedDataFrame = binarizer.transform(dataFrame)
val binarizedFeatures = binarizedDataFrame.select("binarized_feature")
binarizedFeatures.collect().foreach(println)



import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
  Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
  Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(df)
val pcaDF = pca.transform(df)
val result = pcaDF.select("pcaFeatures")
result.show()


import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.dense(-2.0, 2.3),
  Vectors.dense(0.0, 0.0),
  Vectors.dense(0.6, -1.1)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val polynomialExpansion = new PolynomialExpansion().setInputCol("features").setOutputCol("polyFeatures").setDegree(3)
val polyDF = polynomialExpansion.transform(df)
polyDF.select("polyFeatures").take(3).foreach(println)


import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)


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

val converter = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory")

val converted = converter.transform(indexed)
converted.select("id", "originalCategory").show()

val df1 = spark.createDataFrame(Seq(
  (10, 2.0),
  (11, 2.0),
  (12, 0.0),
  (13, 0.0),
  (14, 1.0),
  (15, 1.0)
)).toDF("id", "category1")

val converter1 = new IndexToString().setInputCol("category1").setOutputCol("originalCategory1")
val converted1 = converter1.transform(df1)


converted1.select("id", "originalCategory1").show()
val converter1 = new IndexToString().setInputCol("category1").setOutputCol("originalCategory1").setLabels(indexer.labels)
converter1: org.apache.spark.ml.feature.IndexToString = idxToStr_ca0d061a7865
