## 引言

逻辑回归常用于预测疾病发生的概率，例如因变量是是否恶性肿瘤，自变量是肿瘤的大小、位置、硬度、患者性别、年龄、职业等等（很多文章里举了这个例子，但现代医学发达，可以通过病理检查，即获取标本放到显微镜下观察是否恶变来判断）；广告界中也常用于预测点击率或者转化率(cvr/ctr)，例如因变量是是否点击，自变量是物料的长、宽、广告的位置、类型、用户的性别、爱好等等。
本章主要介绍逻辑回归算法推导、梯度下降法求最优值的推导及spark的源码实现。

## 常规方法
一般回归问题的步骤是：
 1. 寻找预测函数（h函数，hypothesis）
 2. 构造损失函数（J函数）
 3. 使损失函数最小，获得回归系数θ

而第三步中常见的算法有：
	1. 梯度下降
	2. 牛顿迭代算法
	3. 拟牛顿迭代算法（BFGS算法和L-BFGS算法）
其中随机梯度下降和L-BFGS在spark mllib中已经实现，梯度下降是最简单和容易理解的。

## 推导

### 二元逻辑回归

 1. 构造预测函数
$$h_{\theta }(x)=g(\theta ^{T}x)=\frac{1}{1+e^{^{-\theta ^{T}x}}}$$
其中：
$$\theta ^{T}x=\sum_{i=1}^{n}\theta _{i}x_{i}=\theta _{0}+\theta _{1}x_{1}+\theta _{2}x_{2}+...+\theta _{n}x_{n}
\\\theta =\begin{bmatrix}
\theta _{0}\\
\theta _{1}\\
...\\
\theta _{n}
\end{bmatrix}
,x =\begin{bmatrix}
x _{0}\\
x _{1}\\
...\\
x _{n}
\end{bmatrix}$$
为何LR模型偏偏选择sigmoid 函数呢？逻辑回归不是回归问题，而是二分类问题，因变量不是0就是1，那么我们很自然的认为概率函数服从伯努利分布，而伯努利分布的指数形式就是个sigmoid 函数。
函数$h_{\theta}(x)$表示结果取1的概率，那么对于分类1和0的概率分别为：
$$P(y=1|x;\theta )=h_{\theta }(x)\\P(y=0|x;\theta )=1-h_{\theta }(x)$$
概率一般式为：
$$P(y|x;\theta )=(h_{\theta }(x))^{y}((1-h_{\theta }(x)))^{1-y}$$

 2. 最大似然估计的思想
 当从模型总体随机抽取m组样本观测值后，我们的目标是寻求最合理的参数估计$\theta{}'$使得从模型中抽取该m组样本观测值的概率最大。最大似然估计就是解决此类问题的方法。求最大似然函数的步骤是：
     1. 写出似然函数
     2. 对似然函数取对数
     3. 对对数似然函数的参数求偏导并令其为0，得到方程组
     4. 求方程组的参数

 为什么第三步要取对数呢，因为取对数后，乘法就变成加法了，且单调性一致，不会改变极值的位置，后边就更好的求偏导。
 3. 构造损失函数
 线性回归中的损失函数是：
 $$J(\theta )=\frac{1}{2m}\sum_{i=1}^{m}(y^{i}-h_{\theta }(x^{i}))^{2}$$
 线性回归损失函数有很明显的实际意义，就是平方损失。而逻辑回归却不是，它的预测函数$h_{\theta}(x)$明显是非线性的，如果类比的使用线性回归的损失函数于逻辑回归，那$J(\theta)$很有可能就是非凸函数，即存在很多局部最优解，但不一定是全局最优解。我们希望构造一个凸函数，也就是一个碗型函数做为逻辑回归的损失函数。
按照求最大似然函数的方法，逻辑回归似然函数：
$$L(\theta )=\prod_{i=1}^{m}P(y_{i}|x_{i};\theta )=\prod_{i=1}^{m}(h_{\theta }(x_{i}))^{y_{i}}((1-h_{\theta }(x_{i})))^{1-y_{i}}$$
其中m表示样本数量，取对数：
$$l(\theta )=logL(\theta )=\sum_{i=1}^{m}(y_{i}logh_{\theta }(x_{i})+(1-y_{i})log(1-h_{\theta }(x_{i})))$$
我们的目标是求最大$l(\theta )$时的$\theta$，如上函数是一个上凸函数，可以使用梯度上升来求得最大似然函数值(最大值)。或者上式乘以-1，变成下凸函数，就可以使用梯度下降来求得最小负似然函数值（最小值）：
$$J(\theta )=-\frac{1}{m}l(\theta )$$
同样是取极小值，思想与损失函数一致，即我们把如上的$J(\theta)$作为逻辑回归的损失函数。Andrew Ng的课程中，上式乘了一个系数1/m，我怀疑就是为了和线性回归的损失函数保持一致吧。
 4.  求最小值时的参数
  我们求最大似然函数参数的第三步时，令对参数$\theta$偏导=0，然后求解方程组。考虑到参数数量的不确定，即参数数量很大，此时直接求解方程组的解变的很困难，或者根本就求不出精确的参数。于是，我们用随机梯度下降法，求解方程组的值。
 当然也可以使用牛顿法、拟牛顿法。梯度下降法是最容易理解和推导的，如下是推导过程：
 梯度下降$\theta$的更新过程，走梯度方向的反方向：
 $$\theta _{j}:=\theta _{j}-\alpha\frac{\delta }{\delta _{\theta _{j}}}J(\theta )$$
 其中：
 $$\frac{\delta }{\delta _{\theta _{j}}}J(\theta )=-\frac{1}{m}\sum_{i=1}^{m}\left ( y_{i}\frac{1}{h_{\theta }(x_{i})} \frac{\delta }{\delta _{\theta _{j}}}h_{\theta }(x_{i})-(1-y_{i})\frac{1}{1-h_{\theta }(x_{i})}\frac{\delta }{\delta _{\theta _{j}}}h_{\theta }(x_{i})\right )
\\=-\frac{1}{m}\sum_{i=1}^{m}\left (  y_{i}\frac{1}{g(\theta ^{T}x_{i})}-(1-y_{i})\frac{1}{1-g(\theta ^{T}x_{i})}\right )\frac{\delta }{\delta _{\theta _{j}}}g(\theta ^{T}x_{i})
\\=-\frac{1}{m}\sum_{i=1}^{m}\left (  y_{i}\frac{1}{g(\theta ^{T}x_{i})}-(1-y_{i})\frac{1}{1-g(\theta ^{T}x_{i})}\right )g(\theta ^{T}x_{i})(1-g(\theta ^{T}x_{i}))\frac{\delta }{\delta _{\theta _{j}}}\theta ^{T}x_{i}
\\=-\frac{1}{m}\sum_{i=1}^{m}(y_{i}(1-g(\theta ^{T}x_{i}))-(1-y_{i})g(\theta ^{T}x_{i}))x_{i}^{j}
\\=-\frac{1}{m}\sum_{i=1}^{m}(y_{i}-g(\theta ^{T}x_{i}))x_{i}^{j}
\\=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta }(x_{i})-y_{i}))x_{i}^{j}
$$
第二步推导请注意：
$$\left (\frac{f(x)}{g(x)}  \right ){}'=\frac{g(x)f{}'(x)-f(x)g{}'(x)}{g^{2}(x)}
\\\left (e^{x} \right ){}'=e^{x}$$
那么可以推导：
$$\frac{\delta }{\delta _{\theta _{j}}}g(\theta ^{T}x_{i})=-\frac{e^{-\theta^{T} x_{i}}}{(1+e^{-\theta^{T} x_{i}})^{2}}\frac{\delta }{\delta _{\theta _{j}}}(-1)\theta^{T} x_{i}
=g(\theta ^{T}x_{i})(1-g(\theta ^{T}x_{i})){\delta _{\theta _{j}}}\theta^{T} x_{i}$$
因此更新过程可以写成：
$$\theta _{j}:=\theta _{j}-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta }(x_{i})-y_{i}))x_{i}^{j}$$
那迭代多少次停止呢，spark是指定迭代次数和比较两次梯度变化或者cost变化小于一定值时停止。
 5. 过拟合问题
过拟合问题，即我们求得的回归系数在实验集中效果很好，但之外的数据效果很差。机器学习中的特征基本上是靠人的经验选择的，有可能某一些特征或者特征组合与因变量没有任何关系，即某些$\theta _{i}\approx 0$。所以我们需要把不必要的特征剔除，一般我们使用正则化来保留所有特征，并让它相应的系数$\approx 0$，L1范数正则化后$\theta$的更新：
$$\theta _{j}:=\theta _{j}-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta }(x_{i})-y_{i}))x_{i}^{j}-\frac{\lambda }{m}\theta _{j}$$
$\lambda$越大，对模型的复杂度惩罚越大，有可能出现欠拟合现象。$\lambda$越小，惩罚越小，可能新出现过拟合现象。spark逻辑回归的随机梯度下降法中，使用的是L2范数正则化。
### 多元逻辑回归
推广到K元逻辑回归，即因变量为0、1、2、...、k-1。在二元逻辑回归中有这样的性质：
$$log\frac{P(y=1|x,\theta )}{P(y=0|x,\theta )}=\theta ^{T}x$$
推广至K元逻辑回归：
$$log\frac{P(y=1|x,\theta )}{P(y=0|x,\theta )}=\theta_{1} ^{T}x
\\log\frac{P(y=2|x,\theta )}{P(y=0|x,\theta )}=\theta_{2} ^{T}x
\\...
\\log\frac{P(y=K-1|x,\theta )}{P(y=0|x,\theta )}=\theta_{K-1} ^{T}x$$
其中，$\theta =(\theta _{1},\theta _{2},...,\theta _{K-1})^{T}$，是个（k-1）*（n+1）的矩阵，n为特征的个数，加1是增加截距项。去除对数则得到概率分布：
$$P(y=0|x,\theta )=\frac{1}{1+\sum_{i=1}^{K-1}e^{\theta _{i}^{T}x}}
\\P(y=1|x,\theta )=\frac{e^{\theta _{1}^{T}x}}{1+\sum_{i=1}^{K-1}e^{\theta _{i}^{T}x}}
\\...
\\P(y=K-1|x,\theta )=\frac{e^{\theta _{K-1}^{T}x}}{1+\sum_{i=1}^{K-1}e^{\theta _{i}^{T}x}}$$
K元逻辑回归似然函数：
$$L(\theta )=\prod_{i=1}^{m}P(y|x,\theta )$$
定义：
$$\alpha (y_{i})=1 \quad if \quad  y_{i}=0
\\\alpha (y_{i})=0 \quad if  \quad y_{i}\neq 0$$
取对数：
$$l(\theta ,x)=\sum_{i=1}^{m}logP(y_{i}|x_{i},\theta )
\\=\sum_{i=1}^{m}\alpha (y_{i})logP(y=0|x_{i},\theta )+(1-\alpha (y_{i}))logP(y_{i}|x_{i},\theta )
\\=\sum_{i=1}^{m}\alpha (y_{i})log\frac{1}{1+\sum_{k=1}^{K-1}e^{\theta _{k}^{T}x}}+(1-\alpha (y_{i}))log\frac{e^{\theta _{y_{i}}^{T}x}}{1+\sum_{k=1}^{K-1}e^{\theta _{k}^{T}x}}
\\=\sum_{i=1}^{m}(1-\alpha (y_{i}))\theta _{y_{i}}^{T}x-log(1+\sum_{k=1}^{K-1}e^{\theta _{k}^{T}}x)$$
同样的我们得到损失函数：
$$J(\theta ,x)=-\frac{1}{m}l(\theta ,x)$$
$\theta$更新过程：
$$\theta _{j}:=\theta _{j}-\alpha\frac{\delta }{\delta _{\theta _{j}}}J(\theta,x )$$
对$\theta$求偏导得到梯度：
$$G_{kj}(\theta ,x)=-\frac{1}{m}\frac{\delta l(\theta ,x)}{\delta \theta _{kj}}=-\frac{1}{m}(\sum_{i=1}^{m}(1-\alpha (y_{i}))x_{ij}\delta _{k,y_{i}}
-\frac{e^{\theta ^{T}x_{i}}}{1+e^{\theta ^{T}x_{i}}}x_{ij})$$
其中k表示因变量，j表示特征数量，i表示实验数。
spark源码注释中，稍稍不一样，$l(w,x)$乘以了-1，其实与我们上边推导的$-\frac{1}{m}$意义一样。我们来看看spark的推导过程。
$$ P(y=0|x, w) = 1 / (1 + \sum_i^{K-1} \exp(x w_i))
 \\P(y=1|x, w) = exp(x w_1) / (1 + \sum_i^{K-1} \exp(x w_i))
  \\ ...
 \\P(y=K-1|x, w) = exp(x w_{K-1}) / (1 + \sum_i^{K-1} \exp(x w_i)$$
 取对数：
$$ l(w, x) = -log P(y|x, w) = -\alpha(y) log P(y=0|x, w) - (1-\alpha(y)) log P(y|x, w)
  \\       = log(1 + \sum_i^{K-1}\exp(x w_i)) - (1-\alpha(y)) x w_{y-1}
 \\        = log(1 + \sum_i^{K-1}\exp(margins_i)) - (1-\alpha(y)) margins_{y-1}$$
 其中：
$$\alpha(i) = 1 \quad if \quad i != 0,
     \\\alpha(i) = 0 \quad if \quad i == 0,
      \\ margins_i = x w_i.$$
      求偏导：
      $$ \frac{\partial l(w, x)}{\partial w_{ij}}
   = (\exp(x w_i) / (1 + \sum_k^{K-1} \exp(x w_k)) - (1-\alpha(y)\delta_{y, i+1})) * x_j
   = multiplier_i * x_j$$
   其中：
    $$\delta_{i, j} = 1 \quad if  \quad i == j,
      \\ \delta_{i, j} = 0  \quad if  \quad i != j,
      \\ multiplier =
         \exp(margins_i) / (1 + \sum_k^{K-1} \exp(margins_i)) - (1-\alpha(y)\delta_{y, i+1})$$
为了不让数值溢出,xw项减了maxMargin，$l(w,x)$改写为：
$$ l(w, x) = log(1 + \sum_i^{K-1}\exp(margins_i)) - (1-\alpha(y)) margins_{y-1}
 \\        = log(\exp(-maxMargin) + \sum_i^{K-1}\exp(margins_i - maxMargin)) + maxMargin
           - (1-\alpha(y)) margins_{y-1}
\\         = log(1 + sum) + maxMargin - (1-\alpha(y)) margins_{y-1}$$
其中：
$$sum = \exp(-maxMargin) + \sum_i^{K-1}\exp(margins_i - maxMargin) - 1$$
 而multiplier可以表示为：
 $$ multiplier = \exp(margins_i) / (1 + \sum_k^{K-1} \exp(margins_i)) - (1-\alpha(y)\delta_{y, i+1})
   \\         = \exp(margins_i - maxMargin) / (1 + sum) - (1-\alpha(y)\delta_{y, i+1})$$
## spark源码
先看实例代码:
```  scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

// Load training data in LIBSVM format.
//样例数据格式:
//1 特征id1:值id1 特征id2:值id2 ...
//0 特征id1:值id3 特征id4:值id4 ...
//特征和特征对应的值都使用数值一一标示了
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
// 官方样例分类数设置为10,但样例数据因变量是0和1,所以这里应该时设置错了.
// 梯度下降法每次迭代都会变量整个样本集,推荐使用拟牛顿法LBFGS,后续文章中继续介绍
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(10)
  .run(training)

// Compute raw scores on the test set.
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)

// Save and load model
// 输出是个模型,就是一个向量$\theta$,带入概率分布函数求得类型的概率
model.save(sc, "myModelPath")
val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
```
随机梯度下降调用:
``` scala
 /**
   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
   * gradient descent are initialized using the initial weights provided.
   * NOTE: Labels used in Logistic Regression should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.迭代次数
   * @param stepSize Step size to be used for each iteration of gradient descent.步长
   * @param miniBatchFraction Fraction of data to be used per iteration.用于模型预估数据的比例
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to the number of features in the data.初始化权重
   */
  @Since("1.0.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double,
      initialWeights: Vector): LogisticRegressionModel = {
    new LogisticRegressionWithSGD(stepSize, numIterat2 Aions, 0.0, miniBatchFraction)
      .run(input, initialWeights)
  }
```
LogisticRegressionWithLBFGS和LogisticRegressionWithSGD都继承于GeneralizedLinearModel,它的run方法:
```  scala
 def run(input: RDD[LabeledPoint], initialWeights: Vector): M = {

    if (numFeatures < 0) {
    // 输入的特征数等于第一行特征个数.
      numFeatures = input.map(_.features.size).first()
    }
    // 输入数据的存储类别.
    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Check the data properties before running the optimizer
    if (validateData && !validators.forall(func => func(input))) {
      throw new SparkException("Input validation failed.")
    }

    /**
     * Scaling columns to unit variance as a heuristic to reduce the condition number:
     *
     * During the optimization process, the convergence (rate) depends on the condition number of
     * the training dataset. Scaling the variables often reduces this condition number
     * heuristically, thus improving the convergence rate. Without reducing the condition number,
     * some training datasets mixing the columns with different scales may not be able to converge.
     *
     * GLMNET and LIBSVM packages perform the scaling to reduce the condition number, and return
     * the weights in the original scale.
     * See page 9 in http://cran.r-project.org/web/packages/glmnet/glmnet.pdf
     *
     * Here, if useFeatureScaling is enabled, we will standardize the training features by dividing
     * the variance of each column (without subtracting the mean), and train the model in the
     * scaled space. Then we transform the coefficients from the scaled space to the original scale
     * as GLMNET and LIBSVM do.
     *通过每一列除以这一列的标准差,将数据标准化.LBFGS算法中可以启用.
     * Currently, it's only enabled in LogisticRegressionWithLBFGS
     */
    val scaler = if (useFeatureScaling) {
      new StandardScaler(withStd = true, withMean = false).fit(input.map(_.features))
    } else {
      null
    }

    // Prepend an extra variable consisting of all 1.0's for the intercept.
    // TODO: Apply feature scaling to the weight vector instead of input data.
    // 默认是不加入截距项的
    val data =
      if (addIntercept) {
        if (useFeatureScaling) {
          input.map(lp => (lp.label, appendBias(scaler.transform(lp.features)))).cache()
        } else {
          input.map(lp => (lp.label, appendBias(lp.features))).cache()
        }
      } else {
        if (useFeatureScaling) {
          input.map(lp => (lp.label, scaler.transform(lp.features))).cache()
        } else {
          input.map(lp => (lp.label, lp.features))
        }
      }

    /**
     * TODO: For better convergence, in logistic regression, the intercepts should be computed
     * from the prior probability distribution of the outcomes; for linear regression,
     * the intercept should be set as the average of response.
     */
    val initialWeightsWithIntercept = if (addIntercept && numOfLinearPredictor == 1) {
      appendBias(initialWeights)
    } else {
      /** If `numOfLinearPredictor > 1`, initialWeights already contains intercepts. */
      initialWeights
    }
    //SGD 或者 LBFGS算法
    val weightsWithIntercept = optimizer.optimize(data, initialWeightsWithIntercept)
    ...
    createModel(weights, intercept)
  }
```
梯度下降SGD实现:
```scala
def runMiniBatchSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Vector,
      convergenceTol: Double): (Vector, Array[Double]) = {
    ...
      //不知道此数组干啥用的
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    ...
    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            // 返回损失loss,没看明白为何要算loss,及loss为何这么算log(1 + exp(margin))
            // 主要目的时计算c._1梯度向量
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      if (miniBatchSize > 0) {
        /**
         * lossSum is computed using the weights from the previous iteration
         * and regVal is the regularization value computed in the previous iteration as well.
         */
        stochasticLossHistory.append(lossSum / miniBatchSize + regVal)
        // 正则化
        val update = updater.compute(
          weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
          stepSize, i, regParam)
        weights = update._1
        regVal = update._2

        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)

  }
```
combOp是$\theta$的更新过程中的$\sum$过程.在二元逻辑回归情况下:
``` scala
case 2 =>
        /**
         * For Binary Logistic Regression.
         *
         * Although the loss and gradient calculation for multinomial one is more generalized,
         * and multinomial one can also be used in binary case, we still implement a specialized
         * binary version for performance reason.
         */
        val margin = -1.0 * dot(data, weights)
        val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
        axpy(multiplier, data, cumGradient)
        if (label > 0) {
          // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
          MLUtils.log1pExp(margin)
        } else {
          MLUtils.log1pExp(margin) - margin
        }
```
margin 就是($-\theta ^{T}x$),而multiplier就是$h_{\theta }(x_{i})-y_{i}$.axpy方法就是$(h_{\theta }(x_{i})-y_{i}))x_{i}$.

## 参考
1. http://blog.csdn.net/pakko/article/details/37878837
2. http://spark.apache.org/docs/latest/mllib-linear-methods.html
3. http://www.slideshare.net/dbtsai/2014-0620-mlor-36132297
4. https://www.codecogs.com/latex/eqneditor.php?lang=zh-cn(数学公式编辑器)
[1]: http://math.stackexchange.com/
[2]: https://github.com/jmcmanus/pagedown-extra
[3]: http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
[4]: http://bramp.github.io/js-sequence-diagrams/
[5]: http://adrai.github.io/flowchart.js/
[6]: https://github.com/benweet/stackedit
