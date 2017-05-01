- 范数 L0 L1 L2 核范数 概念和应用

  - 概念理解和应用 http://blog.csdn.net/zouxy09/article/details/24971995
  - 定义 http://blog.csdn.net/xlinsist/article/details/51213593

- PCA

  - PCA（Principal Component Analysis）是一种常用的数据分析方法。PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。
  - PCA的数学原理 http://blog.codinglabs.org/articles/pca-tutorial.html

- 最大似然函数、EM算法

  - 求最大似然函数估计值的一般步骤：
    1. 写出似然函数；
    2. 对似然函数取对数，并整理；
    3. 求导数，令导数为0，得到似然方程；
    4. 解似然方程，得到的参数即为所求
  - 从最大似然函数到EM算法 http://blog.csdn.net/zouxy09/article/details/8537620
  - EM算法：期望最大算法是一种从不完全数据或有数据丢失的数据集（存在隐含变量）中求解概率模型参数的最大似然估计方法。
  - EM http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html

- 逻辑回归

  - 逻辑回归梯度下降法详解 http://blog.csdn.net/lookqlp/article/details/51161640

- 感知器算法？


- 深度学习word2vec

  - http://suanfazu.com/t/shen-du-xue-xi-word2vec-bi-ji/192  
  - http://latex.codecogs.com/eqneditor/editor.php

- 深度学习框架

  - tensorflow
     - http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/overview.html
     - https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/TOC.md
     - https://github.com/nlintz/TensorFlow-Tutorials 例子  
     - 卷积
       1. http://www.zhihu.com/question/22298352
       2. http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution
     - 池化 http://deeplearning.stanford.edu/wiki/index.php/Pooling
  - paddle
     - https://github.com/baidu/paddle
     - http://www.paddlepaddle.org/doc_cn/demo/quick_start/index.html

- fine-tuning

    - caffe model zoo https://github.com/BVLC/caffe/wiki/Model-Zoo
    - convert caffe to tf https://github.com/ethereon/caffe-tensorflow/tree/8178f6914accf8e1358b47b68ae4364f4a4de41d
    - example https://github.com/joelthchao/tensorflow-finetune-flickr-style
    - tflearn finetuning https://github.com/tflearn/tflearn/blob/master/examples/basics/finetuning.py
- softmax回归

  - http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
  - 对于k类分类，是选择softmax分类器呢，还是k个独立的logic二元分类好？取决于类别之间是否互斥，如果互斥则softmax，如果不互斥（可能是其中的一类或者多类），则多个独立logic。


- 深度学习（多层神经网络）
  - http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C

  - http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B
  - 神经网络的沉浮 http://www.36dsj.com/archives/39775
  - 卷积神经网络 http://blog.csdn.net/stdcoutzyx/article/details/41596663
  - 参数调整 http://blog.csdn.net/han_xiaoyang/article/details/50521064
  - Neural Networks and Deep Learning http://neuralnetworksanddeeplearning.com/index.html


- 梯度下降

  - 梯度下降（批量），随机梯度下降，批量随机梯度下降，http://www.cnblogs.com/louyihang-loves-baiyan/p/5136447.html
  - 在样本数据量少的时候使用批量梯度下降（下降速度最快），而在数据量大的时候建议使用批量随机梯度下降。
  - http://www.cnblogs.com/LeftNotEasy/archive/2010/12/05/mathmatic_in_machine_learning_1_regression_and_gradient_descent.html

- LR vs. DT vs. SVM

  - http://blog.csdn.net/oliverkehl/article/details/50129999

- AUC and ROC

  - http://www.douban.com/note/284051363/ ROC（Receiver Operating Characteristic）曲线和AUC常被用来评价一个二值分类器（binary classifier）的优劣

    > True Positive （真正, TP）被模型预测为正的正样本；
    >
    > True Negative（真负 , TN）被模型预测为负的负样本 ；
    >
    > False Positive （假正, FP）被模型预测为正的负样本；
    >
    > False Negative（假负 , FN）被模型预测为负的正样本；
    >
    > True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）
    >    TPR = TP /（TP + FN）
    >    正样本预测结果数 / 正样本实际数
    >
    > True Negative Rate（真负率 , TNR）或特指度（specificity）
    >    TNR = TN /（TN + FP）
    >    负样本预测结果数 / 负样本实际数
    >
    > False Positive Rate （假正率, FPR）
    >    FPR = FP /（FP + TN）
    >    被预测为正的负样本结果数 /负样本实际数
    >
    > False Negative Rate（假负率 , FNR）
    >    FNR = FN /（TP + FN）
    >    被预测为负的正样本结果数 / 正样本实际数

- 零基础掌握极大似然估计

    - http://mp.weixin.qq.com/s/Zur3PgwtYvVs9ZTOKwTbYg

- 一文搞懂k邻近（knn）算法

    - http://mp.weixin.qq.com/s/mjkDl_6XUwF9L6GMpbY6Zg
    - http://mp.weixin.qq.com/s/mjkDl_6XUwF9L6GMpbY6Zg

- 带你搞懂朴素贝叶斯分类算法

    - http://mp.weixin.qq.com/s/dV0SQo1vaggXuKQCjHR9ew
    - http://mp.weixin.qq.com/s/dV0SQo1vaggXuKQCjHR9ew

- softmax函数以及相关求导过程

    - http://www.jianshu.com/p/ffa51250ba2e

- BP算法

    - http://blog.csdn.net/pennyliang/article/details/6695355 例子虽然有很多错误值，但可以大概了解反向传播过程
