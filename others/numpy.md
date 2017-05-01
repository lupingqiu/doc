# numpy基础知识

1. 属性
```python
  import numpy as np

  array = np.array([[1,2,3],[4,5,6]]，dtype=int)
  print(array.ndim)#维数
  print(array.shape)#形状
  print(array.size)#元素个数

  a = np.zeros((3,4))#全部为0的三行四列的矩阵
  a = np.ones((3,4))

  a= np.arange(10,20,2)#[10,12,14,16,18]

  a= np.arange(12).reshape((3,4))

  a = np.linspace(1,10,5)#线段 1到10分成5段
```

2. 运算
```python
  import numpy as np

  a = np.array([10,20,30,40])
  b = np.arange(4)
  c = a + b

  c = b**2
  c = np.sin(b)

  b<3#[True，True，True，False]

  #矩阵运算
  a = np.array([[1,1],[0,1]])
  b = np.arange(4).reshape((2,2))

  #逐个相乘
  c = a*b
  #矩阵相乘
  c_dot = np.dot(a,b)
  c_dot_2 = a.dot(b)

  a = np.random.random((2,4))
  np.sum(a)
  np.min(a)
  np.sum(a,axis=1)#axis=0表示列中sum，axis=1表示行中的sum

  a = np.arange(2,14).reshape((3,4))
  #最小值索引
  np.argmin(a)#0
  #平均值
  np.mean(a)
  a.mean()
  #中位数
  np.median(a)
  #逐步相入
  np.cumsum(a)#[2,5,9...]
  #逐行排序
  np.sort(a)
  #矩阵的转置
  np.transpose(a)
  a.T

  np.clip(a,5,9)#a中小于5的数变成5,大于9的变成9,其他的保持不变
```

3. 索引
```python
  import numpy as np

  a = np.arange(3,15).reshape((3,4))

  a[1][1]#第二行第二列
  a[1,1]#效果一样
  a[1,1:2]#第1列，从1到2

  for row in a:
    print(row)
  for column in a.T:
    print(column)

  print(a.flatten())
  for item in a.flat:#平铺后迭代器
    print(item)
```

4. array合并 分割
```python
  import numpy as np
  a = np.array([1,1,1])
  b = np.array([2,2,2])

  c = np.vstack((a,b)) #shape(2,3) 上下合并 vertical stack

  d = np.hstack((a,b)) #shape(6,) 左右合并 horizontal stack

  #因为a只有一维，a.T shape是不会变化的，需要
  a[:,np.newaixs] #shape(3,1)

  c = np.concatenate((a,a,b),axis=1)#1横向，0纵向合并

  #分割
  a = np.arange(12).reshape((3,4))

  np.split(a,2,axis=1)#0横向分割，1纵向分割
  np.array_split(a,2,axis=1)#不等项分割，第一个array最多

  np.vsplit(a,3)#纵向
  np.hsplit(a,2)#横向
```

5. copy & deep copy
```python
  import numpy as np

  a = np.array(4)
  b = a
  c = a
  d = b
  d is a#True,物理索引和数值都是一样

  b = a.copy() # deep copy 只复制了值，没有复制地址
```
