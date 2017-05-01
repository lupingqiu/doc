# pandas基础知识

1. 基本使用
```python
  import pandas as pd
  import numpy as np

  s = pd.Series([1,3,6,np.nan,44,1])
  print s

  dates = pd.date_range('20160101',periods = 6)

  #行为dates 列为columns里的值，默认是0,1,2,3...
  df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])

  df.index
  df.columns
  df.calues
  df.describe()

  df.sort_index(axis=1,ascending=False)#对列索引排序 0就是行索引排序

  df.sort_values(bv='A_name')
```

2. 选择数据 设置值 处理空值
```python
  import pandas as pd
  import numpy as np

  dates = pd.date_range('20160101',periods = 6)
  df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d'])
  print(df['a'],df.a)
  print(df[0:3],df['20160101':'20160102']

  # select by label: loc
  print(df.loc['20160102'])
  print(df.loc[:,['a','b']])

  # select by position: iloc
  print(df.iloc[3])
  print(df.iloc[3:5,1:3])

  # mixed selection: ix label和index混合筛选
  print(df.ix[:3,['a','c']])

  # bolean indexing
  print(df.a < 8)
  print(df[df.a < 8])

  # 设置值
  df.iloc[2,2] = 14
  df[df.a>4] = 0#其他列符合条件的变成0
  df.a[df.a>4] = 0#a列符合条件的变成0

  df['f']= np.nan#加一空列
  df['e']= pd.Series([1,2,3,4,5,6],index=pd.date_range('20160101',periods=6))

  # 处理空值
  df.iloc[0,1] = np.nan
  df.iloc[1,2] = np.nan
  df.dropna(axis=0,how='any')#0丢掉行，1丢掉列,'any'表示只要出现就丢掉,'all'表示都是nan时丢掉
  df.fillna(value=0)#nan添值
  df.isnull()#表示是否缺失数据，所有原子
  print(np.any(df.isnull()) == True)#表示这一堆值里是否有丢失的数据
```

3. 导入 导出
```python
  import pandas as pd

  data = pd.read_csv('file_path')#data则为dataframe，index为0,1,2...
  data.to_pickle('save_file_path')
```

4. 合并concat merge
```python
  import pandas as pd
  import numpy as np

  # concatenating
  df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
  df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
  df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])

  res = pd.concat([df1,df2,df3],axis=0)#0 行合并，1 列合并
  #index变成了0,1,2,用ingore忽略
  res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)

  # join，inner(相同的部分) or outer（所有部分，nan填充）
  df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
  df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
  res = pd.concat([df1,df2],join='outer')#默认outer

  res2 = pd.concat([df1,df2],axis=1,join_axes=[df1.index])#只考虑df1的index
  res2 = pd.concat([df1,df2],axis=1)#记录所有的index，join不上的nan填充

  # append
  s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
  res = df1.append(df2,ignore_index=True)
  res = df1.append([df2,df3],ignore_index=True)
  res = df1.append(s1,ignore_index=True)

  # merge
  left = pd.DataFrame({'key':['k0','k1','k2','k3'],
                        'a':['a0','a1','a2','a3'],
                        'b':['b0','b1','b2','b3']})
  right = pd.DataFrame({'key':['k0','k1','k2','k3'],
                        'c':['c0','c1','c2','c3'],
                        'd':['d0','d1','d2','d3']})
  res = pd.merge(left,right,on='key')#单个key
  res = pd.merge(left,right,on=['key1','key2'])#默认how =‘inner’ outer left right
  # 参数indicator=True 显示left_only rght_only both

  # merge by index
  # 参数left_index=True right_index=True
  # suffixes=['_boyes','_girl']相同列名merge区分时可以使用后缀区分
```

5. plot图表可视化
```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt

  # plot data
  # Series线性
  data = pd.Series(np.random.randn(1000),index=np.arange(1000))
  data = data.cumsum()#逐个累加
  data.plot()
  plt.show()

  # DataFrame矩阵
  data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list('abcd'))
  data = data.cumsum()
  data.plot()
  plt.show()
  # plot methods: bar,hist,box,kde,area,scatter,hexbin,pie
  ax = data.plot.scatter(x='a',y='b',color='DarkBlue',label='class1')
  data.plot.scatter(x='a',y='c',color='DarkGreen',label='class2',ax = ax)#ax图放在一张里
  plt.show()
```
