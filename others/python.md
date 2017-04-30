# python语言知识要点


## 基础

1. 函数当中的有无默认值的参数必须有默认值的在后边，例如：
    ```
    def fun(name, age, school='三中')
    ```

2. 在函数中如果要使用全局变量需要声明global，例如：
    ```
    def fun():
        global a
        a = 100
    print a
    ```

3. file readline 逐行读取，readlines读取全部

4. 输入
    ```
    a = input("Please give a number:") #return a string
    print("your input is :",a)
    ```
5. 删除字典里的元素
  ```
  d = {1:'a', 'x':2}
  del d['x']
  ```

6. 异常处理
  ```
  try:
    do something
  except Exception as e:
    do something
  else：
    do otherthing
  ```

7. zip lambda map
  ```
  #合并
  a=[1,2,3]
  b=[4,5,6]
  list(zip(a,b)) #[(1,4),(2,5),(3,6)]

  def fun(x,y):
    return x+y
  #等同于 lambda可以定义比较简单的方法
  fun = lambda x,y: x+y

  #map能把方法和参数绑定执行
  list(map(fun,[1],[3]))
  list(map(fun,[1,3],[3,5]))
  ```

8. 复制copy 浅复制copy 深复制deepcopy
  ```
  import copy
  a=[1,2,3]
  b=a
  id(a) == id(b)
  #如果改变b的值，a也同样改变，即a和b有相同的物理地址
  #浅复制时怎么样的呢：
  c=copy.copy(a)
  id(c)!=id(a)

  a=[1,2,[3,4]]
  b= copy.copy(a)
  a[2][0]=333
  id(a[2])==id(b[2])
  #即copy的第一层是不同地址空间，但第二层或者更深层地址空间是一样的。
  #需要使用deepcopy,地址空间会完全不一样的，完完全全拷贝内容
  b=copy.deepcopy(a)
  ```

9. 多线程
  ```
  import threading
  import time

  def job():
    for i in range(10):
      time.sleep(0.1)
      print('1')

  def job2():
    for i in range(10):
      time.sleep(0.1)
      print('2')

  def main():
    add_thread = threading.Thread(target=job,name="t1")#添加线程
    add_thread2 = threading.Thread(target=job2,name="t2")#添加线程

    add_thread.start()#启动
    add_thread2.start()

    add_thread.join()#等待其他线程运行完成后，再往下执行主线程
    add_thread2.join()

    print(threading.active_count())#激活线程数
    print(threading.enumerate())#线程名称
    print(threading.current_thread())#当前线程

  if __name__==‘__main__’:
    main()
  ```
  ```
  #queue用于传递线程里的值
  import threading
  import time
  from queue import Queue

  def job(l,q):
    for i in range(len(l))
      l[i] = l[i]**2
    q.put(l)

  def multithreading():
    q = Queue()
    threads = []
    data = [[1,2,3],[4,5,6],[7,8,9],[1,1,2]]
    for i in range(4):
      t = threading.Thread(target=job,args=(data[i],q),name="t"+i)
      t.start()
      threads.append(t)
    for thread in threads:
      thread.join()
    results = []
    for _ in range(4):
      results.append(q.get())
    print(results)
  ```
  因为GIL机制，其实多线程不一定效率高，因为是多个线程来回切换运行，每个时刻只一个线程在运行。
  多线程尽量每个线程做不同种类的事情，才能达到效率，例如当一个线程在io时，另外一个线程就可以跑。
  如果是处理相同事，比如对一大块数据并发进行相同处理，性能不会提升很高。这里需要多核的处理。

  ```
  #lock
  import threading

  def job1():
    global A,lock
    lock.acquire()
    for i in range(10):
      A +=1
      print("job1",A)
    lock.release()

  def job2():
    global A,lock
    lock.acquire()
    for i in range(10):
      A +=10
      print("job2",A)
    lock.release()

  if __name__=='__main__':
    lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

  ```
10. 多核,多进程
  ```
  import multiprocessing as mp
  import threading as td

  def job(a,b):
    print("aaaa")

  if __name__=='__main__':
    t1 = td.Tread(target=job,args=(1,2))
    p1 = mp.Process(targt=job,args=(1,2))
    t.start()
    p1.start()
    t1.join()
    p1.join()
  ```
  跟多线程一样，target方法里也不能有return方法，职能通过queue来传递数据。
  ```
  import multiprocessing as mp

  def job(q):
    res=0
    for in range(1000):
      res+=i+i**2+i**3
    q.put(res)

  if __name__=='__main__':
    q= mp.Queue()#稍微不一样，得使用mp自己的Queue
    p1 = mp.Process(targt=job,args=(q,))#一个参数时后边必须还有逗号
    p2 = mp.Process(targt=job,args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print(res1+res2)

  ```
  如上计算耗时对比，代码省略：
  multicore < normal < multithread

  进程池
  ```
  import multiprocessing as mp

  def job(x):
    return x*x  #可以返回了

  def muticore():
    pool = mp.Pool()#默认是机器所有核，或者指定processes=2
    res = pool.map(job,range(10))#方法和参数map在一起
    print(res)
    res = pool.apply_asyc(job,(2,))#只能传入一个参数
    print(res.get())#如果要多个怎么办
    multi_res=[pool.apply_asyc(job,(i,)) for i in range(10)]
    print([res.get() fro res in multi_res])
    #相对来说map使用更方便

  if __name__=='__main__':
    muticore()

  ```
  共享内存
  如果使用global变量，在多进程当中时行不通的，不能交流这个全局变量。
  可以使用shared memory
  ```
  import multiprocessing as mp

  value = mp.Value('i',0)#i表示整数，d表示double，f表示float
  array = mp.Array('i',[1,2,3])#一维列表
  ```
  锁在共享内存当中的运用
  ```
  import multiprocessing as mp
  import time

  def job(v,num):
    for _ in range(10):
      time.sleep(0.1)
      v.value+=num
      print(v.value)

  def multicore():
    v = mp.Value('i',0)
    p1=mp.Process(target=job,arg=(v,1))
    p2=mp.Process(target=job,arg=(v,3))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
  ```
  ```
  import multiprocessing as mp
  import time

  def job(v,num,l):
    l.acquire()
    for _ in range(10):
      time.sleep(0.1)
      v.value+=num
      print(v.value)
    l.release()

  def multicore():
    l = mp.Lock()
    v = mp.Value('i',0)
    p1=mp.Process(target=job,arg=(v,1,l))
    p2=mp.Process(target=job,arg=(v,3,l))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
  ```

11. tkinter窗口

12. pickle
  ```
  import pickle
  data = [1,2,3]
  file = open("pickle.pickle","wb")
  pickle.dump(data,file) #二进制
  file.close()

  file2 = open("pickle.pickle","rb")
  mydata = pickle.load(file2)
  file2.close()

  # with语句不用close文件
  with open("pickle.pickle","rb") as file3:
      mydata = pickle.load(file3)
  ```

14. set集合
  ```
  l = [1,2,3,1,2]
  print(set(l))
  s="Welcome to shanghai"
  print(set(s))

  myset = set(s)
  myset.add('p')
  myset.remove('p')#没有会报错，可以使用discard
  myset.clear()

  set1.difference(set2)#set1中有set2没有的
  set1.intersection(set2)#交集
  ```
