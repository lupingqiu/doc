# scala 基础知识

## 基础
  1. 内容比较
    scala> val x="Hello"
    x: String = Hello

    scala> val y="Hello"
    y: String = Hello

    //Scala中的对象比较不同于Java中的对象比较
    //Scala基于内容比较，而java中比较的是引用，进行内容比较时须定义比较方法
    scala> x==y
    res36: Boolean = true

  2. if else
    val x= if("111" == "222") 1 else 0

  3. while
    var line = ""
    while ((line = readLine()) != "") // 在Scala中不能这么用，因为Scala中的赋值操作返回的值是Unit，而""是String类型，不能进行比较，这是函数式编程语言特有的特点
    println("Read: "+ line)

  4. for
    val filesHere = (new java.io.File(".")).listFiles
    for (file <- filesHere)
      println(file)
    程序中的<-被称生成器（generator），在执行过程中，集合filesHere中（Array[File])的元素将依次赋给file，file类型为File，打印时调用其toString方法将文件名称打印出来。

    for (i <- 0 to filesHere.length - 1)
      println(filesHere(i))

    for(i <- 1 to 5) //包含5
    for(i <- 1 until 5) //不包含5
    for (file <- filesHere if file.isFile;if file.getName.endsWith(".scala")) //条件过滤，分号分开
      println(file)

    // 双层循环
    for (
    file <- filesHere
    if file.getName.endsWith(".scala");
    line <- fileLines(file) //上面if过滤后的所有file，生成line，针对line执行循环
    if line.trim.matches(pattern)
    ) println(file +": "+ line.trim)

    // 返回结果，组成Array
    def scalaFiles =
      for {
      file <- filesHere
      if file.getName.endsWith(".scala")
      } yield file

    5. Array
      val strArray=new Array[String](10) //这意味着strArray不能被改变，但数组内容是可以改变的
      strArray(0) = "hello"

      val strArrayVar=ArrayBuffer[String]() //可变的ArrayBuffer
      strArrayVar+="Hello"
      strArrayVar+=("World","Programmer") //+=意思是在尾部添加元素/多个元素的集合

      strArrayVar++=Array("Wllcome","To","XueTuWuYou") /++=用于向数组中追加内容，++=右侧可以是任何集合 例如list
      strArrayVar++=List("Wellcome","To","XueTuWuYou")

      for(i <- 0 to intArrayVar.length-1) println("Array Element: " +intArrayVar(i)) //遍历

      var intArrayVar2=for(i <- intArrayVar) yield i*2 // 转化

    6. List
      val fruit=List("Apple","Banana","Orange")
      List一但创建，其值不能被改变
      fruit(0) = "apple" // 报错

      //采用::及Nil进行列表构建
      val nums = 1 :: (2 :: (3 :: (4 :: Nil)))
      //由于::操作符的优先级是从右往左的，因此上一条语句等同于下面这条语句
      val nums=1::2::3::4::Nil
      // Nil是一不可变的空的List()

    7. Map
       val studentInfo=Map("john" -> 21, "stephen" -> 22,"lucy" -> 20) // immutable
       val studentInfoMutable=scala.collection.mutable.Map("john" -> 21, "stephen" -> 22,"lucy" -> 20) // 可变
       for( i <- studentInfoMutable ) println(i) //遍历

       studentInfoMutable.foreach(e=>
        {val (k,v)=e; println(k+":"+v)}
       )
       ```
       studentInfoMutable.foreach(e=> println(e._1 + ':' + e._2))
       ```
       val xMap=new scala.collection.mutable.HashMap[String,Int]() // java类似

    8. Option,None,Some类型
      Option、None、Some是scala中定义的类型，它们在scala语言中十分常用，因此这三个类型非学重要。
      None、Some是Option的子类，它主要解决值为null的问题，在Java语言中，对于定义好的HashMap，如果get方法中传入的键不存在，方法会返回null，在编写代码的时候对于null的这种情况通常需要特殊处理，然而在实际中经常会忘记，因此它很容易引起 NullPointerException异常。在Scala语言中通过Option、None、Some这三个类来避免这样的问题，这样做有几个好处，首先是代码可读性更强，当看到Option时，我们自然而然就知道它的值是可选的，然后变量是Option，比如Option[String]的时候，直接使用String的话，编译直接通不过。
      scala> xMap.get("spark")
      res19: Option[Int] = Some(1)

      def show(x:Option[Int]) =x match{
       case Some(s) => s
       case None => '????'
      }
      show(xMap.get("spark"))

    9. 元组
      元组则是不同类型值的聚集
      var tuple=("Hello","China",1)
      val (first, second, third)=tuple

    10. Queue
      var queue=scala.collection.immutable.Queue(1,2,3)
      queue.dequeue // 出队 (Int, scala.collection.immutable.Queue[Int]) = (1,Queue(2, 3))
      queue.enqueue(4) //入队 scala.collection.immutable.Queue[Int] = Queue(1, 2, 3, 4)

      var queue=scala.collection.mutable.Queue(1,2,3,4,5)
      queue += 5 // 入队
      queue ++= List(6,7,8)//集合方式

## 函数

    1. 函数字面量（值函数）
      val increase=(x:Int)=>x+1  //=>左侧的表示输入，右侧表示转换操作,多个语句则使用{}

## implicit

  1. 隐式参数
  2. 隐式转换类型
  3. 隐式调用函数
  http://blog.csdn.net/jameshadoop/article/details/52337949
