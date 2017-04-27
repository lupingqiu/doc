- spark-submit  /bin/spark-class org.apache.spark.deploy.SparkSubmit
- org.apache.spark.launcher.Main类启动org.apache.spark.deploy.SparkSubmit
- runMain mainMethod.invoke执行SparkWordCount中main方法
- SparkWordCount创建SparkContext
- org.apache.spark.SparkContext.createTaskScheduler(local local[n] standalone yarn-client yarn-cluster)
  local 资源调度LocalBackend 任务调度TaskShceduler->TaskSchedulerImpl
  local[N] 资源调度LocalBackend 任务调度TaskShceduler->TaskSchedulerImpl
  local[N, M] 同上
  Standalone 资源调度SparkDeploySchedulerBackend继承至CoarseGrainedSchedulerBackend 任务调度TaskShceduler->TaskSchedulerImpl
  local-cluster 资源调度SparkDeploySchedulerBackend 任务调度TaskShceduler->TaskSchedulerImpl
  yarn-cluster 资源调度YarnClusterSchedulerBackend 任务调度TaskShceduler->YarnClusterScheduler
  yarn-client 资源调度YarnClientSchedulerBackend 任务调度YarnScheduler为TaskSchedulerImpl的子类
  Mesos 精粒度资源调度CoarseMesosSchedulerBackend 细粒度资源调度MesosSchedulerBackend  任务调度TaskShceduler->TaskSchedulerImpl

- 任务提交如下：

  1. SparkWordCount main counts.saveAsTextFile

  2. org.apache.spark.rdd.RDD.saveAsTextFile

  3. org.apache.spark.rdd.PairRDDFunctions.saveAsHadoopFile  -> saveAsHadoopFile -> saveAsHadoopDataset

  4. org.apache.spark.SparkContext.runJob -> runJob -> runJob

  5. org.apache.spark.scheduler.DAGScheduler.runJob -> submitJob

  6. org.apache.spark.util.EventLoop.post(queue)

  7. org.apache.spark.scheduler.DAGScheduler DAGSchedulerEventProcessLoop.onReceive

  8. org.apache.spark.scheduler.DAGScheduler.handleJobSubmitted
     newResultStage创建新的stage  
     submitStage提交finalStage，该方法会提交所有关联的未提交的stage
     submitWaitingStages检查是否有等待或失败的Stage需要重新提交

  9. org.apache.spark.scheduler.DAGScheduler.submitStage如果有未提交的父Stage，则递归提交  
     submitMissingTasks使用 taskScheduler.submitTasks提交任务TaskSet
     Stage由一系列的tasks组成，这些task被封装成TaskSet

  10. org.apache.spark.scheduler.TaskSchedulerImpl.submitTasks
      创建TaskSetManager，TaskSetManager用于对TaskSet中的Task进行调度，包括跟踪Task的运行、Task失败重试等
      schedulableBuilder中添加TaskSetManager，用于完成所有TaskSet的调度，即整个Spark程序生成的DAG图对应Stage的TaskSet调度
      SchedulerBackend backend.reviveOffers()为Task分配运行资源

  11. 资源调度SchedulerBackend实现类很多
      SparkDeploySchedulerBackend CoarseGrainedSchedulerBackend  以此为例：调用 driverEndpoint.send(ReviveOffers)
      LocalBackend
      YarnClusterSchedulerBackend
      YarnClientSchedulerBackend
      MesosSchedulerBackend

  12. org.apache.spark.rpc.RpcEndpointRef 实现org.apache.spark.rpc.akka AkkaRpcEndpointRef.send actorRef ! AkkaMessage(message, false) org.apache.spark.scheduler

  13. org.apache.spark.scheduler.cluster.CoarseGrainedSchedulerBackend class DriverEndpoint.receive接受消息makeOffers()
      launchTasks(scheduler.resourceOffers(workOffers))
      launchTasks方法后，Task的执行便在Worker节点上运行了，至此完成Task的提交
      resourceOffers随机分配worker
      launchTasks executorData.executorEndpoint.send Worker节点上的CoarseGrainedExecutorBackend对象将接受LaunchTask消息，在Worker节点的Executor上启动Task的执行

  14. CoarseGrainedExecutorBackend standalone模式  receive方法接收消息
      LaunchTask方法处理driver端发来的任务
      executor.launchTask启动任务执行

  15. org.apache.spark.executor.Executor launchTask创建TaskRunner，TaskRunner是一个线程，线程池执行TaskRunner线程，run方法中调用task.run执行任务
      小结
      1 调用Driver端org.apache.spark.scheduler.cluster.CoarseGrainedSchedulerBackend中的launchTasks
      2 调用Worker端的org.apache.spark.executor.CoarseGrainedExecutorBackend.launchTask    
      3 执行org.apache.spark.executor.TaskRunner线程中的run方法
      4 调用org.apache.spark.scheduler.Task.run方法
      5 调用org.apache.spark.scheduler.ResultTask.runTask方法
      6 调用org.apache.spark.rdd.RDD.iterator方法

16. org.apache.spark.executor.TaskRunner中会设置运行状态。CoarseGrainedSchedulerBackend的DriverEndpoint.receive接受TaskRunner发来的status信息，调用 TaskSchedulerImpl 的statusUpdate方法，如果成功，org.apache.spark.scheduler.taskResultGetter.enqueueSuccessfulTask

17. org.apache.spark.scheduler.TaskSchedulerImpl.handleSuccessfulTask  TaskSetManager.handleSuccessfulTask

18. 调用DagScheduler的taskEnded方法  sched.dagScheduler.taskEnded

19. 调用DAGSchedulerEventProcessLoop的post方法将CompletionEvent提交到事件队列中，交由eventThread进行处理，onReceive方法将处理该事件.   doOnReceive ->   dagScheduler.handleTaskCompletion
    小结
    1 org.apache.Spark.executor.TaskRunner.statusUpdate方法
    2 org.apache.spark.executor.CoarseGrainedExecutorBackend.statusUpdate方法
    3 org.apache.spark.scheduler.cluster.CoarseGrainedSchedulerBackend#DriverEndpoint.recieve方法，DriverEndPoint是内部类
    4 org.apache.spark.scheduler.TaskSchedulerImpl中的statusUpdate方法
    5 org.apache.spark.scheduler.TaskResultGetter.enqueueSuccessfulTask方法
    6 org.apache.spark.scheduler.DAGScheduler.handleTaskCompletion方法

资源调度类与子类

| Root | 第一层 | 第二层 | 第三层 |
| ----|----|----|----|
| SchedulerBackend | CoarseGrainedSchedulerBackend | YarnSchedulerBackend        | YarnClientSchedulerBackend  |
|                  |                               |                             | YarnClusterSchedulerBackend |
|                  |                               | CoarseMesosSchedulerBackend |                             |
|                  |                               | SparkDeploySchedulerBackend |                             |
|                  |                               | SimrSchedulerBackend        |                             |
|                  | LocalBackend                  |                             |                             |
|                  | MesosSchedulerBackend         |                             |            null             |

任务调度类与子类

| Root | 第一层 | 第二层 | 第三层 |
| ---- | ---- | -------|
| TaskScheduler | TaskSchedulerImpl | YarnScheduler| YarnClusterScheduler |
