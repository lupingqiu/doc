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
