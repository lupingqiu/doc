#django开发web小项目
##引言
近期使用django开发了个小web项目,使用了bootstrap模板样式,以及echart图形控件,在这里对它们的基本使用做一些小总结.
##Why
django是由python开发的开源网站框架,可以快速的搭建高性能的web站点,个人觉得特别合适于公司级的小项目,Cloudera的hue就是基于django开发的.
bootstrap具有强大的模板样式,拿来即用,界面简洁,使用方便.
echart百度开源的良心产品,图形界面全面,且清爽,炫酷,官方帮助文档详细.
##版本
以我当前前的开发环境为例:
- django 1.8.3
- bootstrap 3.3.5
- python 2.7.3
*注意:django对于版本要求比较高,可能不同版本的django不兼容,尽量保证相同的项目都使用开发时的版本,不然可能出现"意外"的问题;另外bootstrap不同版本样式也不太一样.*

##How
###django
####安装
建议使用pip安装或者apt-get安装,源码包install安装后,若要卸载麻烦,直接删除python包的django路径可能不干净,源码包安装也是挺不错的.
```
sudo apt-get install python-pip
sudo pip install Django==1.8.3
```
*注意:有些系统默认是低版本的python例如1.6.X,当升级到1.7.X后,pip不好用,同样需要升级pip.*
####常用的命令
1. 新建项目
`python /usr/local/bin/django-admin.py/django-admin.py startproject project-name`
2. 新建app
`python manage.py startapp app-name`
3. 同步数据库
`python manage.py syncdb`
*注意:在models.py新增了类时,会自动在数据库中创建相应的表*
4. 启动开发服务器
```
python manage.py runserver
#指定端口
python manage.py runserver 8001
```
*注意:一般发布django工程,需要借助apache或者ngnix,个人觉得内部使用,且访问量并发量小,runserver方式启动和使用完全够用,用起来也挺稳定*
5. 创建超级管理员
`python manage.py createsuperuser`
*注意:django自带用户管理和权限管理,创建的用户可以在login模块中直接使用.*
6. 登陆配置好的数据库
`python manage.py dbshell`

####常用的几个文件
1. urls.py
网址的入口,关联到相同目录下views.py的对应方法,不同版本的urls.py貌似有些区别,使用时参见对应的官方文档.
2. views.py
处理urls.py对应的请求,然后渲染templates的页面.
3. models.py
于数据库相关,一般此文件中的类会与数据库表一一对应(models.Model).上边提到过同步数据库,同时views.py的方法中一般调用models.py来操作数据库,无非就是增删改查.
4. settings.py
django的配置文件,一个工程只有一个,配置app,静态文件的位置,session设置,日志设置等等.

####目录结构
```
project-name
    project-name
        urls.py
        views.py
        models.py
        settings.py
        ......
    app1
        templates
            htmls
        urls.py
        views.py
        models.py
        ......
    app2
    ......
    templates
        htmls
        js
    static
        ......
```
####登陆
网上有很多资料,不多赘述,注意几点.
1. 返回login.html的views.py方法
例如
`return render_to_response('login.html',context_instance=RequestContext(request))`
直接返回return render_to_response('login.html') csrf支持会有问题,必须返回原有的request信息.
2. session控制
```
SESSION_COOKIE_AGE = 60*5#5分钟超时
SESSION_EXPIRE_AT_BROWSER_CLOSE = True#关闭浏览器超时
```
3. 登陆方法
form提交的views.py方法不能为login(request),与django模块的方法名冲突,修改成其他名称.

###bootstrap
下载bootstrap压缩包,放在静态文件夹内,在html中引用即可.
```
<script src="{% static 'bootstrap/js/bootstrap.min.js' %}"></script>
<link href="{% static 'bootstrap/css/bootstrap.min.css' %}" rel="stylesheet">
```
###echart
下载echart js包,放在静态文件夹中,在html中引用即可.
`<script src="{% static 'js/echarts/echarts.js' %}"></script>`
####后台返回值
以线性趋势图为例:
1. legend
图形界面上方选择的项,可要也可不要.
2. x轴series
一般是个list.
3. y轴series
一般是个dict,key为对应的legend值,value是个list,个数与x轴series按顺序一一对应.

####example
```
<script type="text/javascript">
    require.config({
        paths: {
            echarts: '/static/js/echarts'
        }
    });
    // 使用
    require(
        [
            'echarts',
            'echarts/chart/line' , // 折线图
            'echarts/chart/pie'
        ],
        function (ec) {
            // 基于准备好的dom，初始化echarts图表
            var myChart = ec.init(document.getElementById('main_chart1'));
            var option = {
                title : { text: 'Jobs Monitor', x:'center'},
                tooltip: {
                    show: 'false'
                },
                color: [ '#ff7f50', '#da70d6', '#32cd32', '#6495ed', '#ff69b4', '#ba55d3', '#cd5c5c', '#ffa500', '#40e0d0'],
                legend: {
                    x: 80,
                    y: 30,
                    data:[{% for leg in legend_list %}{% if leg %}'{{leg}}'{% endif %}{% if not forloop.last %},{% endif %}{% endfor %}]
                },
                xAxis : [
                    {
                        type : 'category',
                        axisLabel: {interval: 'auto', rotate:35},
                        data : [{% for day in series %}'{{day}}'{% if not forloop.last %},{% endif %}{% endfor %}]
                    }
                ],
                yAxis : [{
                            type : 'value',
                            axisLabel:{formatter: '{value}'}

                        }],
                series : [
                        {% for key, value in out_line_dic.items %}
                          {% if key != None and value != None%}
                          {
                            "name": "{{key}}",
                            "type": "line",
                            "smooth":true,
                            "data": [{% for item_data in value %}{% if item_data %}{{item_data}}{% else %}0{% endif %}{% if not forloop.last %},{% endif %}{% endfor %}],
                            "markLine":{data:[{type:"average",name:"avg"}]},
                            itemStyle:{
                                normal:{
                                      label:{
                                        show: true,
                                      },
                                      labelLine :{show:true}
                                    }
                                },
                            markPoint : {
                                data : [
                                    {type : 'max', name: '最大值'},
                                    {type : 'min', name: '最小值'}
                                ]
                            },
                          }
                          {% if not forloop.last %},{% endif %}
                          {% endif %}
                        {% endfor %}
                    ]
            };
            // 为echarts对象加载数据
            myChart.setOption(option);
        })
</script>
```
##参考资料
[自强学堂django教程](http://www.ziqiangxuetang.com/django/django-intro.html)
[官方django文档](https://www.djangoproject.com/)
[bootstrap官方文档](http://www.bootcss.com/)
[bootstrap中文教程](http://www.runoob.com/bootstrap/bootstrap-typography.html)
[echarts官方文档](http://echarts.baidu.com/)

