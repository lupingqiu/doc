#docker

##common
>**command:**
> - sudo docker images 列出本地images
> - sudo docker ps 列出正在运行的container
> - sudo docker exec -it qmysql bash  进入container bash
> - sudo docker run --name qmysql -e MYSQL_ROOT_PASSWORD=rube -d -p 4444:3306 mysql:latest启动mysql，且端口映射至本机４４４４
> －sudo docker pull mysql下载image

##使用

