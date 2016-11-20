简单点说：

1. git add file

2. git commit

3. git push

详细的说：

1. 运行Git status命令查看变化，new delete modified

2. 运行git diff file 查看修改的内容

3. 提交到本地，如果是modified或者新建则git add file，如果时delete则git rm file；然后执行git commit
   git reset 回退

4. 同步到服务器先执行git pull，如果又冲突git checkout有冲突的文件；git push origin
