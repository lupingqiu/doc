# git 手册

1. 创建版本管理库
    cd dir
    git config --global user.name 'rube'
    git config --global user.eamil 'lookqlp@126.com'
    git init

2. 添加文件
    touch readme.md
    git status
    git add readme.md

3. 修改文件
    modify readme.md
    git diff
    git add readme.md
    git diff --cached
    git diff HEAD （stage和unstage两种状态）
    git status -s
    git commit -m 'comment'

    git commit -am 'comment' #add又commit

    git log
    git add .（所有文件的修改）

4. 管理库回到从前
    git commit --amend --no-edit #上一个提交忘了提交一个文件，补上一个文件或者修改
    git log --online

    #git add后stage状态->modified状态返回
    git reset file

    #commit 后 回到原先版本
    git reset --hard change-id
    #又相反悔
    git reflog #找到id 回到将来
    git reset --hard change-id

5. 单个文件回到从前
    git log --online
    git checkout change-id -- readme.md
    #跳回到文件从前一个版本，其他文件不会跳动。reset是跳动所有文件

6. branch
    git log --oneline --graph
    # 创建分支
    git branch dev
    git branch #查看分支
    git checkout dev #切换分支

    git branch -d dev #删除分支
    #另外一种方法，同时切换了分支
    git checkout -b dev

    #在当前版本上merge别的分支功能
    git checkout master
    git merge --no-ff -m 'keep merge info' dev

7. merge冲突
    git merge dev
    #手动修改，然后提交
    git commit -am "solve conflict"

    #另外方法
    git rebase branch#很危险，建议不使用

8. stash临时修复
    #工作被打断，放在一个暂存的空间，回头可以回到此空间
    git status -s
    git stash
    git status -s
    git checkout -b other
    git commit -am 'job other'
    git checkout master
    git merge --no-ff -m 'merge other' other
    git commit -am "solve conflict"
    git log --online --graph

    git checkout dev
    git branch -D other
    git stash pop #返回

9. 开源空间
    git remote add origin https://github.com/xxxx.git
    git push -u origin master

    git push -u origin dev
