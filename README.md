# pytorch

# git上传(先确定PYTHON PATH)
!git remote rm origin

#把这个文件夹变成Git可管理的仓库。
!git init

#把该目录下的所有文件添加到仓库
!git add "pytorch_learning.py"
!git add "model.py"

#把项目提交到仓库。
!git commit -m 'upload' 

#(将本地仓库与GitHub上创建好的目标远程仓库进行关联。 …后面加的是GitHub目标仓库地址)。
!git remote add origin https://github.com/kason0611/pytorch.git

#把本地库的所有内容推送到GitHub远程仓库上。
!git push -u origin master


!git pull origin master --allow-unrelated-histories

# 新建一个分支
!git branch newbranch 

# 检查分支是否创建成功
!git branch 
# 切换到你的新分支
!git checkout newbranch
# 将你的改动提交到新分支上
!git add .
!git commit -m 'upload_new'

#然后git status检查是否成功
!git status 

#切换到主分支
!git checkout master 

#将新分支提交的改动合并到主分支上


!git merge newbranch
push代码

!git push -u origin master
删除这个分支

!git branch -D newbranch
