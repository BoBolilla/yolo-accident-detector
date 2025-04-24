#### 1. 克隆仓库到本地
```bash
git clone git@github.com:BoBolilla/yolo-accident-detector.git
cd yolo-accident-detector
```

#### 2. 创建个人开发分支（每天开始开发前）
```bash
git checkout -b feature/yourname  # 推荐分支命名规范
# 例如：git checkout -b feature/zd
```

#### 3. 开发与本地提交
```bash
# 修改代码后提交
git add .
git commit -m "完成xxx功能（具体描述变更内容）"

```

#### 4. 推送分支到远程
```bash
git push origin feature/yourname
# 例如 git push origin feature/zd
# 首次推送需要加 -u 参数：
git push -u origin feature/yourname
# 例如 git push -u origin feature/zd
```

#### 5. 创建Pull Request（PR）
- 在GitHub网页端操作
- 选择 Compare & pull request
- 填写清晰的PR描述（做了什么/为什么修改/测试情况）
- 关联相关Issue（如有）


#### 6. 代码审查与合并
1. 其他成员审查代码：
   - 在PR页面的Files changed标签页进行行级评论
   - 使用`/test`命令触发CI（如果配置了自动化测试）

2. 合并代码：
```bash
# 当PR通过审查后（组长操作）
git checkout master
git pull origin master  # 确保本地master最新
git merge --no-ff feature/branch-name  # 保留分支历史
git push origin master
```


### 同步最新代码（组员日常操作）
```bash
# 每天开始工作前
git checkout master
git pull origin master

# 回到开发分支合并最新代码
git checkout feature/your-branch
git merge master  # 处理可能出现的冲突
```

---

### 冲突解决流程
当出现冲突时：
```bash
# 查看冲突文件
git status

# 手动编辑标记了<<<<<<<的文件
# 解决后提交
git add .
git commit -m "解决与master分支的合并冲突"
```



### **常用命令速查表**
| 操作 | 命令 |
|------|------|
| 查看分支 | `git branch -av` |
| 删除已合并分支 | `git branch -d branch-name` |
| 暂存修改 | `git stash` |
| 查看远程仓库 | `git remote -v` |
| 撤销本地修改 | `git checkout -- filename` |

使用 `git log --graph --oneline` 查看提交历史
使用 `git shortlog -sn` 查看成员贡献统计。
