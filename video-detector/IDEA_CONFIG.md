# Python + Java 混合项目 IDEA 配置说明

## 📋 配置步骤总结

### 1. 打开项目结构设置
**快捷键**: `Ctrl+Alt+Shift+S`

### 2. 配置 Modules
在 **Modules** 中添加：
- ✅ `video-detector` (Maven Module)
- ✅ 根目录 (Python Sources Root)

### 3. 配置 SDKs
需要配置两个 SDK：
- **Java SDK 11** - 用于 Java 项目
- **Python Interpreter** - 用于 Python 脚本

### 4. 配置 Run Configurations
创建三个运行配置：

#### Spring Boot 应用
```
Name: VideoDetector Application
Configuration Type: Spring Boot
Main class: com.aigv.videodetector.VideoDetectorApplication
Working directory: E:\new_model\video-detector
Use classpath of module: video-detector
```

#### Python infer.py
```
Name: Run infer.py
Configuration Type: Python
Script path: E:\new_model\infer.py
Parameters: <视频路径> --checkpoint E:\new_model\best.pth
Working directory: E:\new_model
```

#### Maven Build
```
Name: Build Project
Configuration Type: Maven
Working directory: E:\new_model\video-detector
Command line: clean package
```

## ⚡ 快速配置方法

### 方法 1：自动导入（推荐）
1. 打开 IDEA
2. 选择 `Open` → 选择 `E:\new_model\video-detector` 目录
3. IDEA 会自动识别为 Maven 项目并导入

### 方法 2：手动添加
1. 右键 `pom.xml`
2. 选择 `Add as Maven Project`

## 🔧 必要插件

确保安装以下插件：
- [x] Python
- [x] Spring Boot
- [x] Lombok
- [x] Maven Helper

安装路径：`File` → `Settings` → `Plugins`

## 🎯 验证配置成功

### 检查项
1. ✅ `video-detector` 显示为 Maven 模块
2. ✅ `.py` 文件有 Python 图标
3. ✅ `.java` 文件有 Java 图标
4. ✅ `pom.xml` 被识别为 Maven 配置
5. ✅ 可以运行 Spring Boot 应用
6. ✅ 可以运行 Python 脚本

## 💡 使用技巧

### 切换视图
- **Project 视图**: 查看完整项目结构
- **Packages 视图**: 按包查看 Java 代码
- **Problems 视图**: 查看所有错误和警告

### 快速运行
- **Spring Boot**: 双击 `VideoDetectorApplication.java` → `Run`
- **Python**: 右键 `.py` 文件 → `Run`
- **Maven**: 在 Maven 面板中双击目标

### 调试配置
- Java: 在代码行号旁点击设置断点
- Python: 同样方式设置断点
- 使用 `Debug` 模式启动

## 🐛 常见问题

### Q: Python 文件显示为纯文本
**A**: 
1. 安装 Python 插件
2. 右键 `.py` 文件 → `Override File Type` → `Python`

### Q: Maven 依赖标红
**A**: 
1. 右键 `pom.xml` → `Maven` → `Reload project`
2. 或点击 Maven 面板的刷新按钮

### Q: 无法识别 Spring Boot
**A**: 
1. 安装 Spring Boot 插件
2. 确保 `pom.xml` 中有 Spring Boot 依赖

## 📝 推荐的 .gitignore

```gitignore
# IDEA
.idea/
*.iml
*.iws

# Maven
target/
pom.xml.tag
pom.xml.releaseBackup

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# 上传文件
uploads/

# 系统文件
.DS_Store
Thumbs.db
```

## 🚀 下一步

配置完成后，你可以：
1. 运行 Spring Boot 应用
2. 访问 http://localhost:8080
3. 上传视频测试
4. 调试 Python 脚本
5. 享受混合开发的便利！

---

详细配置步骤请参考：**IDEA_SETUP.md**
