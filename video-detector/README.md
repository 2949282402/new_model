# AI 视频检测系统 - README

## 项目简介
这是一个基于 Spring Boot 的 AI 生成视频检测系统，提供网页界面供用户上传视频，并调用 Python AI 检测脚本进行分析，返回检测结果。

## 技术栈
- **后端**: Spring Boot 2.7.18
- **前端**: HTML5 + CSS3 + JavaScript (原生)
- **AI 检测**: Python 脚本 (infer.py)
- **模板引擎**: Thymeleaf

## 功能特性
1. ✅ 视频文件上传（支持拖拽）
2. ✅ 自动调用 Python AI 检测脚本
3. ✅ 实时显示检测进度
4. ✅ 可视化检测结果展示
5. ✅ 支持多种视频格式
6. ✅ 最大支持 500MB 视频文件

## 项目结构
```
video-detector/
├── src/
│   └── main/
│       ├── java/com/aigv/videodetector/
│       │   ├── VideoDetectorApplication.java    # 主应用类
│       │   ├── controller/
│       │   │   └── VideoController.java         # 视频上传控制器
│       │   ├── service/
│       │   │   └── PythonService.java           # Python 脚本调用服务
│       │   └── config/
│       │       └── WebConfig.java               # Web 配置
│       └── resources/
│           ├── application.properties           # 配置文件
│           └── templates/
│               ├── index.html                   # 上传页面
│               └── result.html                  # 结果页面
└── pom.xml                                      # Maven 配置
```

## 安装步骤

### 1. 环境要求
- Java JDK 11+
- Maven 3.6+
- Python 3.8+
- 已安装 AI 检测所需的 Python 依赖

### 2. 配置检查
编辑 `src/main/resources/application.properties` 文件，确保路径配置正确：

```properties
# 上传目录
upload.dir=E:/new_model/uploads/

# Python 脚本路径
python.script.path=E:/new_model/infer.py
python.checkpoint.path=E:/new_model/best.pth
```

### 3. 创建上传目录
确保上传目录存在：
```bash
mkdir E:\new_model\uploads
```

### 4. 编译项目
在项目根目录执行：
```bash
mvn clean package
```

### 5. 运行应用

#### 方式一：使用 Maven
```bash
mvn spring-boot:run
```

#### 方式二：运行 JAR 包
```bash
java -jar target/video-detector-1.0.0.jar
```

### 6. 访问应用
打开浏览器访问：http://localhost:8080

## 使用说明

1. **上传视频**
   - 点击上传区域选择视频文件
   - 或者直接拖拽视频文件到上传区域
   - 支持格式：MP4, AVI, MOV, MKV, WMV, FLV, WebM 等

2. **开始检测**
   - 选择文件后，点击"开始检测"按钮
   - 系统会自动上传视频并调用 AI 检测脚本
   - 等待检测完成（可能需要几分钟）

3. **查看结果**
   - 检测完成后自动跳转到结果页面
   - 查看 AI 生成概率
   - 查看判定结果（真实/AI 生成）
   - 查看详细分析日志

## 注意事项

1. **Python 环境**
   - 确保 Python 已添加到系统 PATH
   - 确保已安装 infer.py 所需的所有依赖
   - 建议使用虚拟环境

2. **模型文件**
   - 确保 `best.pth` 模型文件存在于指定路径
   - FlowFormer 相关依赖需要正确配置

3. **文件大小**
   - 默认最大上传 500MB
   - 可在 `application.properties` 中调整

4. **性能考虑**
   - 大视频处理时间较长
   - 建议在服务器端运行
   - 可调整 Python 脚本的批大小和帧数限制

## 配置选项

### application.properties 配置项

```properties
# 服务器端口
server.port=8080

# 文件大小限制
spring.servlet.multipart.max-file-size=500MB
spring.servlet.multipart.max-request-size=500MB

# 上传目录
upload.dir=E:/new_model/uploads/

# Python 脚本路径
python.script.path=E:/new_model/infer.py
python.checkpoint.path=E:/new_model/best.pth

# 静态资源配置
spring.web.resources.static-locations=file:E:/new_model/uploads/,classpath:/static/
```

## 常见问题

### Q: 上传失败
A: 检查上传目录是否有写权限，确保目录存在

### Q: Python 脚本执行失败
A: 
- 检查 Python 是否正确安装
- 检查 infer.py 路径是否正确
- 检查 Python 依赖是否完整
- 查看控制台错误日志

### Q: 检测时间过长
A: 
- 视频较大属于正常现象
- 可调整 Python 脚本参数减少帧数
- 使用 GPU 加速

## 开发计划
- [ ] 添加历史记录功能
- [ ] 支持批量检测
- [ ] 添加用户认证
- [ ] 优化检测速度
- [ ] 生成 PDF 报告

## 许可证
本项目仅供学习和研究使用

## 联系方式
如有问题，请提交 Issue 或联系开发者
