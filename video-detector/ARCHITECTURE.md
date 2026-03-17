# 系统架构说明

## 🏗️ 技术架构

### 整体架构
```
┌─────────────────┐
│   用户浏览器     │
│  (前端页面)     │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│ Spring Boot     │
│ 应用服务器      │
│ - 视频上传      │
│ - 结果展示      │
└────────┬────────┘
         │ 进程调用
         ▼
┌─────────────────┐
│ Python 脚本     │
│ (infer.py)      │
│ - AI 模型推理   │
│ - 结果分析      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   检测结果      │
└─────────────────┘
```

## 📦 项目结构详解

```
video-detector/
│
├── src/main/java/com/aigv/videodetector/
│   │
│   ├── VideoDetectorApplication.java    # Spring Boot 主启动类
│   │
│   ├── controller/
│   │   └── VideoController.java         # REST API 控制器
│   │       - POST /upload              # 视频上传接口
│   │       - GET  /result              # 结果页面路由
│   │
│   ├── service/
│   │   └── PythonService.java           # Python 服务层
│   │       - runDetection()            # 执行 AI 检测
│   │       - parseResult()             # 解析检测结果
│   │
│   └── config/
│       └── WebConfig.java               # Web 配置
│           - 静态资源映射
│
├── src/main/resources/
│   ├── application.properties           # 应用配置
│   │
│   └── templates/
│       ├── index.html                   # 首页（上传）
│       │   - 拖拽上传
│       │   - 进度显示
│       │   - AJAX 提交
│       │
│       └── result.html                  # 结果页
│           - 概率展示
│           - 可视化图表
│           - 详细日志
│
├── pom.xml                              # Maven 依赖配置
├── start.bat                            # Windows 启动脚本
├── README.md                            # 项目说明文档
└── QUICKSTART.md                        # 快速入门指南
```

## 🔄 工作流程

### 1. 视频上传流程
```
用户选择视频
    ↓
前端验证格式
    ↓
AJAX 上传到后端
    ↓
保存到 uploads/目录
    ↓
生成唯一文件名
```

### 2. AI 检测流程
```
Spring Boot 接收请求
    ↓
PythonService 构建命令
    ↓
ProcessBuilder 启动 Python 进程
    ↓
执行 infer.py 脚本
    ↓
实时读取输出
    ↓
等待完成
    ↓
解析结果
```

### 3. 结果展示流程
```
后端返回 JSON
    ↓
前端解析数据
    ↓
跳转到结果页
    ↓
渲染概率条
    ↓
显示详细信息
```

## 🔧 核心组件说明

### VideoController
**职责**: 处理 HTTP 请求
- `@PostMapping("/upload")` - 处理视频上传
- `@GetMapping("/result")` - 显示结果页面
- 文件验证
- 错误处理

### PythonService
**职责**: 调用 Python 脚本
- 使用 `ProcessBuilder` 创建进程
- 设置工作目录
- 捕获标准输出和错误流
- 解析文本结果为 JSON

### WebConfig
**职责**: 配置静态资源
- 映射上传目录到 `/uploads/**`
- 允许直接访问上传的文件

## 🛠️ 关键技术点

### 1. 文件上传
```java
@PostMapping("/upload")
@ResponseBody
public ResponseEntity<Map<String, Object>> uploadVideo(
    @RequestParam("video") MultipartFile file
)
```

### 2. Python 进程调用
```java
ProcessBuilder processBuilder = new ProcessBuilder(
    "python",
    scriptPath,
    videoPath,
    "--checkpoint", checkpointPath,
    "--threshold", "0.5"
);
```

### 3. 实时输出捕获
```java
BufferedReader reader = new BufferedReader(
    new InputStreamReader(process.getInputStream())
);
```

### 4. 结果解析
从 Python 脚本的标准输出中提取：
- AI 生成概率
- 判定结果
- 分析帧数
- 详细日志

## 📊 数据流

### 上传阶段
```
Browser → [Multipart Request] → Controller → File System
```

### 检测阶段
```
Controller → Service → [Process] → Python Script → AI Model
```

### 返回阶段
```
Python Output → Service → Parse JSON → Controller → Browser
```

## 🔐 安全考虑

### 已实现的安全措施
1. ✅ 文件类型验证
2. ✅ 文件大小限制 (500MB)
3. ✅ 唯一文件名生成 (UUID)
4. ✅ 路径验证

### 建议的安全增强
- 添加用户认证
- 实现访问控制
- 添加病毒扫描
- 限制上传频率

## 🚀 性能优化

### 当前优化
1. 使用临时文件存储上传视频
2. 流式读取 Python 输出
3. 异步处理大文件

### 可扩展方向
- 添加消息队列处理长任务
- 实现分布式部署
- 添加缓存机制
- 支持并发检测

## 📈 未来扩展

### 功能扩展
- [ ] 历史记录数据库
- [ ] 用户账户系统
- [ ] 批量检测
- [ ] 报告导出 (PDF)
- [ ] API 接口开放

### 性能提升
- [ ] Redis 缓存
- [ ] 异步任务队列
- [ ] 负载均衡
- [ ] CDN 加速

---

**本系统完全开源，可供学习和研究使用！**
