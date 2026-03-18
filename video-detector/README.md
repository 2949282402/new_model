# AI生成视频检测系统

## 项目简介
本项目是一个基于 Spring Boot 的 AI生成视频检测系统，提供网页端上传、参数配置、视频预览、检测进度展示、结果报告展示和历史记录管理能力。后端调用 `E:\new_model\infer.py` 完成模型推理，并结合 `ffmpeg/ffprobe` 处理预览转码和视频帧信息校验。

当前代码已实现以下完整链路：
- 前端选择视频后优先尝试浏览器本地预览
- 浏览器无法直接播放时，自动请求后端使用 `ffmpeg` 生成预览版 MP4
- 用户提交检测任务后，后端按单任务队列串行执行
- 前端通过 SSE 实时接收排队状态、检测进度、完成结果和错误信息
- 检测完成后生成结果页，并将记录写入历史记录文件
- 服务重启时自动清理未完成任务残留内容，避免上传目录堆积垃圾

## 技术栈
- 后端：Spring Boot 3.2.0
- 前端：Thymeleaf + 原生 HTML / CSS / JavaScript
- 模型推理：Python 脚本 `infer.py`
- 预览转码：FFmpeg
- 视频信息读取：FFprobe

## 主要功能
- 视频上传与基础参数输入
- 本地预览优先，失败后自动走后端预览转码
- 单任务串行检测队列
- 检测进度条与排队人数提示
- 检测报告展示
- 注意力热力图展示与点击放大
- 历史记录查看、详情回显、删除记录
- 检测参数、排队耗时、检测耗时持久化
- 服务启动时清理未完成检测残留文件

## 检测流程
1. 用户在首页选择视频文件。
2. 前端先尝试本地预览。
3. 若浏览器无法直接播放，前端调用 `POST /preview`，由后端转码生成预览文件。
4. 用户点击开始检测后，前端将视频文件和参数 `attnTopK`、`threshold`、`everyN` 提交到 `POST /upload`。
5. 后端保存原视频，校验参数范围，并通过 `ffprobe` 校验 `everyN + 4 < 总帧数`。
6. 合法任务进入单线程队列，前端通过 `GET /progress/{taskId}` 订阅进度。
7. 后端调用 Python 脚本执行推理，解析概率、判定结果、分析帧数、详细分析和热力图路径。
8. 检测完成后，后端保存历史记录，前端跳转结果页展示报告。

## 项目结构
```text
video-detector/
├─ src/main/java/com/aigv/videodetector/
│  ├─ VideoDetectorApplication.java
│  ├─ config/
│  │  ├─ GlobalExceptionHandler.java
│  │  ├─ MultipartConfig.java
│  │  └─ WebConfig.java
│  ├─ controller/
│  │  └─ VideoController.java
│  ├─ model/
│  │  └─ DetectionRecord.java
│  └─ service/
│     ├─ DetectionRecordService.java
│     └─ PythonService.java
├─ src/main/resources/
│  ├─ application.properties
│  ├─ static/css/
│  └─ templates/
│     ├─ index.html
│     ├─ result.html
│     └─ history.html
├─ uploads/
├─ detection_records.txt
├─ pom.xml
├─ start.bat
├─ QUICKSTART.md
└─ ARCHITECTURE.md
```

## 运行环境
- JDK 21
- Maven 3.9+
- Python 3.8+
- 可用的 `ffmpeg` 和 `ffprobe`
- 模型权重文件 `E:\new_model\best.pth`

说明：
- 项目当前未包含 `mvnw`，请使用本机安装的 Maven。
- Python 环境需能直接执行 `python` 命令。

## 当前默认配置
配置文件位置：`src/main/resources/application.properties`

| 配置项 | 当前值 | 说明 |
| --- | --- | --- |
| `server.port` | `8080` | Web 服务端口 |
| `spring.servlet.multipart.max-file-size` | `500MB` | 单文件上传限制 |
| `spring.servlet.multipart.max-request-size` | `500MB` | 请求总大小限制 |
| `upload.dir` | `./uploads/` | 上传与检测输出目录 |
| `records.file` | `./detection_records.txt` | 历史记录文件 |
| `python.script.path` | `../infer.py` | 推理脚本路径 |
| `python.checkpoint.path` | `../best.pth` | 模型权重路径 |
| `python.save.attention` | `true` | 是否保存热力图 |
| `ffmpeg.path` | `../ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe` | FFmpeg 路径 |
| `preview.max.seconds` | `0` | 预览转码时长限制，`0` 表示不截断 |

## 参数约束
后端会对输入参数做边界校验：

- 热力图数量 `attnTopK`：`1` 到 `10`
- 判断阈值 `threshold`：`0.0` 到 `1.0`
- 抽帧间隔 `everyN`：`0` 到 `1000`
- 帧窗口约束：`everyN + 4 < 视频总帧数`

不满足约束时，后端会直接返回错误，不进入检测队列。

## 启动方式
### 方式一：使用 Maven
在项目根目录执行：

```bash
mvn spring-boot:run
```

### 方式二：先打包再运行
```bash
mvn clean package
java -jar target/video-detector-1.0.0.jar
```

### 方式三：Windows 下使用脚本
```bat
start.bat
```

启动后访问：

```text
http://localhost:8080
```

## 前端页面说明
### 首页 `/`
功能包括：
- 视频选择与文件信息展示
- 参数输入：热力图数量、判断阈值、抽帧间隔
- 本地预览与后端预览转码
- 检测进度展示
- 排队状态提示
- 历史记录入口

### 结果页 `/result`
展示内容包括：
- AI生成概率
- 判定结果
- 分析帧数
- 排队耗时与检测耗时
- 本次检测使用的三个参数
- 详细分析
- 注意力热力图与放大预览

### 历史记录页 `/history`
支持：
- 查看历史检测结果
- 回显概率、结果、帧数、参数、耗时
- 跳转查看详情
- 删除历史记录及相关文件

## 后端接口
| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 首页 |
| `POST` | `/upload` | 上传并创建检测任务 |
| `POST` | `/preview` | 生成预览转码文件 |
| `GET` | `/progress/{taskId}` | SSE 订阅检测进度 |
| `GET` | `/result` | 结果页 |
| `GET` | `/history` | 历史记录页 |
| `POST` | `/delete/{id}` | 删除历史记录 |

## 队列与清理机制
### 单任务串行执行
后端使用单线程执行器处理检测任务，因此同一时刻只允许一个检测任务运行，其余任务进入等待队列。前端会收到当前排队人数提示。

### 历史记录持久化
每次检测完成后，会保存以下信息：
- 原始文件名
- 视频路径
- 帧目录路径
- 热力图路径
- AI生成概率
- 判定结果
- 分析帧数
- 详细分析
- 排队耗时
- 检测耗时
- 参数 `attnTopK`、`threshold`、`everyN`
- 检测时间

### 服务启动清理
应用启动时会执行：
- 清理不完整的历史记录
- 清理上传目录中未被已完成记录引用的残留目录和文件
- 清理未完成检测留下的临时内容和预览转码残留

## 预览转码说明
当前实现不是所有视频都直接交给后端转码，而是采用“前端先试播，失败再转码”的策略：

- 浏览器可直接播放：直接本地预览
- 浏览器不可直接播放：上传到 `/preview`
- 后端使用 `ffmpeg` 生成可播放 MP4
- 预览文件默认放在 `uploads/_preview/`

## 静态资源与输出路径
- `/uploads/**` 会映射到本地上传目录
- 检测输出文件和中间结果存放在 `upload.dir` 对应目录下
- 历史记录默认写入项目根目录下的 `detection_records.txt`

## 常见问题
### 1. 预览转码失败
请检查：
- `ffmpeg.path` 是否指向真实可执行文件
- `ffprobe` 是否与 `ffmpeg` 位于同一目录，或可从系统 PATH 访问
- 上传目录是否可写

### 2. Python 脚本执行失败
请检查：
- `python` 命令是否可直接执行
- `infer.py` 路径是否正确
- `best.pth` 是否存在
- Python 环境依赖是否完整

### 3. 检测无法开始
请检查：
- 上传文件是否为空
- 参数是否越界
- 是否满足 `everyN + 4 < 总帧数`

### 4. 结果页没有热力图
请检查：
- `python.save.attention=true`
- 推理脚本是否实际输出热力图
- `/uploads/**` 资源映射是否正常

## 开发说明
- 结果页数据主要通过 `sessionStorage` 与历史记录回显传递
- 后端对详细分析中的敏感本地路径做了脱敏处理
- 新生成的热力图会优先展示概率最高的若干帧
- 前端和后端都对检测参数做了约束校验

## 许可证
本项目当前主要用于学习、研究和毕业设计相关工作。
