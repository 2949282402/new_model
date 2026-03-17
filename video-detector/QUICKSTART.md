# 快速使用指南

## 🚀 快速开始

### 1. 环境准备
确保已安装以下软件：
- **Java JDK 11** 或更高版本
- **Maven 3.6+**
- **Python 3.8+** (已配置好 infer.py 所需依赖)

### 2. 一键启动

#### Windows 用户：
双击运行 `start.bat` 文件

或在命令行中执行：
```bash
cd E:\new_model\video-detector
start.bat
```

#### 手动启动：
```bash
# 编译项目
mvn clean package

# 运行应用
mvn spring-boot:run
```

### 3. 访问系统

打开浏览器访问：**http://localhost:8080**

## 📝 使用流程

### 步骤 1：上传视频
- 点击页面中央的上传区域
- 选择要检测的视频文件
- 或者直接将视频拖拽到上传区域

### 步骤 2：开始检测
- 确认文件已选择
- 点击"开始检测"按钮
- 等待 AI 分析完成（可能需要几分钟）

### 步骤 3：查看结果
检测完成后会自动显示：
- ✅ **AI 生成概率** - 以百分比和进度条显示
- 🔍 **判定结果** - "真实视频"或"AI 生成视频"
- 📊 **分析帧数** - 用于检测的视频帧数量
- 📋 **详细日志** - Python 脚本的完整输出

## ⚙️ 配置说明

### 修改上传目录
编辑 `src/main/resources/application.properties`:
```properties
upload.dir=你的上传目录路径/
```

### 修改 Python 脚本路径
```properties
python.script.path=你的/infer.py 路径
python.checkpoint.path=你的/best.pth 路径
```

### 修改端口号
```properties
server.port=你想要的端口
```

## ⚠️ 注意事项

1. **首次使用**
   - 确保 `best.pth` 模型文件存在
   - 确保 FlowFormer 相关依赖正确配置
   - 检查 Python 依赖是否完整

2. **视频格式支持**
   - MP4, AVI, MOV, MKV
   - WMV, FLV, WebM
   - 以及其他常见格式

3. **文件大小**
   - 默认最大：500MB
   - 可在配置文件中调整

4. **性能提示**
   - 大视频需要更长处理时间
   - 建议使用 GPU 加速
   - 可在 infer.py 中调整参数优化速度

## 🔧 故障排除

### 问题 1：启动失败
**解决方案：**
```bash
# 检查 Java 版本
java -version

# 检查 Maven
mvn -version
```

### 问题 2：上传失败
**解决方案：**
- 检查 `E:\new_model\uploads` 目录是否存在
- 检查磁盘空间是否充足

### 问题 3：Python 脚本执行错误
**解决方案：**
```bash
# 测试 Python 脚本
python E:\new_model\infer.py --help

# 检查 Python 依赖
pip list
```

### 问题 4：检测时间过长
**解决方案：**
- 这是正常现象，特别是大视频
- 可调整 infer.py 中的参数：
  - `--max_frames`: 限制最大帧数
  - `--batch_size`: 调整批处理大小
  - `--every_n`: 隔 N 帧抽取一帧

## 📞 获取帮助

如遇到其他问题：
1. 查看控制台日志
2. 检查 Python 脚本输出
3. 确认所有路径配置正确
4. 重启应用试试

## 🎯 使用示例

### 命令行测试（可选）
你也可以先在命令行测试 Python 脚本：
```bash
python E:\new_model\infer.py E:\videos\test.mp4 --checkpoint E:\new_model\best.pth --threshold 0.5
```

### 网页使用
1. 打开 http://localhost:8080
2. 上传视频
3. 等待结果
4. 查看详细报告

---

**祝使用愉快！** 🎉
