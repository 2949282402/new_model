package com.aigv.videodetector.controller;

import com.aigv.videodetector.model.DetectionRecord;
import com.aigv.videodetector.service.DetectionRecordService;
import com.aigv.videodetector.service.PythonService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * 视频上传和检测控制器
 */
@Slf4j
@Controller
@RequestMapping("/")
public class VideoController {

    @Autowired
    private PythonService pythonService;

    @Autowired
    private DetectionRecordService recordService;

    // 存储正在进行的检测任务
    private final Map<String, SseEmitter> activeEmitters = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Object queueLock = new Object();
    private final java.util.Deque<QueueTask> pendingQueue = new java.util.ArrayDeque<>();
    private volatile QueueTask currentTask;

    @Value("${upload.dir}")
    private String uploadDir;
    @Value("${ffmpeg.path:ffmpeg}")
    private String ffmpegPath;
    @Value("${preview.max.seconds:0}")
    private Integer previewMaxSeconds;
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * 首页 - 上传页面
     */
    @GetMapping
    public String index() {
        return "index";
    }

    /**
     * 上传视频并检测
     */
    @PostMapping("/upload")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> uploadVideo(@RequestParam("video") MultipartFile file,
                                                           @RequestParam(value = "attnTopK", required = false) Integer attnTopK,
                                                           @RequestParam(value = "threshold", required = false) Double threshold,
                                                           @RequestParam(value = "everyN", required = false) Integer everyN) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            // 验证文件
            if (file.isEmpty()) {
                response.put("success", false);
                response.put("message", "请选择要上传的视频文件");
                return ResponseEntity.badRequest().body(response);
            }

            // 验证文件类型
            String originalFilename = file.getOriginalFilename();
            if (originalFilename == null || !isValidVideoFile(originalFilename)) {
                response.put("success", false);
                response.put("message", "请上传有效的视频文件 (mp4, avi, mov, mkv 等)");
                return ResponseEntity.badRequest().body(response);
            }

            // 创建上传目录
            File uploadPath = new File(uploadDir);
            if (!uploadPath.exists()) {
                uploadPath.mkdirs();
            }

            // 生成唯一 ID 和文件夹名
            String uniqueId = UUID.randomUUID().toString();
            // 为每个视频创建独立文件夹，所有相关文件都放在此文件夹内
            String videoFolderName = uniqueId + "_" + originalFilename.substring(0, originalFilename.lastIndexOf('.'));
            File videoFolder = new File(uploadDir + videoFolderName);
            if (!videoFolder.exists()) {
                videoFolder.mkdirs();
                log.info("创建视频文件夹：{}", videoFolder.getAbsolutePath());
            }

            // 保存视频文件到该文件夹
            String videoFilename = "original_video." + originalFilename.substring(originalFilename.lastIndexOf('.') + 1);
            Path filePath = Paths.get(videoFolder.getAbsolutePath() + File.separator + videoFilename);

            // 保存文件
            Files.copy(file.getInputStream(), filePath);

            // 执行 AI 检测（异步，通过 SSE 推送进度）
            log.info("开始 AI 检测...");
            String taskId = UUID.randomUUID().toString();
            
            // 先创建 SSE emitter，确保前端可以连接
            SseEmitter emitter = new SseEmitter(5 * 60 * 1000L); // 5 分钟超时
            activeEmitters.put(taskId, emitter);
            
            // 设置 emitter 的回调处理
            emitter.onCompletion(() -> {
                log.info("SSE 连接完成：{}", taskId);
                activeEmitters.remove(taskId);
            });
            
            emitter.onTimeout(() -> {
                log.info("SSE 连接超时：{}", taskId);
                activeEmitters.remove(taskId);
            });
            
            emitter.onError((e) -> {
                log.error("SSE 连接错误：{}", taskId, e);
                activeEmitters.remove(taskId);
            });
            
            response.put("success", true);
            response.put("message", "检测已启动");
            response.put("taskId", taskId);
                        response.put("filename", originalFilename);
            response.put("filepath", filePath.toString());
            
            QueueTask task = new QueueTask(taskId, originalFilename, filePath.toString(),
                    videoFolder.getAbsolutePath(), attnTopK, threshold, everyN);
            enqueueTask(task);
            
            return ResponseEntity.ok(response);

        } catch (IOException e) {
            log.error("文件上传失败", e);
            response.put("success", false);
            response.put("message", "文件上传失败：" + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        } catch (Exception e) {
            log.error("检测过程出错", e);
            response.put("success", false);
            response.put("message", "检测失败：" + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    /**
     * 查看检测结果页面
     */
    /**
     * 仅生成预览视频（不触发检测）
     */
    @PostMapping("/preview")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> previewVideo(@RequestParam("video") MultipartFile file) {
        Map<String, Object> response = new HashMap<>();
        log.info("===== 收到预览请求 =====");
        try {
            if (file.isEmpty()) {
                log.warn("预览请求：文件为空");
                response.put("success", false);
                response.put("message", "请选择要预览的视频文件");
                return ResponseEntity.badRequest().body(response);
            }

            String originalFilename = file.getOriginalFilename();
            log.info("预览请求：文件名={}, 大小={} bytes, 类型={}", 
                    originalFilename, file.getSize(), file.getContentType());
            
            if (originalFilename == null || !isValidVideoFile(originalFilename)) {
                log.warn("预览请求：无效的视频文件：{}", originalFilename);
                response.put("success", false);
                response.put("message", "请上传有效的视频文件 (mp4, avi, mov, mkv 等)");
                return ResponseEntity.badRequest().body(response);
            }

            File previewDir = new File(uploadDir, "_preview");
            if (!previewDir.exists()) {
                previewDir.mkdirs();
                log.info("创建预览目录：{}", previewDir.getAbsolutePath());
            }
            log.info("预览目录：{}", previewDir.getAbsolutePath());

            String ext = originalFilename.contains(".")
                    ? originalFilename.substring(originalFilename.lastIndexOf('.'))
                    : ".mp4";
            String sourceName = "source_" + UUID.randomUUID() + ext;
            Path sourcePath = Paths.get(previewDir.getAbsolutePath(), sourceName);
            log.info("保存临时文件：{}", sourcePath.toAbsolutePath());
            Files.copy(file.getInputStream(), sourcePath, StandardCopyOption.REPLACE_EXISTING);

            String previewName = "preview_" + UUID.randomUUID() + ".mp4";
            Path previewPath = Paths.get(previewDir.getAbsolutePath(), previewName);
            log.info("预览输出路径：{}", previewPath.toAbsolutePath());

            try {
                log.info("开始创建预览视频...");
                createPreviewVideo(sourcePath, previewPath);
            } finally {
                log.info("删除临时文件：{}", sourcePath.toAbsolutePath());
                Files.deleteIfExists(sourcePath);
            }

            if (Files.exists(previewPath)) {
                log.info("✅ 预览生成成功：{}, 大小：{} bytes", previewPath.toAbsolutePath(), Files.size(previewPath));
                String previewUrl = toUploadUrl(previewPath);
                log.info("预览 URL: {}", previewUrl);
                response.put("success", true);
                response.put("previewUrl", previewUrl);
                return ResponseEntity.ok(response);
            } else {
                log.error("❌ 预览文件不存在：{}", previewPath.toAbsolutePath());
                response.put("success", false);
                response.put("message", "预览生成失败");
                return ResponseEntity.internalServerError().body(response);
            }
        } catch (Exception e) {
            log.error("预览生成失败", e);
            response.put("success", false);
            response.put("message", "预览生成失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    @GetMapping("/result")
    public String resultPage(@RequestParam(value = "filename", required = false) String filename,
                            @RequestParam(value = "filepath", required = false) String filepath,
                            @RequestParam(value = "result", required = false) String result,
                            Model model) {
        model.addAttribute("filename", filename != null ? filename : "");
        model.addAttribute("filepath", filepath != null ? filepath : "");
        model.addAttribute("result", result != null ? result : "");
        return "result";
    }

    /**
     * 查看历史检测记录页面
     */
    @GetMapping("/history")
    public String historyPage(Model model) {
        try {
            List<DetectionRecord> records = recordService.getAllRecords();
            log.info("历史页面加载成功，共 {} 条记录", records.size());
            model.addAttribute("records", records);
            return "history";
        } catch (Exception e) {
            log.error("加载历史页面失败", e);
            model.addAttribute("error", "加载历史记录失败：" + e.getMessage());
            model.addAttribute("records", new ArrayList<>());
            return "history";
        }
    }

    /**
     * 删除检测记录
     */
    @PostMapping("/delete/{id}")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> deleteRecord(@PathVariable String id) {
        Map<String, Object> response = new HashMap<>();
        
        log.info("收到删除请求，ID: {}", id);
        
        try {
            boolean deleted = recordService.deleteRecord(id);
            
            if (deleted) {
                log.info("删除成功，ID: {}", id);
                response.put("success", true);
                response.put("message", "记录已删除");
            } else {
                log.warn("删除失败，记录不存在，ID: {}", id);
                response.put("success", false);
                response.put("message", "记录不存在");
            }
        } catch (Exception e) {
            log.error("删除记录时发生异常，ID: {}", id, e);
            response.put("success", false);
            response.put("message", "删除失败：" + e.getMessage());
        }
        
        return ResponseEntity.ok(response);
    }

    /**
     * SSE 进度推送端点（仅用于前端连接）
     */
    @GetMapping(value = "/progress/{taskId}", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter progress(@PathVariable String taskId) {
        // 返回已存在的 emitter
        SseEmitter emitter = activeEmitters.get(taskId);
        if (emitter == null) {
            log.warn("未找到对应的 SSE emitter，taskId: {}", taskId);
            // 创建一个新的但立即完成，避免前端一直等待
            SseEmitter emptyEmitter = new SseEmitter();
            emptyEmitter.complete();
            return emptyEmitter;
        }
        try {
            Map<String, Object> connected = new HashMap<>();
            connected.put("type", "connected");
            emitter.send(connected);

            Integer position = getQueuePosition(taskId);
            if (position != null && position > 0) {
                Map<String, Object> queued = new HashMap<>();
                queued.put("type", "queued");
                queued.put("position", position);
                emitter.send(queued);
            } else if (currentTask != null && currentTask.taskId.equals(taskId)) {
                Map<String, Object> start = new HashMap<>();
                start.put("type", "start");
                emitter.send(start);
            }
        } catch (IOException e) {
            log.warn("发送连接状态失败，taskId: {}", taskId, e);
        }
        return emitter;
    }

    /**
     * 验证是否为有效的视频文件
     */
    private boolean isValidVideoFile(String filename) {
        String lowerName = filename.toLowerCase();
        return lowerName.endsWith(".mp4") || 
               lowerName.endsWith(".avi") || 
               lowerName.endsWith(".mov") || 
               lowerName.endsWith(".mkv") || 
               lowerName.endsWith(".wmv") || 
               lowerName.endsWith(".flv") || 
               lowerName.endsWith(".webm") || 
               lowerName.endsWith(".m4v") || 
               lowerName.endsWith(".mpg") || 
               lowerName.endsWith(".mpeg");
    }

    private void createPreviewVideo(Path inputFile, Path outputFile) throws IOException, InterruptedException {
        String ffmpegExecutable = resolveFfmpegExecutable();
        List<String> command = new ArrayList<>();
        command.add(ffmpegExecutable);
        command.add("-y");
        command.add("-i");
        command.add(inputFile.toString());
        if (previewMaxSeconds != null && previewMaxSeconds > 0) {
            command.add("-t");
            command.add(String.valueOf(previewMaxSeconds));
        }
        command.add("-vf");
        command.add("scale=min(960\\,iw):-2,fps=15");
        command.add("-an");
        command.add("-c:v");
        command.add("libx264");
        command.add("-preset");
        command.add("veryfast");
        command.add("-crf");
        command.add("28");
        command.add("-pix_fmt");
        command.add("yuv420p");
        command.add("-movflags");
        command.add("+faststart");
        command.add(outputFile.toString());

        log.info("开始FFmpeg预览转码: {}", String.join(" ", command));
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.redirectErrorStream(true);
        Process process = builder.start();
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append('\n');
            }
        }
        int exitCode = process.waitFor();
        log.info("FFmpeg转码结束，exitCode={}, 输出片段: {}", exitCode,
                output.length() > 1000 ? output.substring(output.length() - 1000) : output.toString());
        if (exitCode != 0 || !Files.exists(outputFile)) {
            throw new IOException("FFmpeg转码失败: " + output.toString().trim());
        }
    }

    private String resolveFfmpegExecutable() throws IOException {
        String configured = ffmpegPath != null ? ffmpegPath.trim() : "";
        if (configured.isEmpty()) {
            return "ffmpeg";
        }

        Path rawPath = Paths.get(configured);
        List<Path> candidates = new ArrayList<>();
        if (rawPath.isAbsolute()) {
            candidates.add(rawPath);
        } else {
            Path userDir = Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize();
            candidates.add(userDir.resolve(rawPath).normalize());
            Path parent = userDir.getParent();
            if (parent != null) {
                candidates.add(parent.resolve(rawPath).normalize());
            }
        }

        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                log.info("使用FFmpeg路径: {}", candidate);
                return candidate.toString();
            }
            if (Files.isDirectory(candidate)) {
                Path exe = candidate.resolve(isWindows ? "ffmpeg.exe" : "ffmpeg");
                if (Files.isRegularFile(exe)) {
                    log.info("使用FFmpeg路径: {}", exe);
                    return exe.toString();
                }
            }
        }

        StringBuilder tried = new StringBuilder();
        for (Path candidate : candidates) {
            if (tried.length() > 0) {
                tried.append(", ");
            }
            tried.append(candidate);
        }
        String userDir = System.getProperty("user.dir");
        throw new IOException("FFmpeg 未找到: " + configured + " (user.dir=" + userDir + ", 尝试路径: " + tried + ")");
    }

    private String toUploadUrl(Path filePath) {
        Path root = Paths.get(uploadDir).toAbsolutePath().normalize();
        Path absolute = filePath.toAbsolutePath().normalize();
        if (absolute.startsWith(root)) {
            Path relative = root.relativize(absolute);
            String urlPath = relative.toString().replace(File.separatorChar, '/');
            return "/uploads/" + urlPath;
        }
        return "/uploads/" + filePath.getFileName();
    }

    /**
     * 从 JSON 字符串中提取字段值（简单实现）
     */
    private String extractJsonField(String json, String fieldName) {
        if (json == null || json.isEmpty()) {
            return "";
        }
        
        // 查找字段名
        String searchKey = "\"" + fieldName + "\":";
        int keyIndex = json.indexOf(searchKey);
        if (keyIndex == -1) {
            return "";
        }
        
        // 找到值的起始位置
        int valueStart = keyIndex + searchKey.length();
        
        // 跳过空白字符
        while (valueStart < json.length() && Character.isWhitespace(json.charAt(valueStart))) {
            valueStart++;
        }
        
        // 检查是否是字符串值
        if (valueStart >= json.length() || json.charAt(valueStart) != '"') {
            return "";
        }
        
        // 找到字符串的结束位置
        valueStart++; // 跳过开始的引号
        int valueEnd = valueStart;
        while (valueEnd < json.length()) {
            char c = json.charAt(valueEnd);
            if (c == '\\') {
                valueEnd += 2; // 跳过转义字符
            } else if (c == '"') {
                break;
            } else {
                valueEnd++;
            }
        }
        
        return json.substring(valueStart, valueEnd);
    }

    /**
     * 保存检测记录
     */
    private String saveDetectionRecord(String originalFilename, String filePath,
                                       String framesDir, String attentionMapPath, String result) {
        // 解析结果获取概率和帧数
        String probability = "0";
        String frameCount = "0";
        
        try {
            // 从 JSON 结果中提取信息
            int probStart = result.indexOf("\"probability\":\"") + 15;
            if (probStart > 14) {
                int probEnd = result.indexOf("\"", probStart);
                probability = result.substring(probStart, probEnd);
            }
            
            int framesStart = result.indexOf("\"frames\":\"") + 10;
            if (framesStart > 9) {
                int framesEnd = result.indexOf("\"", framesStart);
                frameCount = result.substring(framesStart, framesEnd);
            }
        } catch (Exception e) {
            log.warn("解析检测结果失败：{}", e.getMessage());
        }
        
        // 规范化路径 - 处理可能的相对路径或格式化问题
        String normalizedFramesDir = normalizePath(framesDir);
        String normalizedAttentionMapPath = normalizePath(attentionMapPath);
        
        log.info("保存检测记录：framesDir={}, attentionMapPath={}", normalizedFramesDir, normalizedAttentionMapPath);
        
        // 添加到服务
        String recordId = recordService.addRecord(originalFilename, filePath, normalizedFramesDir, normalizedAttentionMapPath);
        String resultText = Double.parseDouble(probability) < 0.5 ? "真实视频" : "AI 生成视频";
        recordService.updateResult(recordId, probability, resultText, frameCount);
        return recordId;
    }
    
    /**
     * 规范化路径 - 将相对路径转换为绝对路径，并修正路径分隔符
     */
    private String normalizePath(String path) {
        if (path == null || path.isEmpty()) {
            return "";
        }
        
        // 替换双反斜杠为单反斜杠
        String cleanedPath = path.replace("\\\\", "\\");
        
        // 如果是相对路径（不以盘符开头），则转换为绝对路径
        if (!cleanedPath.matches("^[a-zA-Z]:.*")) {
            // 移除开头的反斜杠
            cleanedPath = cleanedPath.replaceFirst("^\\\\+", "");
            // 添加上传目录前缀
            Path fullPath = Paths.get(uploadDir, cleanedPath).toAbsolutePath().normalize();
            cleanedPath = fullPath.toString();
            log.debug("路径规范化：{} -> {}", path, cleanedPath);
        }
        
        return cleanedPath;
    }

    private void enqueueTask(QueueTask task) {
        synchronized (queueLock) {
            pendingQueue.addLast(task);
            updateQueuePositionsLocked();
        }
        executor.submit(() -> processQueuedTask(task));
    }

    private void processQueuedTask(QueueTask task) {
        synchronized (queueLock) {
            pendingQueue.remove(task);
            currentTask = task;
            updateQueuePositionsLocked();
        }

        SseEmitter taskEmitter = activeEmitters.get(task.taskId);
        log.info("任务开始执行，taskId: {}, emitter: {}", task.taskId, taskEmitter != null ? "存在" : "null");

        if (taskEmitter == null) {
            log.error("无法获取 SSE emitter，taskId: {}", task.taskId);
            synchronized (queueLock) {
                currentTask = null;
                updateQueuePositionsLocked();
            }
            return;
        }

        try {
            task.startTimeMs = System.currentTimeMillis();
            Map<String, Object> startData = new HashMap<>();
            startData.put("type", "start");
            taskEmitter.send(startData);

            Consumer<String> progressCallback = (line) -> {
                if (taskEmitter != null) {
                    try {
                        Map<String, Object> progressData = new HashMap<>();
                        progressData.put("type", "progress");
                        progressData.put("message", line);

                        if (line.contains("/") && line.matches(".*\\d+/\\d+.*")) {
                            String[] parts = line.split("/");
                            if (parts.length >= 2) {
                                try {
                                    String currentStr = parts[0].replaceAll("[^\\d]", "");
                                    String totalStr = parts[1].replaceAll("[^0-9].*", "").replaceAll("[^\\d]", "");

                                    if (!currentStr.isEmpty() && !totalStr.isEmpty()) {
                                        int current = Integer.parseInt(currentStr);
                                        int total = Integer.parseInt(totalStr);
                                        progressData.put("frames", current);
                                        progressData.put("total", total);
                                        log.debug("进度更新：{}/{}", current, total);
                                    }
                                } catch (NumberFormatException e) {
                                    log.debug("解析帧数失败：{}", line, e);
                                }
                            }
                        } else if (line.contains("帧") && line.matches(".*\\d+.*")) {
                            progressData.put("framesInfo", line);
                        }

                        taskEmitter.send(progressData);
                    } catch (IOException e) {
                        log.error("发送进度更新失败", e);
                    }
                }
            };

            String result = pythonService.runDetection(task.filePath, progressCallback, task.outputDir, task.attnTopK, task.threshold, task.everyN);
            task.endTimeMs = System.currentTimeMillis();

            log.info("检测完成，准备发送结果到前端，taskId: {}", task.taskId);
            log.info("检测结果 JSON 长度：{}", result != null ? result.length() : 0);
            log.info("检测结果 JSON 前 200 字符：{}", result != null && result.length() > 200 ? result.substring(0, 200) : result);

            String attentionMapPath = extractJsonField(result, "attentionMapPath");
            String framesDir = extractJsonField(result, "framesDir");

            Map<String, Object> completeData = new HashMap<>();
            completeData.put("type", "complete");
            completeData.put("result", result);
            completeData.put("filename", task.originalFilename);
            completeData.put("filepath", task.outputDir);
            if (!attentionMapPath.isEmpty()) {
                try {
                    completeData.put("attentionMapUrl", toUploadUrl(Paths.get(attentionMapPath)));
                } catch (Exception e) {
                    log.warn("无法转换热力图路径为URL: {}", attentionMapPath);
                }
            }
            long queueDurationMs = Math.max(0L, task.startTimeMs - task.enqueueTimeMs);
            long detectDurationMs = Math.max(0L, task.endTimeMs - task.startTimeMs);
            completeData.put("detectStartTime", formatTime(task.startTimeMs));
            completeData.put("detectEndTime", formatTime(task.endTimeMs));
            completeData.put("queueDurationMs", queueDurationMs);
            completeData.put("detectDurationMs", detectDurationMs);

            log.info("准备发送 SSE 数据到前端:");
            log.info("  - result: {} (长度：{})", result != null ? "有值" : "null", result != null ? result.length() : 0);
            log.info("  - filename: {}", task.originalFilename);
            log.info("  - filepath: {}", task.outputDir);

            try {
                taskEmitter.send(completeData);
                log.info("成功发送完成消息到前端，taskId: {}", task.taskId);
            } catch (IOException e) {
                log.error("发送完成消息失败，taskId: {}", task.taskId, e);
            }

            taskEmitter.complete();
            activeEmitters.remove(task.taskId);
            log.info("SSE emitter 已完成，taskId: {}", task.taskId);

            String recordId = saveDetectionRecord(task.originalFilename, task.filePath,
                    framesDir, attentionMapPath, result);
            log.info("已保存检测记录：{}, 文件：{}", recordId, task.originalFilename);
        } catch (Exception e) {
            log.error("检测失败", e);
            if (taskEmitter != null) {
                Map<String, Object> errorData = new HashMap<>();
                errorData.put("type", "error");
                errorData.put("message", e.getMessage());
                try {
                    taskEmitter.send(errorData);
                } catch (IOException ex) {
                    log.error("发送错误消息失败", ex);
                }
                taskEmitter.completeWithError(e);
                activeEmitters.remove(task.taskId);
            }
        } finally {
            synchronized (queueLock) {
                currentTask = null;
                updateQueuePositionsLocked();
            }
        }
    }

    private void updateQueuePositionsLocked() {
        int index = 0;
        int base = currentTask != null ? 1 : 0;
        for (QueueTask task : pendingQueue) {
            int positionAhead = base + index;
            SseEmitter emitter = activeEmitters.get(task.taskId);
            if (emitter != null) {
                Map<String, Object> queuedData = new HashMap<>();
                queuedData.put("type", "queued");
                queuedData.put("position", positionAhead);
                try {
                    emitter.send(queuedData);
                } catch (IOException e) {
                    log.warn("发送排队信息失败，taskId: {}", task.taskId, e);
                }
            }
            index++;
        }
    }

    private Integer getQueuePosition(String taskId) {
        synchronized (queueLock) {
            if (currentTask != null && currentTask.taskId.equals(taskId)) {
                return 0;
            }
            int index = 0;
            int base = currentTask != null ? 1 : 0;
            for (QueueTask task : pendingQueue) {
                if (task.taskId.equals(taskId)) {
                    return base + index;
                }
                index++;
            }
            return null;
        }
    }

    private String formatTime(Long epochMs) {
        if (epochMs == null) {
            return "";
        }
        return LocalDateTime.ofInstant(Instant.ofEpochMilli(epochMs), ZoneId.systemDefault())
                .format(TIME_FORMATTER);
    }

    private static class QueueTask {
        private final String taskId;
        private final String originalFilename;
        private final String filePath;
        private final String outputDir;
        private final Integer attnTopK;
        private final Double threshold;
        private final Integer everyN;
        private final long enqueueTimeMs;
        private Long startTimeMs;
        private Long endTimeMs;

        private QueueTask(String taskId, String originalFilename, String filePath, String outputDir,
                          Integer attnTopK, Double threshold, Integer everyN) {
            this.taskId = taskId;
            this.originalFilename = originalFilename;
            this.filePath = filePath;
            this.outputDir = outputDir;
            this.attnTopK = attnTopK;
            this.threshold = threshold;
            this.everyN = everyN;
            this.enqueueTimeMs = System.currentTimeMillis();
        }
    }
}



