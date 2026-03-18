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
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
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
    private static final int MIN_ATTN_TOP_K = 1;
    private static final int MAX_ATTN_TOP_K = 10;
    private static final double MIN_THRESHOLD = 0.0;
    private static final double MAX_THRESHOLD = 1.0;
    private static final int MIN_EVERY_N = 0;
    private static final int MAX_EVERY_N = 1000;

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
            log.info("Received detection params: attnTopK={}, threshold={}, everyN={}", attnTopK, threshold, everyN);

            String paramsError = validateDetectionParams(attnTopK, threshold, everyN);
            if (paramsError != null) {
                response.put("success", false);
                response.put("message", paramsError);
                return ResponseEntity.badRequest().body(response);
            }

            if (file.isEmpty()) {
                response.put("success", false);
                response.put("message", "请选择要上传的视频文件");
                return ResponseEntity.badRequest().body(response);
            }

            String originalFilename = sanitizeOriginalFilename(file.getOriginalFilename());
            if (originalFilename == null || !isValidVideoFile(originalFilename)) {
                response.put("success", false);
                response.put("message", "请上传有效的视频文件（mp4、avi、mov、mkv 等）");
                return ResponseEntity.badRequest().body(response);
            }

            Path uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
            Files.createDirectories(uploadPath);

            String uniqueId = UUID.randomUUID().toString();
            String videoFolderName = uniqueId + "_" + sanitizePathSegment(removeExtension(originalFilename));
            Path videoFolder = uploadPath.resolve(videoFolderName);
            Files.createDirectories(videoFolder);
            log.info("Created upload directory: {}", videoFolder.toAbsolutePath());

            String videoFilename = "original_video." + originalFilename.substring(originalFilename.lastIndexOf('.') + 1);
            Path filePath = videoFolder.resolve(videoFilename);

            try (InputStream inputStream = file.getInputStream()) {
                Files.copy(inputStream, filePath, StandardCopyOption.REPLACE_EXISTING);
            }

            long totalFrames;
            try {
                totalFrames = getVideoFrameCount(filePath);
            } catch (Exception e) {
                cleanupDirectoryQuietly(videoFolder);
                log.error("Failed to read total frame count: {}", filePath, e);
                response.put("success", false);
                response.put("message", "无法读取视频总帧数，请检查视频文件是否完整");
                return ResponseEntity.badRequest().body(response);
            }

            String frameWindowError = validateFrameWindow(everyN, totalFrames);
            if (frameWindowError != null) {
                cleanupDirectoryQuietly(videoFolder);
                response.put("success", false);
                response.put("message", frameWindowError);
                return ResponseEntity.badRequest().body(response);
            }

            String heatmapCountError = validateAttentionTopK(attnTopK, everyN, totalFrames);
            if (heatmapCountError != null) {
                cleanupDirectoryQuietly(videoFolder);
                response.put("success", false);
                response.put("message", heatmapCountError);
                return ResponseEntity.badRequest().body(response);
            }

            log.info("Starting async detection task");
            String taskId = UUID.randomUUID().toString();
            createEmitter(taskId);

            response.put("success", true);
            response.put("message", "检测已启动");
            response.put("taskId", taskId);
            response.put("filename", originalFilename);
            response.put("filepath", filePath.toString());

            QueueTask task = new QueueTask(taskId, originalFilename, filePath.toString(),
                    videoFolder.toString(), attnTopK, threshold, everyN);
            enqueueTask(task);

            return ResponseEntity.ok(response);
        } catch (IOException e) {
            log.error("File upload failed", e);
            response.put("success", false);
            response.put("message", "文件上传失败：" + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        } catch (Exception e) {
            log.error("Detection request failed", e);
            response.put("success", false);
            response.put("message", "检测失败：" + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    @PostMapping("/preview")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> previewVideo(@RequestParam("video") MultipartFile file) {
        Map<String, Object> response = new HashMap<>();
        log.info("Received preview request");
        try {
            if (file.isEmpty()) {
                log.warn("Preview request rejected because file is empty");
                response.put("success", false);
                response.put("message", "请选择要预览的视频文件");
                return ResponseEntity.badRequest().body(response);
            }

            String originalFilename = sanitizeOriginalFilename(file.getOriginalFilename());
            log.info("Preview request filename={}, size={} bytes, contentType={}",
                    originalFilename, file.getSize(), file.getContentType());

            if (originalFilename == null || !isValidVideoFile(originalFilename)) {
                log.warn("Preview request rejected because file type is invalid: {}", originalFilename);
                response.put("success", false);
                response.put("message", "请上传有效的视频文件（mp4、avi、mov、mkv 等）");
                return ResponseEntity.badRequest().body(response);
            }

            Path previewDir = Paths.get(uploadDir).toAbsolutePath().normalize().resolve("_preview");
            Files.createDirectories(previewDir);
            log.info("Preview directory: {}", previewDir.toAbsolutePath());

            String ext = originalFilename.contains(".")
                    ? originalFilename.substring(originalFilename.lastIndexOf('.'))
                    : ".mp4";
            String sourceName = "source_" + UUID.randomUUID() + ext;
            Path sourcePath = previewDir.resolve(sourceName);
            log.info("Saving preview source file: {}", sourcePath.toAbsolutePath());
            try (InputStream inputStream = file.getInputStream()) {
                Files.copy(inputStream, sourcePath, StandardCopyOption.REPLACE_EXISTING);
            }

            String previewName = "preview_" + UUID.randomUUID() + ".mp4";
            Path previewPath = previewDir.resolve(previewName);
            log.info("Preview output path: {}", previewPath.toAbsolutePath());

            try {
                log.info("Starting preview transcoding");
                createPreviewVideo(sourcePath, previewPath);
            } finally {
                log.info("Deleting preview source file: {}", sourcePath.toAbsolutePath());
                Files.deleteIfExists(sourcePath);
            }

            if (Files.exists(previewPath)) {
                log.info("Preview generated successfully: {}, size={} bytes", previewPath.toAbsolutePath(), Files.size(previewPath));
                String previewUrl = toUploadUrl(previewPath);
                log.info("Preview URL: {}", previewUrl);
                response.put("success", true);
                response.put("previewUrl", previewUrl);
                return ResponseEntity.ok(response);
            }

            log.error("Preview file was not generated: {}", previewPath.toAbsolutePath());
            response.put("success", false);
            response.put("message", "预览生成失败");
            return ResponseEntity.internalServerError().body(response);
        } catch (Exception e) {
            log.error("Preview generation failed", e);
            response.put("success", false);
            response.put("message", "预览生成失败：" + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }

    @PostMapping("/preview/stored")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> previewStoredVideo(@RequestParam("path") String rawPath) {
        Map<String, Object> response = new HashMap<>();
        log.info("Received stored preview request, path={}", rawPath);

        try {
            Path sourcePath = resolveManagedVideoPath(rawPath);
            if (!Files.exists(sourcePath) || !Files.isRegularFile(sourcePath)) {
                response.put("success", false);
                response.put("message", "视频文件不存在");
                return ResponseEntity.badRequest().body(response);
            }

            Path previewPath = buildStoredPreviewPath(sourcePath);
            if (!Files.exists(previewPath) || Files.size(previewPath) <= 0) {
                Files.createDirectories(previewPath.getParent());
                createPreviewVideo(sourcePath, previewPath);
            }

            response.put("success", true);
            response.put("previewUrl", toUploadUrl(previewPath));
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            log.warn("Stored preview request rejected: {}", e.getMessage());
            response.put("success", false);
            response.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(response);
        } catch (Exception e) {
            log.error("Stored preview generation failed", e);
            response.put("success", false);
            response.put("message", "预览生成失败：" + e.getMessage());
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
            log.info("History page loaded, recordCount={}", records.size());
            model.addAttribute("records", records);
            return "history";
        } catch (Exception e) {
            log.error("Failed to load history page", e);
            model.addAttribute("error", "加载历史记录失败：" + e.getMessage());
            model.addAttribute("records", new ArrayList<>());
            return "history";
        }
    }

    @PostMapping("/delete/{id}")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> deleteRecord(@PathVariable String id) {
        Map<String, Object> response = new HashMap<>();

        log.info("Received delete request, id={}", id);

        try {
            boolean deleted = recordService.deleteRecord(id);

            if (deleted) {
                log.info("Record deleted, id={}", id);
                response.put("success", true);
                response.put("message", "记录已删除");
            } else {
                log.warn("Record not found when deleting, id={}", id);
                response.put("success", false);
                response.put("message", "记录不存在");
            }
        } catch (Exception e) {
            log.error("Failed to delete record, id={}", id, e);
            response.put("success", false);
            response.put("message", "删除失败：" + e.getMessage());
        }

        return ResponseEntity.ok(response);
    }

    @GetMapping(value = "/progress/{taskId}", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter progress(@PathVariable String taskId) {
        SseEmitter emitter = activeEmitters.get(taskId);
        if (emitter == null) {
            if (isTaskPendingOrRunning(taskId)) {
                log.info("Recreating SSE emitter for active task, taskId={}", taskId);
                emitter = createEmitter(taskId);
            } else {
                log.warn("No SSE emitter and task is no longer active, taskId={}", taskId);
                SseEmitter emptyEmitter = new SseEmitter();
                emptyEmitter.complete();
                return emptyEmitter;
            }
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
            log.warn("Failed to send SSE connection state, taskId={}", taskId, e);
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

    private String sanitizeOriginalFilename(String originalFilename) {
        if (originalFilename == null) {
            return null;
        }

        String normalized = originalFilename.replace('\\', '/').trim();
        if (normalized.isEmpty()) {
            return null;
        }

        String fileNameOnly = Paths.get(normalized).getFileName().toString().trim();
        String cleaned = fileNameOnly.replaceAll("[\\p{Cntrl}]", "");
        return cleaned.isEmpty() ? null : cleaned;
    }

    private String removeExtension(String filename) {
        if (filename == null || filename.isEmpty()) {
            return "video";
        }
        int dotIndex = filename.lastIndexOf('.');
        if (dotIndex <= 0) {
            return filename;
        }
        return filename.substring(0, dotIndex);
    }

    private String getFileExtension(String filename) {
        if (filename == null || filename.isEmpty()) {
            return "mp4";
        }
        int dotIndex = filename.lastIndexOf('.');
        if (dotIndex < 0 || dotIndex == filename.length() - 1) {
            return "mp4";
        }
        return filename.substring(dotIndex + 1);
    }

    private String sanitizePathSegment(String value) {
        if (value == null || value.trim().isEmpty()) {
            return "video";
        }

        String sanitized = value.replaceAll("[\\\\/:*?\"<>|]", "_")
                .replaceAll("\\s+", " ")
                .trim();
        return sanitized.isEmpty() ? "video" : sanitized;
    }

    private String validateDetectionParams(Integer attnTopK, Double threshold, Integer everyN) {
        if (attnTopK != null && (attnTopK < MIN_ATTN_TOP_K || attnTopK > MAX_ATTN_TOP_K)) {
            return String.format("热力图数量需在 %d 到 %d 之间", MIN_ATTN_TOP_K, MAX_ATTN_TOP_K);
        }

        if (threshold != null && (threshold < MIN_THRESHOLD || threshold > MAX_THRESHOLD)) {
            return String.format("判断阈值需在 %.1f 到 %.1f 之间", MIN_THRESHOLD, MAX_THRESHOLD);
        }

        if (everyN != null && (everyN < MIN_EVERY_N || everyN > MAX_EVERY_N)) {
            return String.format("抽帧间隔需在 %d 到 %d 之间", MIN_EVERY_N, MAX_EVERY_N);
        }

        return null;
    }

    private String validateFrameWindow(Integer everyN, long totalFrames) {
        int stride = everyN != null ? everyN : 0;
        if (stride + 4 >= totalFrames) {
            return String.format(
                    "抽帧间隔+4需要小于总帧数，当前抽帧间隔=%d，总帧数=%d",
                    stride,
                    totalFrames
            );
        }
        return null;
    }

    private String validateAttentionTopK(Integer attnTopK, Integer everyN, long totalFrames) {
        if (attnTopK == null) {
            return null;
        }

        long maxHeatmaps = calculateMaxHeatmapCount(totalFrames, everyN);
        if (maxHeatmaps <= 0) {
            return "当前视频无法生成有效热力图";
        }

        if (attnTopK > maxHeatmaps) {
            return String.format(
                    "热力图数量不能超过当前视频在该抽帧间隔下可生成的数量，当前最多可生成 %d 张",
                    maxHeatmaps
            );
        }

        return null;
    }

    private long calculateMaxHeatmapCount(long totalFrames, Integer everyN) {
        long analyzableFrames = totalFrames - 4;
        if (analyzableFrames <= 0) {
            return 0;
        }

        long stride = (everyN != null && everyN > 0) ? everyN : 1;
        return (analyzableFrames + stride - 1) / stride;
    }

    private long getVideoFrameCount(Path videoFile) throws IOException, InterruptedException {

        long count = runFfprobeForFrameCount(videoFile, true);
        if (count > 0) {
            return count;
        }

        count = runFfprobeForFrameCount(videoFile, false);
        if (count > 0) {
            return count;
        }

        throw new IOException("FFprobe 无法读取视频总帧数");
    }

    private long runFfprobeForFrameCount(Path videoFile, boolean countFrames) throws IOException, InterruptedException {
        String ffprobeExecutable = resolveFfprobeExecutable();
        List<String> command = new ArrayList<>();
        command.add(ffprobeExecutable);
        command.add("-v");
        command.add("error");
        command.add("-select_streams");
        command.add("v:0");
        if (countFrames) {
            command.add("-count_frames");
        }
        command.add("-show_entries");
        command.add(countFrames ? "stream=nb_read_frames" : "stream=nb_frames");
        command.add("-of");
        command.add("default=nokey=1:noprint_wrappers=1");
        command.add(videoFile.toString());

        ProcessBuilder builder = new ProcessBuilder(command);
        builder.redirectErrorStream(true);
        Process process = builder.start();
        String output;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            output = reader.lines()
                    .map(String::trim)
                    .filter(line -> !line.isEmpty() && !"N/A".equalsIgnoreCase(line))
                    .findFirst()
                    .orElse("");
        }

        int exitCode = process.waitFor();
        if (exitCode != 0 || output.isEmpty()) {
            return -1L;
        }

        try {
            return Long.parseLong(output);
        } catch (NumberFormatException e) {
            return -1L;
        }
    }

    private String resolveFfprobeExecutable() throws IOException {
        String ffmpegExecutable = resolveFfmpegExecutable();
        try {
            Path ffmpegResolvedPath = Paths.get(ffmpegExecutable);
            if (Files.isRegularFile(ffmpegResolvedPath)) {
                boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
                String ffprobeName = isWindows ? "ffprobe.exe" : "ffprobe";
                Path parent = ffmpegResolvedPath.getParent();
                if (parent != null) {
                    Path sibling = parent.resolve(ffprobeName);
                    if (Files.isRegularFile(sibling)) {
                        return sibling.toString();
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return "ffprobe";
    }

    private void cleanupDirectoryQuietly(Path directory) {
        if (directory == null || !Files.exists(directory)) {
            return;
        }
        try (java.util.stream.Stream<Path> walk = Files.walk(directory)) {
            walk
                    .sorted(java.util.Comparator.reverseOrder())
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            log.warn("清理临时目录失败: {}", path, e);
                        }
                    });
        } catch (IOException e) {
            log.warn("遍历临时目录失败: {}", directory, e);
        }
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

        log.info("开始 FFmpeg 预览转码: {}", String.join(" ", command));
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
        log.info("FFmpeg 转码结束，exitCode={}, 输出片段: {}", exitCode,
                output.length() > 1000 ? output.substring(output.length() - 1000) : output.toString());
        if (exitCode != 0 || !Files.exists(outputFile)) {
            throw new IOException("FFmpeg 转码失败: " + output.toString().trim());
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
                log.info("使用 FFmpeg 路径: {}", candidate);
                return candidate.toString();
            }
            if (Files.isDirectory(candidate)) {
                Path exe = candidate.resolve(isWindows ? "ffmpeg.exe" : "ffmpeg");
                if (Files.isRegularFile(exe)) {
                    log.info("使用 FFmpeg 路径: {}", exe);
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
        log.warn("无法将路径转为 uploads URL: {}", absolute);
        return "";
    }

    private Path resolveManagedVideoPath(String rawPath) {
        if (rawPath == null || rawPath.trim().isEmpty()) {
            throw new IllegalArgumentException("缺少视频路径");
        }

        Path uploadRoot = Paths.get(uploadDir).toAbsolutePath().normalize();
        Path candidate = Paths.get(rawPath.trim());
        Path normalized = candidate.isAbsolute()
                ? candidate.toAbsolutePath().normalize()
                : uploadRoot.resolve(candidate).toAbsolutePath().normalize();

        if (!normalized.startsWith(uploadRoot)) {
            throw new IllegalArgumentException("视频路径不合法");
        }
        return normalized;
    }

    private Path buildStoredPreviewPath(Path sourcePath) {
        String filename = sourcePath.getFileName() != null ? sourcePath.getFileName().toString() : "video";
        int dotIndex = filename.lastIndexOf('.');
        String basename = dotIndex > 0 ? filename.substring(0, dotIndex) : filename;
        return sourcePath.resolveSibling(basename + "_preview.mp4");
    }

    private Path preferRasterAttentionMap(Path attentionPath) {
        if (attentionPath == null) {
            return null;
        }
        String filename = attentionPath.getFileName() != null ? attentionPath.getFileName().toString() : "";
        if (filename.toLowerCase().endsWith(".svg")) {
            Path pngPath = attentionPath.resolveSibling(filename.substring(0, filename.length() - 4) + ".png");
            if (Files.exists(pngPath)) {
                return pngPath;
            }
        }
        return attentionPath;
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
                                       String framesDir, String attentionMapPath, String result,
                                       Long queueDurationMs, Long detectDurationMs,
                                       Integer attnTopK, Double threshold, Integer everyN) {
        // 解析结果获取概率和帧数
        String probability = "0";
        String frameCount = "0";
        String details = "";
        String resultText = "";

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
            details = extractJsonField(result, "details")
                    .replace("\\n", System.lineSeparator())
                    .replace("\\r", "")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\");
            resultText = extractJsonField(result, "result");
        } catch (Exception e) {
            log.warn("解析检测结果失败：{}", e.getMessage());
        }

        // 规范化路径 - 处理可能的相对路径或格式化问题
        String normalizedFramesDir = normalizePath(framesDir);
        String normalizedAttentionMapPath = normalizePath(attentionMapPath);

        log.info("保存检测记录：framesDir={}, attentionMapPath={}", normalizedFramesDir, normalizedAttentionMapPath);

        // 添加到服务
        String recordId = recordService.addRecord(originalFilename, filePath, normalizedFramesDir, normalizedAttentionMapPath);
        if (resultText == null || resultText.isBlank()) {
            resultText = Double.parseDouble(probability) < 0.5 ? "真实视频" : "AI 生成视频";
        }
        recordService.updateResult(recordId, probability, resultText, frameCount, details,
                queueDurationMs, detectDurationMs, attnTopK, threshold, everyN);
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
        Path rawPath = Paths.get(cleanedPath);
        if (!rawPath.isAbsolute()) {
            // 移除开头的反斜杠
            cleanedPath = cleanedPath.replaceFirst("^\\\\+", "");
            // 添加上传目录前缀
            Path fullPath = Paths.get(uploadDir).toAbsolutePath().normalize().resolve(cleanedPath).normalize();
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

        log.info("Task started, taskId={}, emitter={}", task.taskId,
                activeEmitters.containsKey(task.taskId) ? "present" : "missing");

        try {
            task.startTimeMs = System.currentTimeMillis();
            Map<String, Object> startData = new HashMap<>();
            startData.put("type", "start");
            sendEvent(task.taskId, startData);

            Consumer<String> progressCallback = (line) -> {
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
                            }
                        } catch (NumberFormatException e) {
                            log.debug("Failed to parse progress line: {}", line, e);
                        }
                    }
                } else if ((line.contains("帧") || line.toLowerCase().contains("frame")) && line.matches(".*\\d+.*")) {
                    progressData.put("framesInfo", line);
                }

                sendEvent(task.taskId, progressData);
            };

            String result = pythonService.runDetection(task.filePath, progressCallback, task.outputDir,
                    task.attnTopK, task.threshold, task.everyN);
            task.endTimeMs = System.currentTimeMillis();

            String attentionMapPath = extractJsonField(result, "attentionMapPath");
            String framesDir = extractJsonField(result, "framesDir");
            if (framesDir == null || framesDir.isBlank()) {
                framesDir = task.outputDir;
            }

            Map<String, Object> completeData = new HashMap<>();
            completeData.put("type", "complete");
            completeData.put("result", result);
            completeData.put("filename", task.originalFilename);
            completeData.put("filepath", task.filePath);
            try {
                String videoUrl = toUploadUrl(Paths.get(task.filePath));
                if (!videoUrl.isEmpty()) {
                    completeData.put("videoUrl", videoUrl);
                }
            } catch (Exception e) {
                log.warn("Failed to convert video path to URL: {}", task.filePath, e);
            }
            if (!attentionMapPath.isEmpty()) {
                try {
                    String attentionMapUrl = toUploadUrl(preferRasterAttentionMap(Paths.get(attentionMapPath)));
                    if (!attentionMapUrl.isEmpty()) {
                        completeData.put("attentionMapUrl", attentionMapUrl);
                    }
                } catch (Exception e) {
                    log.warn("Failed to convert attention map path to URL: {}", attentionMapPath, e);
                }
            }

            long queueDurationMs = Math.max(0L, task.startTimeMs - task.enqueueTimeMs);
            long detectDurationMs = Math.max(0L, task.endTimeMs - task.startTimeMs);
            completeData.put("detectStartTime", formatTime(task.startTimeMs));
            completeData.put("detectEndTime", formatTime(task.endTimeMs));
            completeData.put("queueDurationMs", queueDurationMs);
            completeData.put("detectDurationMs", detectDurationMs);
            completeData.put("attnTopK", task.attnTopK);
            completeData.put("threshold", task.threshold);
            completeData.put("everyN", task.everyN);

            sendEvent(task.taskId, completeData);
            completeEmitter(task.taskId, null);

            String recordId = saveDetectionRecord(task.originalFilename, task.filePath,
                    framesDir, attentionMapPath, result, queueDurationMs, detectDurationMs,
                    task.attnTopK, task.threshold, task.everyN);
            log.info("Detection record saved, id={}, filename={}", recordId, task.originalFilename);
        } catch (Exception e) {
            log.error("Detection failed", e);
            Map<String, Object> errorData = new HashMap<>();
            errorData.put("type", "error");
            errorData.put("message", e.getMessage());
            sendEvent(task.taskId, errorData);
            completeEmitter(task.taskId, e);
        } finally {
            synchronized (queueLock) {
                currentTask = null;
                updateQueuePositionsLocked();
            }
        }
    }

    private SseEmitter createEmitter(String taskId) {
        SseEmitter emitter = new SseEmitter(5 * 60 * 1000L);
        activeEmitters.put(taskId, emitter);

        emitter.onCompletion(() -> {
            log.info("SSE completed, taskId={}", taskId);
            activeEmitters.remove(taskId, emitter);
        });

        emitter.onTimeout(() -> {
            log.info("SSE timed out, taskId={}", taskId);
            activeEmitters.remove(taskId, emitter);
        });

        emitter.onError((e) -> {
            log.warn("SSE error, taskId={}", taskId, e);
            activeEmitters.remove(taskId, emitter);
        });
        return emitter;
    }

    private boolean isTaskPendingOrRunning(String taskId) {
        synchronized (queueLock) {
            if (currentTask != null && currentTask.taskId.equals(taskId)) {
                return true;
            }
            for (QueueTask task : pendingQueue) {
                if (task.taskId.equals(taskId)) {
                    return true;
                }
            }
            return false;
        }
    }

    private boolean sendEvent(String taskId, Map<String, Object> data) {
        SseEmitter emitter = activeEmitters.get(taskId);
        if (emitter == null) {
            return false;
        }
        try {
            emitter.send(data);
            return true;
        } catch (IOException e) {
            log.warn("Failed to send SSE event, taskId={}, type={}", taskId, data.get("type"), e);
            return false;
        }
    }

    private void completeEmitter(String taskId, Exception error) {
        SseEmitter emitter = activeEmitters.remove(taskId);
        if (emitter == null) {
            return;
        }
        try {
            if (error == null) {
                emitter.complete();
            } else {
                emitter.completeWithError(error);
            }
        } catch (Exception completeError) {
            log.debug("Failed to close emitter, taskId={}", taskId, completeError);
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