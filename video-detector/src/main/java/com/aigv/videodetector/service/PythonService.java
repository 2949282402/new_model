package com.aigv.videodetector.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Python 脚本调用服务
 */
@Slf4j
@Service
public class PythonService {

    @Value("${python.script.path}")
    private String scriptPath;

    @Value("${python.checkpoint.path}")
    private String checkpointPath;

    @Value("${python.save.attention:true}")
    private boolean saveAttention;

    // AI 检测阈值
    private static final double DETECTION_THRESHOLD = 0.1;

    /**
     * 执行 AI 检测脚本
     * 
     * @param videoPath 视频文件路径
     * @return 检测结果 JSON 字符串
     * @throws IOException IO 异常
     * @throws InterruptedException 中断异常
     */
    public String runDetection(String videoPath) throws IOException, InterruptedException {
        return runDetection(videoPath, null);
    }

    /**
     * 执行 AI 检测脚本（带进度回调）
     * 
     * @param videoPath 视频文件路径
     * @param progressCallback 进度回调函数，接收帧数信息
     * @return 检测结果 JSON 字符串
     * @throws IOException IO 异常
     * @throws InterruptedException 中断异常
     */
    public String runDetection(String videoPath, Consumer<String> progressCallback) throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, null, null);
    }
    
    /**
     * 执行 AI 检测脚本（带进度回调和输出目录）
     * 
     * @param videoPath 视频文件路径
     * @param progressCallback 进度回调函数，接收帧数信息
     * @param outputDir 输出目录（帧和热力图保存到此目录），如果为 null 则使用默认位置
     * @return 检测结果 JSON 字符串
     * @throws IOException IO 异常
     * @throws InterruptedException 中断异常
     */
    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir) throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, outputDir, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir, Integer attnTopK) throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, outputDir, attnTopK, null, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir, Integer attnTopK, Double threshold) throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, outputDir, attnTopK, threshold, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir, Integer attnTopK, Double threshold, Integer everyN) throws IOException, InterruptedException {
        log.info("开始执行 AI 检测脚本，视频路径：{}", videoPath);
        
        // 构建命令：python infer.py video.mp4 --checkpoint best.pth --threshold 0.1
        String resolvedScriptPath = resolvePath(scriptPath);
        String resolvedCheckpointPath = resolvePath(checkpointPath);
        java.util.List<String> command = new java.util.ArrayList<>();
        command.add("python");
        command.add("-u");  // 无缓冲输出
        command.add(resolvedScriptPath);
        command.add(videoPath);
        command.add("--checkpoint");
        command.add(resolvedCheckpointPath);
        command.add("--threshold");
        double effectiveThreshold = (threshold != null && threshold > 0 && threshold < 1)
                ? threshold
                : DETECTION_THRESHOLD;
        command.add(String.valueOf(effectiveThreshold));

        if (everyN != null) {
            command.add("--every_n");
            command.add(String.valueOf(everyN));
        }
        
        // 如果指定了输出目录，添加 --attn_output 参数
        if (outputDir != null && !outputDir.isEmpty()) {
            command.add("--attn_output");
            command.add(outputDir);
        }
        
        // 如果启用了 attention，添加 --save_attn 参数
        if (saveAttention) {
            command.add("--save_attn");
        }

        if (attnTopK != null && attnTopK > 0) {
            command.add("--attn_top_k");
            command.add(String.valueOf(attnTopK));
        }
        
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        
        // 设置环境变量，确保 Python 使用 UTF-8 编码
        processBuilder.environment().put("PYTHONIOENCODING", "UTF-8");
        
        // 设置工作目录
        processBuilder.directory(new java.io.File(resolvedScriptPath).getParentFile());
        
        // 重定向错误流到标准输出
        processBuilder.redirectErrorStream(true);
        
        // 启动进程
        Process process = processBuilder.start();
        
        // 读取输出
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append(System.lineSeparator());
                log.debug("Python 输出：{}", line);
                
                // 如果有进度回调，检查是否是进度信息
                if (progressCallback != null) {
                    // 检查是否包含帧数信息（格式：数字/数字）
                    if (line.contains("/") && line.matches(".*\\d+/\\d+.*")) {
                        progressCallback.accept(line);
                    } else if (line.contains("帧") && line.matches(".*\\d+.*")) {
                        // 或者包含"帧"字和数字
                        progressCallback.accept(line);
                    }
                }
            }
        }
        
        // 等待进程完成
        int exitCode = process.waitFor();
        
        if (exitCode != 0) {
            log.error("Python 脚本执行失败，退出码：{}, 输出：{}", exitCode, output.toString());
            throw new RuntimeException("AI 检测脚本执行失败：" + output.toString());
        }
        
        log.info("AI 检测脚本执行完成");
        return parseResult(output.toString());
    }

    private String resolvePath(String pathValue) {
        if (pathValue == null || pathValue.trim().isEmpty()) {
            return pathValue;
        }
        java.nio.file.Path rawPath = java.nio.file.Paths.get(pathValue);
        if (rawPath.isAbsolute()) {
            return rawPath.normalize().toString();
        }
        String userDir = System.getProperty("user.dir");
        java.nio.file.Path resolved = java.nio.file.Paths.get(userDir).resolve(rawPath).normalize();
        return resolved.toString();
    }

    /**
     * 解析 Python 脚本输出结果
     */
    private String parseResult(String output) {
        log.debug("解析 Python 输出：{}", output);
        
        // 提取概率和结果
        String aiProbability = extractValue(output, "AI生成概率:");
        if ("N/A".equals(aiProbability)) {
            aiProbability = extractValue(output, "AI 生成概率:");
        }
        String detectionResult = extractValue(output, "检测结果:");
        String frameCount = extractValue(output, "分析帧数:");
        
        // 提取 Attention 热力图路径和帧目录
        String attentionMapPath = extractAttentionMapPath(output);
        String framesDir = extractFramesDir(output);
        
        // 如果没有找到关键信息，尝试从输出中提取
        if ("N/A".equals(aiProbability)) {
            // 尝试从输出中提取数字（可能是概率值）
            String[] lines = output.split(System.lineSeparator());
            for (String line : lines) {
                // 查找包含数字的行，可能是概率
                if (line.matches(".*\\d+\\.\\d+.*")) {
                    String[] parts = line.split(":");
                    if (parts.length >= 2) {
                        String value = parts[1].trim();
                        // 如果是 0-1 之间的数字，可能是概率
                        try {
                            double prob = parseProbabilityValue(value);
                            if (prob >= 0 && prob <= 1) {
                                aiProbability = value;
                                detectionResult = prob < DETECTION_THRESHOLD ? "真实视频" : "AI 生成视频";
                            }
                        } catch (NumberFormatException e) {
                            // 忽略
                        }
                    }
                }
            }
        }
        
        // 如果还是 N/A，设置默认值
        if ("N/A".equals(aiProbability)) {
            aiProbability = "0.0";
        }
        double rawProbability = parseProbabilityValue(aiProbability);
        if ("N/A".equals(detectionResult)) {
            detectionResult = rawProbability < DETECTION_THRESHOLD ? "真实视频" : "AI 生成视频";
        }
        if ("N/A".equals(frameCount)) {
            frameCount = "未知";
        }
        
        // 概率映射：将原始概率映射到显示概率
        // 0.0-0.1 -> 0%-50%, 0.1-1.0 -> 50%-100%
        String displayProbability = mapProbabilityToDisplay(rawProbability);
        
        // 转义特殊字符，确保 JSON 格式正确
        // 注意：必须先转义反斜杠和双引号，再转义换行符
        String escapedDetails = output
                .replace("\\", "\\\\")    // 1. 先转义反斜杠 \ -> \\
                .replace("\"", "\\\"")    // 2. 再转义双引号 " -> \"
                .replace("\n", "\\n")       // 3. 转义换行符
                .replace("\r", "\\r");      // 4. 转义回车符
        
        log.debug("原始输出长度：{}", output.length());
        log.debug("转义后输出长度：{}", escapedDetails.length());
        log.debug("转义后的前 200 个字符：{}", escapedDetails.length() > 200 ? escapedDetails.substring(0, 200) : escapedDetails);
        
        // 转义 attentionMapPath
        String escapedAttentionPath = attentionMapPath
                .replace("\\", "\\\\")    // 转义反斜杠
                .replace("\"", "\\\"");   // 转义双引号
        
        log.info("提取到的热力图路径：{}", attentionMapPath);
        log.info("转义后的热力图路径：{}", escapedAttentionPath);
        
        String jsonResult = String.format("{\"success\":true,\"probability\":\"%s\",\"result\":\"%s\",\"frames\":\"%s\",\"details\":\"%s\",\"attentionMapPath\":\"%s\"}",
                displayProbability, detectionResult, frameCount, escapedDetails, escapedAttentionPath);
        
        log.info("生成的 JSON 结果长度：{}", jsonResult.length());
        log.debug("完整 JSON: {}", jsonResult);
        
        return jsonResult;
    }

    /**
     * 从输出中提取值
     */
    private String extractValue(String output, String key) {
        String[] lines = output.split(System.lineSeparator());
        for (String line : lines) {
            if (line.contains(key)) {
                String[] parts = line.split(":");
                if (parts.length >= 2) {
                    return parts[1].trim();
                }
            }
        }
        return "N/A";
    }

    private double parseProbabilityValue(String value) {
        if (value == null) {
            return 0.0;
        }
        Matcher matcher = Pattern.compile("([0-9]*\\.?[0-9]+)").matcher(value);
        if (matcher.find()) {
            return Double.parseDouble(matcher.group(1));
        }
        return 0.0;
    }

    /**
     * 从输出中提取 Attention 热力图路径
     */
    private String extractAttentionMapPath(String output) {
        // 查找包含注意力热力图路径的行
        // 格式可能为：[热力图] 已保存至：D:\path\to\attention_map.png
        // 优先查找 SVG 格式（矢量图，浏览器可直接显示）
        String[] lines = output.split(System.lineSeparator());
        
        // 先尝试查找 SVG 格式
        for (String line : lines) {
            if ((line.contains("热力图") || line.contains("attention") || line.contains("Attention")) 
                && line.endsWith(".svg")) {
                // 尝试提取路径
                int colonIndex = line.lastIndexOf(':');
                if (colonIndex != -1 && colonIndex < line.length() - 1) {
                    String path = line.substring(colonIndex + 1).trim();
                    // 验证是否是有效的文件路径
                    if (path.endsWith(".svg")) {
                        log.info("找到 SVG 热力图路径：{}", path);
                        return path;
                    }
                }
            }
        }
        
        // 如果没有 SVG，再尝试查找 PNG 格式（向后兼容）
        for (String line : lines) {
            if ((line.contains("热力图") || line.contains("attention") || line.contains("Attention")) 
                && line.endsWith(".png")) {
                // 尝试提取路径
                int colonIndex = line.lastIndexOf(':');
                if (colonIndex != -1 && colonIndex < line.length() - 1) {
                    String path = line.substring(colonIndex + 1).trim();
                    // 验证是否是有效的文件路径
                    if (path.endsWith(".png")) {
                        log.info("找到 PNG 热力图路径：{}", path);
                        return path;
                    }
                }
            }
        }
        
        log.debug("未找到热力图路径");
        return "";
    }

    /**
     * 从输出中提取帧目录路径
     */
    private String extractFramesDir(String output) {
        // 查找包含帧目录的行
        // 格式可能为：[抽帧] 已保存至目录：D:\path\to\frames\
        String[] lines = output.split(System.lineSeparator());
        for (String line : lines) {
            if (line.contains("帧") && (line.contains("目录") || line.contains("folder") || line.contains("dir"))) {
                // 尝试提取路径
                int colonIndex = line.lastIndexOf(':');
                if (colonIndex != -1 && colonIndex < line.length() - 1) {
                    String path = line.substring(colonIndex + 1).trim();
                    // 验证是否是有效的目录路径
                    if (!path.isEmpty() && !path.equals("N/A")) {
                        log.info("找到帧目录：{}", path);
                        return path;
                    }
                }
            }
        }
        log.debug("未找到帧目录");
        return "";
    }

    /**
     * 将原始概率映射到显示概率
     * 映射规则：
     * - 0.0 - 0.1  -> 0% - 50%
     * - 0.1 - 1.0  -> 50% - 100%
     */
    private String mapProbabilityToDisplay(double rawProbability) {
        double displayProb;
        
        if (rawProbability <= DETECTION_THRESHOLD) {
            // 0.0 - 0.1 映射到 0.0 - 0.5
            displayProb = rawProbability / DETECTION_THRESHOLD * 0.5;
        } else {
            // 0.1 - 1.0 映射到 0.5 - 1.0
            displayProb = 0.5 + (rawProbability - DETECTION_THRESHOLD) / (1.0 - DETECTION_THRESHOLD) * 0.5;
        }
        
        // 限制在 0-1 范围内
        displayProb = Math.max(0.0, Math.min(1.0, displayProb));
        
        return String.format("%.4f", displayProb);
    }
}
