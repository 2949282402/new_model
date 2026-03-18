package com.aigv.videodetector.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.function.Consumer;

@Slf4j
@Service
public class PythonService {
    private static final double DETECTION_THRESHOLD = 0.1;
    private static final String REAL_VIDEO = "真实视频";
    private static final String AI_VIDEO = "AI 生成视频";
    private static final String UNKNOWN = "N/A";
    private static final String UNKNOWN_FRAME_COUNT = "未知";
    private static final String SCRIPT_FAILURE = "AI 检测脚本执行失败: ";
    private static final String MASKED_PATH = "[PATH_REDACTED]";
    private static final char FULL_WIDTH_COLON = '：';
    private static final String FRAME_CHAR = "帧";
    private static final String DIRECTORY_WORD = "目录";
    private static final String HEATMAP_WORD = "热力图";

    private static final Pattern NUMBER_PATTERN = Pattern.compile("([0-9]*\\.?[0-9]+)");
    private static final Pattern WINDOWS_PATH_PATTERN = Pattern.compile("(?i)\\b[a-z]:[\\\\/][^\\s\"']+");
    private static final Pattern UNIX_PATH_PATTERN = Pattern.compile("(?<!:)(/[^\\s\"']+)+");
    private static final Pattern PROBABILITY_LINE_PATTERN = Pattern.compile(
            "(?i)(?:AI\\s*(?:generated\\s*)?(?:probability|score)|AI生成概率|AI 生成概率)\\s*[:：]\\s*(.+)");
    private static final Pattern RESULT_LINE_PATTERN = Pattern.compile(
            "(?i)(?:检测结果|result)\\s*[:：]\\s*(.+)");
    private static final Pattern FRAME_COUNT_LINE_PATTERN = Pattern.compile(
            "(?i)(?:分析帧数|frame\\s*count|frames\\s*analyzed)\\s*[:：]\\s*(.+)");

    @Value("${python.script.path}")
    private String scriptPath;

    @Value("${python.checkpoint.path}")
    private String checkpointPath;

    @Value("${python.save.attention:true}")
    private boolean saveAttention;

    public String runDetection(String videoPath) throws IOException, InterruptedException {
        return runDetection(videoPath, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback) throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, null, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir)
            throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, outputDir, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir, Integer attnTopK)
            throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, outputDir, attnTopK, null, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir, Integer attnTopK,
                               Double threshold) throws IOException, InterruptedException {
        return runDetection(videoPath, progressCallback, outputDir, attnTopK, threshold, null);
    }

    public String runDetection(String videoPath, Consumer<String> progressCallback, String outputDir,
                               Integer attnTopK, Double threshold, Integer everyN)
            throws IOException, InterruptedException {
        log.info("Starting detection script, videoPath={}", videoPath);

        String resolvedScriptPath = resolvePath(scriptPath);
        String resolvedCheckpointPath = resolvePath(checkpointPath);
        double effectiveThreshold = (threshold != null && threshold >= 0 && threshold <= 1)
                ? threshold
                : DETECTION_THRESHOLD;

        List<String> command = new ArrayList<>();
        command.add("python");
        command.add("-u");
        command.add(resolvedScriptPath);
        command.add(videoPath);
        command.add("--checkpoint");
        command.add(resolvedCheckpointPath);
        command.add("--threshold");
        command.add(String.valueOf(effectiveThreshold));

        if (everyN != null) {
            command.add("--every_n");
            command.add(String.valueOf(everyN));
        }

        if (outputDir != null && !outputDir.isEmpty()) {
            command.add("--attn_output");
            command.add(outputDir);
        }

        if (saveAttention) {
            command.add("--save_attn");
        }

        if (attnTopK != null && attnTopK > 0) {
            command.add("--attn_top_k");
            command.add(String.valueOf(attnTopK));
        }

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.environment().put("PYTHONIOENCODING", "UTF-8");
        processBuilder.directory(new java.io.File(resolvedScriptPath).getParentFile());
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append(System.lineSeparator());
                log.debug("Python output: {}", line);

                if (progressCallback != null && shouldReportProgress(line)) {
                    progressCallback.accept(line);
                }
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            log.error("Detection script failed, exitCode={}, output={}", exitCode, output);
            throw new RuntimeException(SCRIPT_FAILURE + output);
        }

        log.info("Detection script finished");
        return parseResult(output.toString(), effectiveThreshold);
    }

    private boolean shouldReportProgress(String line) {
        if (line == null || line.isBlank()) {
            return false;
        }
        if (line.contains("/") && line.matches(".*\\d+/\\d+.*")) {
            return true;
        }
        String lower = line.toLowerCase();
        return line.matches(".*\\d+.*")
                && (lower.contains("frame") || lower.contains("frames") || line.contains(FRAME_CHAR));
    }

    private String resolvePath(String pathValue) {
        if (pathValue == null || pathValue.trim().isEmpty()) {
            return pathValue;
        }

        Path rawPath = Paths.get(pathValue);
        if (rawPath.isAbsolute()) {
            return rawPath.normalize().toString();
        }

        String userDir = System.getProperty("user.dir");
        return Paths.get(userDir).resolve(rawPath).normalize().toString();
    }

    private String parseResult(String output, double effectiveThreshold) {
        String aiProbability = extractByPattern(output, PROBABILITY_LINE_PATTERN);
        String detectionResult = extractByPattern(output, RESULT_LINE_PATTERN);
        String frameCount = extractByPattern(output, FRAME_COUNT_LINE_PATTERN);
        String attentionMapPath = extractAttentionMapPath(output);
        String framesDir = extractFramesDir(output);

        if (UNKNOWN.equals(aiProbability)) {
            aiProbability = findFallbackProbability(output);
        }

        if (UNKNOWN.equals(aiProbability)) {
            aiProbability = "0.0";
        }

        double rawProbability = parseProbabilityValue(aiProbability);
        if (UNKNOWN.equals(detectionResult)) {
            detectionResult = rawProbability < effectiveThreshold ? REAL_VIDEO : AI_VIDEO;
        }
        if (UNKNOWN.equals(frameCount)) {
            frameCount = UNKNOWN_FRAME_COUNT;
        }

        String displayProbability = mapProbabilityToDisplay(rawProbability, effectiveThreshold);
        String sanitizedDetails = sanitizeSensitivePaths(output, framesDir, attentionMapPath);

        return String.format(
                "{\"success\":true,\"probability\":\"%s\",\"result\":\"%s\",\"frames\":\"%s\",\"details\":\"%s\",\"attentionMapPath\":\"%s\",\"framesDir\":\"%s\"}",
                displayProbability,
                escapeJsonValue(detectionResult),
                escapeJsonValue(frameCount),
                escapeJsonValue(sanitizedDetails),
                escapeJsonValue(attentionMapPath),
                escapeJsonValue(framesDir)
        );
    }

    private String extractByPattern(String output, Pattern pattern) {
        if (output == null || output.isEmpty()) {
            return UNKNOWN;
        }
        String[] lines = output.split(System.lineSeparator());
        for (String line : lines) {
            Matcher matcher = pattern.matcher(line.trim());
            if (matcher.find()) {
                return matcher.group(1).trim();
            }
        }
        return UNKNOWN;
    }

    private String findFallbackProbability(String output) {
        if (output == null || output.isEmpty()) {
            return UNKNOWN;
        }
        String[] lines = output.split(System.lineSeparator());
        for (String line : lines) {
            if (!line.matches(".*\\d+\\.\\d+.*")) {
                continue;
            }
            double prob = parseProbabilityValue(line);
            if (prob >= 0 && prob <= 1) {
                return String.valueOf(prob);
            }
        }
        return UNKNOWN;
    }

    private String sanitizeSensitivePaths(String output, String framesDir, String attentionMapPath) {
        if (output == null || output.isEmpty()) {
            return output;
        }

        String sanitized = output;
        sanitized = maskLiteralPath(sanitized, framesDir);
        sanitized = maskLiteralPath(sanitized, attentionMapPath);
        sanitized = WINDOWS_PATH_PATTERN.matcher(sanitized).replaceAll(MASKED_PATH);
        sanitized = UNIX_PATH_PATTERN.matcher(sanitized).replaceAll(MASKED_PATH);
        return sanitized;
    }

    private String maskLiteralPath(String text, String path) {
        if (text == null || text.isEmpty() || path == null || path.isEmpty()) {
            return text;
        }

        return text.replace(path, MASKED_PATH)
                .replace(path.replace("\\", "/"), MASKED_PATH)
                .replace(path.replace("/", "\\"), MASKED_PATH);
    }

    private double parseProbabilityValue(String value) {
        if (value == null) {
            return 0.0;
        }

        Matcher matcher = NUMBER_PATTERN.matcher(value);
        if (matcher.find()) {
            return Double.parseDouble(matcher.group(1));
        }
        return 0.0;
    }

    private String extractAttentionMapPath(String output) {
        String svgPath = findPathBySuffix(output, ".svg", true);
        if (!svgPath.isEmpty()) {
            return svgPath;
        }
        return findPathBySuffix(output, ".png", true);
    }

    private String extractFramesDir(String output) {
        if (output == null || output.isEmpty()) {
            return "";
        }
        String[] lines = output.split(System.lineSeparator());
        for (String line : lines) {
            String lower = line.toLowerCase();
            boolean frameLike = lower.contains("frame") || line.contains(FRAME_CHAR);
            boolean dirLike = lower.contains("dir") || lower.contains("folder") || line.contains(DIRECTORY_WORD);
            if (frameLike && dirLike) {
                String path = extractValueAfterColon(line);
                if (!path.isEmpty() && !UNKNOWN.equals(path)) {
                    return path;
                }
            }
        }
        return "";
    }

    private String findPathBySuffix(String output, String suffix, boolean preferHeatmapLine) {
        if (output == null || output.isEmpty()) {
            return "";
        }
        String fallback = "";
        String[] lines = output.split(System.lineSeparator());
        for (String line : lines) {
            String trimmed = line.trim();
            if (!trimmed.toLowerCase().endsWith(suffix)) {
                continue;
            }
            String lower = trimmed.toLowerCase();
            String path = extractValueAfterColon(trimmed);
            if (path.isEmpty() || !path.toLowerCase().endsWith(suffix)) {
                path = trimmed;
            }
            boolean heatmapLike = lower.contains("attention") || lower.contains("heatmap") || trimmed.contains(HEATMAP_WORD);
            if (!preferHeatmapLine || heatmapLike) {
                return path;
            }
            if (fallback.isEmpty()) {
                fallback = path;
            }
        }
        return fallback;
    }

    private String extractValueAfterColon(String line) {
        if (line == null || line.isEmpty()) {
            return "";
        }

        int fullWidthColon = line.indexOf(FULL_WIDTH_COLON);
        int asciiColon = line.indexOf(':');
        int colonIndex;

        if (fullWidthColon >= 0) {
            colonIndex = fullWidthColon;
        } else {
            colonIndex = asciiColon;
        }

        if (colonIndex == -1 || colonIndex >= line.length() - 1) {
            return "";
        }
        return line.substring(colonIndex + 1).trim();
    }

    private String escapeJsonValue(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }

    private String mapProbabilityToDisplay(double rawProbability, double effectiveThreshold) {
        double displayProb;

        if (effectiveThreshold <= 0.0) {
            displayProb = rawProbability <= 0.0 ? 0.0 : 0.5 + rawProbability * 0.5;
        } else if (effectiveThreshold >= 1.0) {
            displayProb = rawProbability >= 1.0 ? 1.0 : rawProbability * 0.5;
        } else if (rawProbability <= effectiveThreshold) {
            displayProb = rawProbability / effectiveThreshold * 0.5;
        } else {
            displayProb = 0.5 + (rawProbability - effectiveThreshold) / (1.0 - effectiveThreshold) * 0.5;
        }

        displayProb = Math.max(0.0, Math.min(1.0, displayProb));
        return String.format("%.4f", displayProb);
    }
}