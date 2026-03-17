package com.aigv.videodetector.service;

import com.aigv.videodetector.model.DetectionRecord;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.beans.factory.annotation.Value;

/**
 * 检测记录管理服务
 */
@Slf4j
@Service
public class DetectionRecordService {
    
    // 内存中的记录存储
    private final Map<String, DetectionRecord> records = new ConcurrentHashMap<>();
    
    // 持久化文件路径（运行时解析）
    private Path recordsFilePath;

    @Value("${records.file:}")
    private String recordsFileOverride;

    @Value("${upload.dir:}")
    private String uploadDir;
    
    // 日期时间格式化器
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    
    /**
     * 初始化 - 加载历史记录
     */
    @PostConstruct
    public void init() {
        recordsFilePath = resolveRecordsFilePath();
        log.info("历史记录文件路径: {}", recordsFilePath);
        loadRecords();
        cleanupPendingUploads();
    }

    private Path resolveRecordsFilePath() {
        if (recordsFileOverride != null && !recordsFileOverride.trim().isEmpty()) {
            return Paths.get(recordsFileOverride).toAbsolutePath().normalize();
        }
        if (uploadDir != null && !uploadDir.trim().isEmpty()) {
            Path uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
            Path parent = uploadPath.getParent();
            if (parent != null) {
                return parent.resolve("detection_records.txt");
            }
        }
        return Paths.get("detection_records.txt").toAbsolutePath().normalize();
    }

    private void ensureRecordsFileParent() {
        if (recordsFilePath == null) {
            return;
        }
        Path parent = recordsFilePath.getParent();
        if (parent != null) {
            try {
                Files.createDirectories(parent);
            } catch (IOException e) {
                log.warn("创建历史记录目录失败: {}", parent, e);
            }
        }
    }

    private void cleanupPendingUploads() {
        if (uploadDir == null || uploadDir.trim().isEmpty()) {
            return;
        }
        Path uploadPath = Paths.get(uploadDir).toAbsolutePath().normalize();
        if (!Files.exists(uploadPath) || !Files.isDirectory(uploadPath)) {
            return;
        }

        Set<Path> keepFolders = new HashSet<>();
        for (DetectionRecord record : records.values()) {
            String filePath = record.getFilePath();
            if (filePath == null || filePath.isEmpty()) {
                continue;
            }
            Path path = Paths.get(filePath).toAbsolutePath().normalize();
            Path folder = Files.isDirectory(path) ? path : path.getParent();
            if (folder != null) {
                keepFolders.add(folder);
            }
        }

        try {
            Files.list(uploadPath)
                    .filter(Files::isDirectory)
                    .forEach(path -> {
                        if (!keepFolders.contains(path.toAbsolutePath().normalize())) {
                            try {
                                Files.walk(path)
                                        .sorted(java.util.Comparator.reverseOrder())
                                        .forEach(p -> {
                                            try {
                                                Files.deleteIfExists(p);
                                            } catch (IOException e) {
                                                log.warn("删除文件失败: {}", p, e);
                                            }
                                        });
                                log.info("清理未完成检测目录: {}", path);
                            } catch (IOException e) {
                                log.warn("清理目录失败: {}", path, e);
                            }
                        }
                    });
        } catch (IOException e) {
            log.warn("扫描上传目录失败: {}", uploadPath, e);
        }
    }
    
    /**
     * 添加检测记录
     */
    public String addRecord(String originalFilename, String filePath, 
                         String framesDir, String attentionMapPath) {
        String id = UUID.randomUUID().toString();
        DetectionRecord record = new DetectionRecord(id, originalFilename, filePath, 
                                                     framesDir, attentionMapPath);
        records.put(id, record);
        saveRecord(record);
        log.info("已添加检测记录：{}, 文件名：{}", id, originalFilename);
        return id;
    }
    
    /**
     * 更新检测结果
     */
    public void updateResult(String id, String probability, String result, String frameCount) {
        DetectionRecord record = records.get(id);
        if (record != null) {
            // 将概率转换为百分比格式（例如：0.7222 -> 72.22%）
            String percentageProbability = probability;
            try {
                double prob = Double.parseDouble(probability);
                percentageProbability = String.format("%.2f", prob * 100);
            } catch (NumberFormatException e) {
                log.warn("概率转换失败：{}", probability);
            }
            
            record.setProbability(percentageProbability);
            record.setResult(result);
            record.setFrameCount(frameCount);
            // 更新文件中的记录
            rebuildRecordsFile();
            log.info("已更新检测结果：{}", id);
        }
    }
    
    /**
     * 获取所有记录（按时间倒序）
     */
    public List<DetectionRecord> getAllRecords() {
        if (records.isEmpty()) {
            log.info("记录为空，尝试从文件加载");
            loadRecords();
        }
        List<DetectionRecord> list = new ArrayList<>(records.values());
        list.sort((a, b) -> b.getDetectTime().compareTo(a.getDetectTime()));
        log.info("返回 {} 条历史记录", list.size());
        return list;
    }
    
    /**
     * 根据 ID 获取记录
     */
    public DetectionRecord getRecord(String id) {
        return records.get(id);
    }
    
    /**
     * 删除记录及其关联的文件
     */
    public boolean deleteRecord(String id) {
        DetectionRecord record = records.get(id);
        if (record == null) {
            log.warn("记录不存在：{}", id);
            return false;
        }
        
        log.info("开始删除记录：{}, 文件路径：{}", id, record.getFilePath());
        
        // 先删除文件
        deleteFiles(record);
        
        // 从内存中移除记录（无论文件是否删除成功）
        records.remove(id);
        
        // 从持久化文件中移除
        try {
            rebuildRecordsFile();
            log.info("已删除记录：{}", id);
            return true;
        } catch (Exception e) {
            log.error("重建记录文件失败，但记录已从内存中移除", e);
            // 记录已从内存中移除，返回成功
            return true;
        }
    }
    
    /**
     * 删除记录关联的所有文件（统一文件夹模式）
     */
    private void deleteFiles(DetectionRecord record) {
        if (record.getFilePath() == null || record.getFilePath().isEmpty()) {
            log.warn("记录的文件路径为空，跳过文件删除");
            return;
        }
        
        try {
            Path videoFolderPath = Paths.get(record.getFilePath());
            
            if (!Files.exists(videoFolderPath)) {
                log.warn("视频文件夹不存在：{}", record.getFilePath());
                return;
            }
            
            if (!Files.isDirectory(videoFolderPath)) {
                log.warn("路径不是文件夹：{}", record.getFilePath());
                // 如果是单个文件，直接删除
                Files.deleteIfExists(videoFolderPath);
                log.info("已删除文件：{}", record.getFilePath());
                return;
            }
            
            log.info("开始删除视频文件夹：{}", videoFolderPath);
            
            // 递归删除整个文件夹及其内容
            Files.walk(videoFolderPath)
                .sorted(java.util.Comparator.reverseOrder())
                .forEach(path -> {
                    try {
                        // 给 Windows 一些时间释放文件锁
                        Thread.sleep(10);
                        boolean deleted = Files.deleteIfExists(path);
                        if (deleted) {
                            log.debug("成功删除：{}", path);
                        } else {
                            log.warn("文件不存在：{}", path);
                        }
                    } catch (IOException e) {
                        log.error("删除文件失败：{} - {}", path, e.getMessage());
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        log.error("删除过程被中断", e);
                    }
                });
            
            // 最后尝试删除文件夹本身
            if (Files.exists(videoFolderPath)) {
                Files.deleteIfExists(videoFolderPath);
            }
            
            log.info("视频文件夹删除完成：{}", record.getFilePath());
            
        } catch (IOException e) {
            log.error("删除视频文件夹时发生 IO 异常：{}", record.getFilePath(), e);
        } catch (Exception e) {
            log.error("删除视频文件夹时发生意外错误：{}", record.getFilePath(), e);
        }
    }
    
    /**
     * 保存单条记录到文件
     */
    private synchronized void saveRecord(DetectionRecord record) {
        ensureRecordsFileParent();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(recordsFilePath.toFile(), true))) {
            String line = String.format("%s|%s|%s|%s|%s|%s|%s|%s|%s",
                    record.getId(),
                    record.getOriginalFilename(),
                    record.getFilePath(),
                    record.getFramesDir(),
                    record.getAttentionMapPath(),
                    record.getProbability() != null ? record.getProbability() : "",
                    record.getResult() != null ? record.getResult() : "",
                    record.getFrameCount() != null ? record.getFrameCount() : "",
                    record.getDetectTime().format(FORMATTER));
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            log.error("保存检测记录失败", e);
        }
    }
    
    /**
     * 从文件加载所有记录
     */
    private void loadRecords() {
        if (recordsFilePath == null) {
            log.info("历史记录文件路径未初始化");
            return;
        }
        File file = recordsFilePath.toFile();
        if (!file.exists()) {
            log.info("未找到历史记录文件：{}", recordsFilePath);
            return;
        }
            
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            int count = 0;
            int errorCount = 0;
                
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                    
                String[] parts = line.split("\\|", -1);
                if (parts.length < 9) {
                    log.warn("无效的记录格式，字段数不足：{}, 字段数：{}", line, parts.length);
                    errorCount++;
                    continue;
                }
                    
                try {
                    DetectionRecord record = new DetectionRecord();
                    record.setId(parts[0]);
                    record.setOriginalFilename(parts[1]);
                    record.setFilePath(parts[2]);
                    record.setFramesDir(parts[3]);
                    record.setAttentionMapPath(parts[4]);
                    record.setProbability(parts[5].isEmpty() ? null : parts[5]);
                    record.setResult(parts[6].isEmpty() ? null : parts[6]);
                    record.setFrameCount(parts[7].isEmpty() ? null : parts[7]);
                    record.setDetectTime(LocalDateTime.parse(parts[8], FORMATTER));
                        
                    records.put(record.getId(), record);
                    count++;
                } catch (Exception e) {
                    log.error("解析单条记录失败：{}, 错误：{}", line, e.getMessage());
                    errorCount++;
                }
            }
                
            log.info("已加载 {} 条历史检测记录，跳过 {} 条错误记录", count, errorCount);
                
        } catch (IOException e) {
            log.error("加载历史记录失败：{}", recordsFilePath, e);
        } catch (Exception e) {
            log.error("解析历史记录失败：{}", recordsFilePath, e);
        }
    }
    
    /**
     * 重建记录文件（用于删除后）
     */
    private synchronized void rebuildRecordsFile() {
        ensureRecordsFileParent();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(recordsFilePath.toFile()))) {
            for (DetectionRecord record : records.values()) {
                String line = String.format("%s|%s|%s|%s|%s|%s|%s|%s|%s",
                        record.getId(),
                        record.getOriginalFilename(),
                        record.getFilePath(),
                        record.getFramesDir(),
                        record.getAttentionMapPath(),
                        record.getProbability() != null ? record.getProbability() : "",
                        record.getResult() != null ? record.getResult() : "",
                        record.getFrameCount() != null ? record.getFrameCount() : "",
                        record.getDetectTime().format(FORMATTER));
                writer.write(line);
                writer.newLine();
            }
            log.debug("已重建记录文件，共 {} 条记录", records.size());
        } catch (IOException e) {
            log.error("重建记录文件失败", e);
        }
    }
}
