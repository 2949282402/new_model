package com.aigv.videodetector.model;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * 检测记录实体类
 */
@Data
public class DetectionRecord {
    
    private String id;              // 记录 ID
    private String originalFilename; // 原始文件名
    private String filePath;        // 文件路径
    private String framesDir;       // 帧目录路径
    private String attentionMapPath; // 热力图路径
    private String probability;     // 概率
    private String result;          // 检测结果
    private String frameCount;      // 帧数
    private LocalDateTime detectTime; // 检测时间
    
    public DetectionRecord() {
        this.detectTime = LocalDateTime.now();
    }
    
    public DetectionRecord(String id, String originalFilename, String filePath, 
                          String framesDir, String attentionMapPath) {
        this.id = id;
        this.originalFilename = originalFilename;
        this.filePath = filePath;
        this.framesDir = framesDir;
        this.attentionMapPath = attentionMapPath;
        this.detectTime = LocalDateTime.now();
    }
}
