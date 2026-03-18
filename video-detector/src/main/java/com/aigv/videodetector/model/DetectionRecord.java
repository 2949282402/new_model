package com.aigv.videodetector.model;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class DetectionRecord {
    private String id;
    private String originalFilename;
    private String filePath;
    private String framesDir;
    private String attentionMapPath;
    private String probability;
    private String result;
    private String frameCount;
    private String details;
    private Long queueDurationMs;
    private Long detectDurationMs;
    private Integer attnTopK;
    private Double threshold;
    private Integer everyN;
    private LocalDateTime detectTime;

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
