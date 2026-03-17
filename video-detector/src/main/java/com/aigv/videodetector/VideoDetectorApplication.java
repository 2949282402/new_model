package com.aigv.videodetector;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.MultipartConfigFactory;
import org.springframework.boot.web.servlet.ServletComponentScan;
import org.springframework.context.annotation.Bean;

import jakarta.servlet.MultipartConfigElement;
import java.io.File;

/**
 * AI 视频检测系统主应用类
 */
@SpringBootApplication
@ServletComponentScan
public class VideoDetectorApplication {

    public static void main(String[] args) {
        SpringApplication.run(VideoDetectorApplication.class, args);
    }

    /**
     * 配置 MultipartConfig，解决 Windows 上临时文件删除问题
     */
    @Bean
    public MultipartConfigElement multipartConfigElement() {
        MultipartConfigFactory factory = new MultipartConfigFactory();
        // 设置临时文件目录
        String tmpDir = System.getProperty("java.io.tmpdir") + "/spring-uploads";
        File tmpDirFile = new File(tmpDir);
        if (!tmpDirFile.exists()) {
            tmpDirFile.mkdirs();
        }
        factory.setLocation(tmpDir);
        return factory.createMultipartConfig();
    }
}
