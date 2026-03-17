package com.aigv.videodetector;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.MultipartConfigFactory;
import org.springframework.boot.web.servlet.ServletComponentScan;
import org.springframework.context.annotation.Bean;

import javax.servlet.MultipartConfigElement;
import java.io.File;

/**
 * AI 视频检测系统主应用类
 */
@SpringBootApplication
@ServletComponentScan
public class VideoDetectorApplication {

    public static void main(String[] args) {
        SpringApplication.run(VideoDetectorApplication.class, args);
        System.out.println("\n========================================");
        System.out.println("AI 视频检测系统已启动！");
        System.out.println("访问地址：http://localhost:8080");
        System.out.println("注意：如果看到'临时文件清理失败'警告，这是正常的，不影响功能使用。");
        System.out.println("========================================\n");
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
