package com.aigv.videodetector.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.multipart.MultipartResolver;
import org.springframework.web.multipart.support.StandardServletMultipartResolver;

import jakarta.servlet.MultipartConfigElement;
import jakarta.servlet.ServletRegistration;
import java.io.File;

/**
 * 文件上传配置
 */
@Slf4j
@Configuration
public class MultipartConfig {

    /**
     * 配置 MultipartResolver
     */
    @Bean
    public MultipartResolver multipartResolver() {
        return new StandardServletMultipartResolver();
    }
}
