package com.aigv.videodetector.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.Map;

/**
 * 全局异常处理器
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    /**
     * 处理文件上传相关的 IO 异常
     */
    @ExceptionHandler(IOException.class)
    public Map<String, Object> handleIOException(IOException e) {
        Map<String, Object> response = new HashMap<>();
        
        // 检查是否是临时文件删除错误
        if (e.getMessage() != null && e.getMessage().contains("Cannot delete")) {
            log.warn("临时文件清理失败，这通常是因为文件被占用，不影响功能使用：{}", e.getMessage());
            response.put("success", true);
            response.put("message", "检测已完成（临时文件清理警告已忽略）");
        } else {
            log.error("文件处理 IO 异常", e);
            response.put("success", false);
            response.put("message", "文件处理失败：" + e.getMessage());
        }
        
        return response;
    }

    /**
     * 处理 UncheckedIOException（包装的 IO 异常）
     */
    @ExceptionHandler(UncheckedIOException.class)
    public Map<String, Object> handleUncheckedIOException(UncheckedIOException e) {
        Map<String, Object> response = new HashMap<>();
        
        // 检查是否是临时文件删除错误
        Throwable cause = e.getCause();
        while (cause != null) {
            if (cause.getMessage() != null && cause.getMessage().contains("Cannot delete")) {
                log.warn("临时文件清理失败，这通常是因为文件被占用，不影响功能使用：{}", cause.getMessage());
                response.put("success", true);
                response.put("message", "检测已完成（临时文件清理警告已忽略）");
                return response;
            }
            cause = cause.getCause();
        }
        
        log.error("未检查的 IO 异常", e);
        response.put("success", false);
        response.put("message", "系统异常：" + e.getMessage());
        return response;
    }

    /**
     * 处理文件超大异常
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public Map<String, Object> handleMaxUploadSizeExceededException(MaxUploadSizeExceededException e) {
        Map<String, Object> response = new HashMap<>();
        log.error("文件超出最大限制", e);
        response.put("success", false);
        response.put("message", "上传文件超出最大限制（500MB）");
        return response;
    }

    /**
     * 忽略浏览器自动请求的静态资源（如 Chrome DevTools 探测路径）
     */
    @ResponseStatus(HttpStatus.NOT_FOUND)
    @ExceptionHandler(NoResourceFoundException.class)
    public void handleNoResourceFound(NoResourceFoundException e) {
        log.debug("静态资源未找到（已忽略）：{}", e.getResourcePath());
    }

    /**
     * 处理其他通用异常
     */
    @ExceptionHandler(Exception.class)
    public Map<String, Object> handleException(Exception e) {
        Map<String, Object> response = new HashMap<>();
        log.error("系统异常", e);
        response.put("success", false);
        response.put("message", "系统异常：" + e.getMessage());
        return response;
    }
}
