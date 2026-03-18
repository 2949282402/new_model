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

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(IOException.class)
    public Map<String, Object> handleIOException(IOException e) {
        Map<String, Object> response = new HashMap<>();

        if (e.getMessage() != null && e.getMessage().contains("Cannot delete")) {
            log.warn("Temporary file cleanup failed but was ignored: {}", e.getMessage());
            response.put("success", true);
            response.put("message", "检测已完成，临时文件清理警告已忽略");
        } else {
            log.error("IO exception while processing request", e);
            response.put("success", false);
            response.put("message", "文件处理失败：" + e.getMessage());
        }

        return response;
    }

    @ExceptionHandler(UncheckedIOException.class)
    public Map<String, Object> handleUncheckedIOException(UncheckedIOException e) {
        Map<String, Object> response = new HashMap<>();

        Throwable cause = e.getCause();
        while (cause != null) {
            if (cause.getMessage() != null && cause.getMessage().contains("Cannot delete")) {
                log.warn("Temporary file cleanup failed but was ignored: {}", cause.getMessage());
                response.put("success", true);
                response.put("message", "检测已完成，临时文件清理警告已忽略");
                return response;
            }
            cause = cause.getCause();
        }

        log.error("Unchecked IO exception while processing request", e);
        response.put("success", false);
        response.put("message", "系统异常：" + e.getMessage());
        return response;
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public Map<String, Object> handleMaxUploadSizeExceededException(MaxUploadSizeExceededException e) {
        Map<String, Object> response = new HashMap<>();
        log.error("Uploaded file exceeded max size", e);
        response.put("success", false);
        response.put("message", "上传文件超过最大限制（500MB）");
        return response;
    }

    @ResponseStatus(HttpStatus.NOT_FOUND)
    @ExceptionHandler(NoResourceFoundException.class)
    public void handleNoResourceFound(NoResourceFoundException e) {
        log.debug("Static resource not found and ignored: {}", e.getResourcePath());
    }

    @ExceptionHandler(Exception.class)
    public Map<String, Object> handleException(Exception e) {
        Map<String, Object> response = new HashMap<>();
        log.error("Unhandled system exception", e);
        response.put("success", false);
        response.put("message", "系统异常：" + e.getMessage());
        return response;
    }
}