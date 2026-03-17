@echo off
echo ========================================
echo   AI 视频检测系统 - 启动脚本
echo ========================================
echo.

REM 检查 Java 是否安装
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Java，请先安装 JDK 11 或更高版本
    pause
    exit /b 1
)

echo [信息] Java 已安装
echo.

REM 创建上传目录
if not exist "E:\new_model\uploads" (
    echo [信息] 创建上传目录...
    mkdir "E:\new_model\uploads"
    echo [成功] 上传目录已创建
) else (
    echo [信息] 上传目录已存在
)
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到 Python，AI 检测功能可能无法使用
) else (
    echo [信息] Python 已安装
    python --version
)
echo.

echo ========================================
echo   正在启动 Spring Boot 应用...
echo ========================================
echo.
echo 访问地址：http://localhost:8080
echo.
echo 按 Ctrl+C 可停止服务
echo.

cd /d "%~dp0"
mvn spring-boot:run

pause
