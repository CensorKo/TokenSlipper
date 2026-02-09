# Gunicorn 配置文件
import os
import multiprocessing

# 读取环境变量
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# 服务绑定
bind = "0.0.0.0:8000"

# 工作进程数
def get_workers():
    if DEBUG:
        return 1  # Debug 模式只用一个 worker，方便调试
    return multiprocessing.cpu_count() * 2 + 1

workers = get_workers()

# 工作模式 - 使用 uvicorn.workers.UvicornWorker 来支持 ASGI
worker_class = "uvicorn.workers.UvicornWorker"

# 守护进程模式
daemon = False

# 日志级别
loglevel = "debug" if DEBUG else LOG_LEVEL

# 访问日志
accesslog = "/var/log/tokenslipper/access.log"

# 错误日志
errorlog = "/var/log/tokenslipper/error.log"

# 进程 PID 文件
pidfile = "/var/run/tokenslipper.pid"

# 超时设置
timeout = 120
keepalive = 5

# 预加载应用 - Debug 模式关闭预加载，方便代码热更新
preload_app = not DEBUG

# Debug 模式设置
reload = DEBUG  # 代码变更时自动重载
reload_engine = "auto"

# 工作进程名称
proc_name = "tokenslipper"

# 捕获输出
capture_output = True
enable_stdio_inheritance = DEBUG

# 访问日志格式
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

print(f"🚀 Gunicorn 配置加载完成")
print(f"   Debug 模式: {DEBUG}")
print(f"   工作进程: {workers}")
print(f"   自动重载: {reload}")
print(f"   日志级别: {loglevel}")
