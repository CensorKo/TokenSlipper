#!/bin/bash

# TokenSlipper Gunicorn 管理脚本

APP_DIR="/root/TokenSlipper"
APP_MODULE="proxy:app"
PID_FILE="/var/run/tokenslipper.pid"
LOG_DIR="/var/log/tokenslipper"
GUNICORN="/usr/local/bin/gunicorn"

# 确保日志目录存在
mkdir -p $LOG_DIR

cd $APP_DIR || exit 1

start() {
    if [ -f $PID_FILE ] && kill -0 $(cat $PID_FILE) 2>/dev/null; then
        echo "⚠️  TokenSlipper 已在运行 (PID: $(cat $PID_FILE))"
        return 1
    fi
    
    echo "🚀 启动 TokenSlipper..."
    
    # 设置环境变量
    export PYTHONPATH=$APP_DIR:$PYTHONPATH
    
    # 使用 gunicorn 启动
    $GUNICORN \
        -c $APP_DIR/gunicorn.conf.py \
        --daemon \
        $APP_MODULE
    
    sleep 2
    
    if [ -f $PID_FILE ] && kill -0 $(cat $PID_FILE) 2>/dev/null; then
        echo "✅ TokenSlipper 已启动 (PID: $(cat $PID_FILE))"
        echo "📋 访问地址: http://0.0.0.0:8000"
        echo "📊 管理后台: http://0.0.0.0:8000/admin"
    else
        echo "❌ 启动失败，查看错误日志: $LOG_DIR/error.log"
        return 1
    fi
}

start_fg() {
    echo "🚀 启动 TokenSlipper (前台模式)..."
    echo "按 Ctrl+C 停止服务"
    echo ""
    
    export PYTHONPATH=$APP_DIR:$PYTHONPATH
    
    $GUNICORN \
        -c $APP_DIR/gunicorn.conf.py \
        $APP_MODULE
}

stop() {
    if [ ! -f $PID_FILE ]; then
        echo "⚠️  PID 文件不存在，尝试查找进程..."
        PID=$(pgrep -f "gunicorn.*proxy:app" | head -1)
        if [ -z "$PID" ]; then
            echo "❌ 未找到运行中的进程"
            return 1
        fi
    else
        PID=$(cat $PID_FILE)
    fi
    
    if kill -0 $PID 2>/dev/null; then
        echo "🛑 停止 TokenSlipper (PID: $PID)..."
        kill -TERM $PID
        
        # 等待进程结束
        for i in {1..10}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo "✅ 已停止"
                rm -f $PID_FILE
                return 0
            fi
            sleep 1
        done
        
        # 强制结束
        echo "⚠️  强制结束进程..."
        kill -KILL $PID 2>/dev/null
        rm -f $PID_FILE
    else
        echo "⚠️  进程未运行"
        rm -f $PID_FILE
    fi
}

restart() {
    stop
    sleep 2
    start
}

status() {
    if [ -f $PID_FILE ]; then
        PID=$(cat $PID_FILE)
        if kill -0 $PID 2>/dev/null; then
            echo "✅ TokenSlipper 运行中 (PID: $PID)"
            echo "📊 工作进程:"
            ps aux | grep gunicorn | grep -v grep
            echo ""
            echo "🌐 监听端口:"
            ss -tlnp | grep -E "8000|gunicorn" || netstat -tlnp 2>/dev/null | grep -E "8000|gunicorn"
        else
            echo "❌ PID 文件存在但进程未运行"
        fi
    else
        PID=$(pgrep -f "gunicorn.*proxy:app" | head -1)
        if [ -n "$PID" ]; then
            echo "⚠️  进程运行中但 PID 文件丢失 (PID: $PID)"
        else
            echo "❌ TokenSlipper 未运行"
        fi
    fi
}

reload() {
    if [ -f $PID_FILE ]; then
        PID=$(cat $PID_FILE)
        echo "🔄 重新加载配置 (PID: $PID)..."
        kill -HUP $PID
        echo "✅ 已发送重载信号"
    else
        echo "❌ 未找到 PID 文件"
        return 1
    fi
}

debug() {
    echo "🐛 Debug 模式启动 (Ctrl+C 停止)..."
    export LOG_LEVEL=debug
    
    # 先尝试普通启动，如有问题可以看到详细错误
    python3 proxy.py
}

log() {
    echo "📋 错误日志 (最近 50 行):"
    tail -50 $LOG_DIR/error.log 2>/dev/null || echo "暂无错误日志"
    echo ""
    echo "📋 访问日志 (最近 20 行):"
    tail -20 $LOG_DIR/access.log 2>/dev/null || echo "暂无访问日志"
}

case "${1:-}" in
    start)
        start
        ;;
    start-fg|fg)
        start_fg
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    reload)
        reload
        ;;
    debug)
        debug
        ;;
    log|logs)
        log
        ;;
    *)
        echo "🩴 TokenSlipper Gunicorn 管理脚本"
        echo ""
        echo "用法: $0 [命令]"
        echo ""
        echo "命令:"
        echo "  start       启动服务（后台守护模式）"
        echo "  start-fg    启动服务（前台模式）"
        echo "  stop        停止服务"
        echo "  restart     重启服务"
        echo "  reload      重新加载配置"
        echo "  status      查看状态"
        echo "  debug       Debug 模式启动（Python 直接运行）"
        echo "  log         查看日志"
        echo ""
        exit 1
        ;;
esac
