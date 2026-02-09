#!/bin/bash

# MySQL Docker 启动脚本（仅本地访问）

CONTAINER_NAME="tokenslipper-mysql"
DATA_DIR="$(dirname "$0")/mysql_data"

# 创建数据目录
mkdir -p "$DATA_DIR"

# 检查容器是否已存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "📦 MySQL 容器已存在"
    
    # 检查容器运行状态
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "✅ MySQL 已在运行中"
        exit 0
    else
        echo "🚀 启动 MySQL 容器..."
        docker start "$CONTAINER_NAME"
    fi
else
    echo "🚀 创建并启动 MySQL 容器（仅本地访问）..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart always \
        -e MYSQL_ROOT_PASSWORD=rootpassword \
        -e MYSQL_DATABASE=tokenslipper \
        -e MYSQL_USER=tokenslipper \
        -e MYSQL_PASSWORD=tokenslipper123 \
        -p 127.0.0.1:3306:3306 \
        -v "$DATA_DIR":/var/lib/mysql \
        mysql:8.0 \
        --default-authentication-plugin=mysql_native_password \
        --character-set-server=utf8mb4 \
        --collation-server=utf8mb4_unicode_ci
fi

# 等待 MySQL 启动
echo "⏳ 等待 MySQL 启动..."
for i in {1..30}; do
    if docker exec "$CONTAINER_NAME" mysqladmin ping -h localhost --silent 2>/dev/null; then
        echo "✅ MySQL 已启动！"
        echo ""
        echo "📋 连接信息："
        echo "   主机: localhost (仅本地访问)"
        echo "   端口: 3306"
        echo "   数据库: tokenslipper"
        echo "   用户名: tokenslipper"
        echo "   密码: tokenslipper123"
        echo ""
        echo "🔒 安全状态: 仅允许本地连接"
        exit 0
    fi
    sleep 1
done

echo "❌ MySQL 启动超时，查看日志："
docker logs "$CONTAINER_NAME" --tail 20
