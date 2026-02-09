#!/bin/bash

# MySQL Docker 启动脚本

CONTAINER_NAME="tokenslipper-mysql"
DATA_DIR="$(dirname "$0")/mysql_data"

# 创建数据目录
mkdir -p "$DATA_DIR"

# 检查容器是否已存在
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "📦 MySQL 容器已存在，正在启动..."
    docker start "$CONTAINER_NAME"
else
    echo "🚀 创建并启动 MySQL 容器..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart always \
        -e MYSQL_ROOT_PASSWORD=rootpassword \
        -e MYSQL_DATABASE=tokenslipper \
        -e MYSQL_USER=tokenslipper \
        -e MYSQL_PASSWORD=tokenslipper123 \
        -p 3306:3306 \
        -v "$DATA_DIR":/var/lib/mysql \
        mysql:8.0 \
        --default-authentication-plugin=mysql_native_password \
        --character-set-server=utf8mb4 \
        --collation-server=utf8mb4_unicode_ci
fi

echo ""
echo "⏳ 等待 MySQL 启动..."
sleep 5

# 检查是否启动成功
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo "✅ MySQL 已启动！"
    echo ""
    echo "📋 连接信息："
    echo "   主机: localhost"
    echo "   端口: 3306"
    echo "   数据库: tokenslipper"
    echo "   用户名: tokenslipper"
    echo "   密码: tokenslipper123"
    echo "   Root 密码: rootpassword"
    echo ""
    echo "🔧 常用命令："
    echo "   查看日志: docker logs $CONTAINER_NAME"
    echo "   停止: docker stop $CONTAINER_NAME"
    echo "   重启: docker restart $CONTAINER_NAME"
    echo "   进入容器: docker exec -it $CONTAINER_NAME mysql -u tokenslipper -p"
else
    echo "❌ MySQL 启动失败，查看日志："
    docker logs "$CONTAINER_NAME"
fi
