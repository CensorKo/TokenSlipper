#!/bin/bash

# 停止 MySQL Docker 容器

CONTAINER_NAME="tokenslipper-mysql"

if docker ps | grep -q "$CONTAINER_NAME"; then
    echo "🛑 停止 MySQL 容器..."
    docker stop "$CONTAINER_NAME"
    echo "✅ MySQL 已停止"
else
    echo "⚠️ MySQL 容器未运行"
fi
