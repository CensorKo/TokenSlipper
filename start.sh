#!/bin/bash

# TokenSlipper 启动脚本

cd "$(dirname "$0")"

# 检查虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 安装依赖（如果还没安装）
pip install -q -r requirements.txt 2>/dev/null

# 检查 MySQL 是否运行
if ! docker ps | grep -q tokenslipper-mysql; then
    echo "⚠️ 警告: MySQL 容器未运行，请先启动: docker-compose up -d mysql"
    echo ""
fi

# 启动代理
echo "🚀 启动 TokenSlipper..."
python proxy.py
