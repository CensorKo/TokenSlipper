#!/bin/bash

# OpenAI API 代理启动脚本

cd "$(dirname "$0")"

# 检查虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 安装依赖（如果还没安装）
pip install -q -r requirements.txt 2>/dev/null

# 启动代理
echo "正在启动 OpenAI API 代理..."
python proxy.py
