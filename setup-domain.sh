#!/bin/bash

# TokenSlipper 域名配置脚本
# 配置 www.tokenslipper.com 域名和 80 端口

DOMAIN="www.tokenslipper.com"
GUNICORN_PORT=8000

echo "🌐 TokenSlipper 域名配置"
echo "================================"
echo ""

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  请使用 sudo 或 root 用户运行此脚本"
    exit 1
fi

echo "📋 配置信息:"
echo "  域名: $DOMAIN"
echo "  外部端口: 80"
echo "  内部端口: $GUNICORN_PORT"
echo ""

# 方案1: 使用 iptables 端口转发
echo "🔧 方案1: 设置 iptables 端口转发 (80 → $GUNICORN_PORT)"
echo "--------------------------------"

# 清除旧规则
iptables -t nat -D PREROUTING -p tcp --dport 80 -j REDIRECT --to-port $GUNICORN_PORT 2>/dev/null || true

# 添加新规则
iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port $GUNICORN_PORT
iptables -t nat -A OUTPUT -p tcp -o lo --dport 80 -j REDIRECT --to-port $GUNICORN_PORT

echo "✅ iptables 规则已添加"
echo ""

# 保存 iptables 规则
if command -v iptables-save &> /dev/null; then
    mkdir -p /etc/iptables
    iptables-save > /etc/iptables/rules.v4
    echo "✅ iptables 规则已保存"
fi

echo ""
echo "📋 当前 iptables 规则:"
iptables -t nat -L PREROUTING -n --line-numbers | grep -E "80|PORT"

echo ""
echo "📝 修改 hosts 文件 (本地测试用)"
echo "--------------------------------"
echo "如果需要在本地测试，请在 /etc/hosts 中添加:"
echo "127.0.0.1 $DOMAIN"
echo ""

echo "🚀 重启服务..."
cd /root/TokenSlipper
./gunicorn.sh restart

echo ""
echo "================================"
echo "✅ 配置完成!"
echo ""
echo "🌐 访问地址:"
echo "  http://$DOMAIN"
echo "  http://$DOMAIN/admin"
echo ""
echo "⚠️  注意:"
echo "1. 请确保 DNS 已正确解析到本服务器 IP"
echo "2. 防火墙已放行 80 端口"
echo "3. 如需 HTTPS，请配置 SSL 证书"
echo ""
