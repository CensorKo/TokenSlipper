#!/bin/bash

# 设置防火墙规则
# - 允许本地访问 MySQL (3306)
# - 禁止外网访问 MySQL (3306)
# - 允许外网访问 TokenSlipper (8000)
# - 允许 SSH (22)

echo "🛡️  配置防火墙规则..."

# 清空现有规则（谨慎操作，确保有替代访问方式）
iptables -F
iptables -X

# 默认策略：允许出站，拒绝入站
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 允许本地回环接口的所有流量
iptables -A INPUT -i lo -j ACCEPT

# 允许已建立和相关的连接
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 允许 SSH (22端口) - 如果需要的话，可以先注释掉
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# 允许 TokenSlipper 代理服务 (8000端口)
iptables -A INPUT -p tcp --dport 8000 -j ACCEPT

# 允许本地访问 MySQL (3306) - 仅允许来自本地的连接
iptables -A INPUT -p tcp -s 127.0.0.1 --dport 3306 -j ACCEPT
iptables -A INPUT -p tcp -s ::1 --dport 3306 -j ACCEPT

# 拒绝其他所有到3306的连接（记录日志可选）
# iptables -A INPUT -p tcp --dport 3306 -j LOG --log-prefix "MySQL denied: "
iptables -A INPUT -p tcp --dport 3306 -j DROP

# ICMP (ping)
iptables -A INPUT -p icmp -j ACCEPT

echo "✅ 防火墙规则已应用"
echo ""
echo "📋 当前规则:"
iptables -L INPUT -n --line-numbers | head -20
echo ""
echo "🔒 MySQL (3306) 只允许本地访问"
echo "🌐 TokenSlipper (8000) 允许外网访问"
