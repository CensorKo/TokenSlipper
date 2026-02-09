#!/bin/bash

# TokenSlipper 防火墙管理脚本
# 功能：管理防火墙规则，保护 MySQL 不外网暴露

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

show_help() {
    echo "🛡️  TokenSlipper 防火墙管理"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  apply     应用防火墙规则（禁止外网访问3306）"
    echo "  allow-db  允许特定IP访问MySQL（如内网IP）"
    echo "  block-db  阻止特定IP访问MySQL"
    echo "  status    查看当前防火墙状态"
    echo "  reset     重置防火墙规则（清空所有规则）"
    echo "  save      保存当前规则"
    echo "  help      显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 apply              # 应用基础规则"
    echo "  $0 allow-db 10.0.0.5  # 允许10.0.0.5访问MySQL"
}

apply_rules() {
    echo "🛡️  应用防火墙规则..."
    
    # 清空现有规则
    iptables -F
    iptables -X
    
    # 默认策略
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    # 本地回环
    iptables -A INPUT -i lo -j ACCEPT
    
    # 已建立的连接
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # SSH (22)
    iptables -A INPUT -p tcp --dport 22 -j ACCEPT
    
    # TokenSlipper (8000)
    iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
    
    # MySQL - 仅本地
    iptables -A INPUT -p tcp -s 127.0.0.1 --dport 3306 -j ACCEPT
    
    # 拒绝其他MySQL连接
    iptables -A INPUT -p tcp --dport 3306 -j DROP
    
    # ICMP
    iptables -A INPUT -p icmp -j ACCEPT
    
    echo "✅ 规则已应用"
    iptables -L INPUT -n --line-numbers | grep -E "3306|8000|22"
}

allow_db() {
    local ip=$1
    if [ -z "$ip" ]; then
        echo "❌ 请指定IP地址"
        echo "用法: $0 allow-db <IP地址>"
        exit 1
    fi
    
    # 在DROP规则之前插入ALLOW规则
    iptables -I INPUT -p tcp -s $ip --dport 3306 -j ACCEPT
    echo "✅ 已允许 $ip 访问 MySQL"
}

block_db() {
    local ip=$1
    if [ -z "$ip" ]; then
        echo "❌ 请指定IP地址"
        exit 1
    fi
    
    iptables -A INPUT -p tcp -s $ip --dport 3306 -j DROP
    echo "✅ 已阻止 $ip 访问 MySQL"
}

show_status() {
    echo "📋 当前防火墙规则 (INPUT链):"
    echo "================================"
    iptables -L INPUT -n --line-numbers
    echo ""
    echo "🔍 端口监听状态:"
    ss -tlnp | grep -E "3306|8000" || netstat -tlnp 2>/dev/null | grep -E "3306|8000"
}

reset_rules() {
    echo "⚠️  确定要清空所有防火墙规则吗？"
    echo "这将允许所有连接！"
    read -p "输入 'yes' 确认: " confirm
    
    if [ "$confirm" = "yes" ]; then
        iptables -F
        iptables -X
        iptables -P INPUT ACCEPT
        iptables -P FORWARD ACCEPT
        iptables -P OUTPUT ACCEPT
        echo "✅ 防火墙已重置（允许所有连接）"
    else
        echo "❌ 操作已取消"
    fi
}

save_rules() {
    echo "💾 保存防火墙规则..."
    mkdir -p /etc/iptables 2>/dev/null
    if iptables-save > /etc/iptables/rules.v4 2>/dev/null; then
        echo "✅ 规则已保存到 /etc/iptables/rules.v4"
    else
        iptables-save > /root/iptables-rules.v4
        echo "✅ 规则已保存到 /root/iptables-rules.v4"
    fi
}

# 主逻辑
case "${1:-}" in
    apply)
        apply_rules
        ;;
    allow-db)
        allow_db $2
        ;;
    block-db)
        block_db $2
        ;;
    status)
        show_status
        ;;
    reset)
        reset_rules
        ;;
    save)
        save_rules
        ;;
    help|--help|-h|*)
        show_help
        ;;
esac
