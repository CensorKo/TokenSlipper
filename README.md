![TokenSlipper Banner](logo/banner.png)

# 🩴 TokenSlipper

> **给各大模型厂商 "敲警钟" 的 "拖鞋"！**

TokenSlipper 是一款智能 API 代理服务器，专为精简 Token 请求、剔除冗余上下文而设计，终结无效 Token 导致的费用飙升 —— 告别费用爆炸，让大模型 API 使用回归理性。

## 🎯 核心理念

- 📉 **精简 Token**: 智能压缩上下文，减少无效请求
- 🔍 **透明监控**: 完整记录每次请求，费用一目了然
- 🧠 **理性消费**: 数据驱动，让每一分钱都花在刀刃上

## ✨ 特性

- ✅ 完全兼容 OpenAI API 格式
- ✅ 支持流式 (SSE) 和非流式响应
- ✅ **完整的请求/响应记录** - 所有请求保存到 MySQL 数据库
- ✅ **Web 管理后台** - 可视化查看所有请求、上下文、响应
- ✅ **模型名称映射** - 自动转换 Cursor 和第三方 API 的模型名
- ✅ **Token 使用统计** - 实时监控 Token 消耗
- ✅ 支持 `/v1/chat/completions` 聊天补全
- ✅ 支持 `/v1/completions` 文本补全
- ✅ 支持 `/v1/embeddings` 向量嵌入
- ✅ 支持 Claude、GPT 等多种模型
- ✅ 轻量快速，基于 FastAPI

## 📦 安装

```bash
# 进入目录
cd TokenSlipper

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# ==================== 代理配置 ====================
UPSTREAM_BASE_URL=https://api.anthropic.com/v1
UPSTREAM_API_KEY=your-api-key-here
PROXY_PORT=8000
VERIFY_CLIENT_AUTH=false
LOG_LEVEL=info

# ==================== 数据库配置 ====================
DB_HOST=localhost
DB_PORT=3306
DB_USER=tokenslipper
DB_PASSWORD=your-db-password
DB_NAME=tokenslipper

# ==================== 管理后台配置 ====================
ADMIN_PORT=8080
```

### 数据库初始化

```sql
CREATE DATABASE tokenslipper CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## 🚀 使用

### 启动 TokenSlipper

```bash
python proxy.py
```

启动后看到：

```
╔══════════════════════════════════════════════════════════════════╗
║     🩴 TokenSlipper - 让大模型 API 使用回归理性                    ║
╠══════════════════════════════════════════════════════════════════╣
║  代理端口:        8000                                            ║
║  管理后台:        http://localhost:8080/admin                     ║
╚══════════════════════════════════════════════════════════════════╝
```

### 在 Cursor 中使用

- **OpenAI Base URL**: `http://localhost:8000/v1`
- **OpenAI API Key**: 任意字符串（如果 `VERIFY_CLIENT_AUTH=false`）

### 管理后台

访问：`http://localhost:8080/admin`

#### 📊 概览页
- 总请求数、消息数、Token 消耗、平均响应时间
- 最近 20 条请求
- 模型使用统计

#### 📋 请求列表
- 分页显示（可选 10/20/50/100/200 条/页）
- 支持翻页

#### 📄 请求详情
- 完整的请求/响应信息
- **对话上下文**（所有消息内容）
- **AI 回复内容**
- Token 使用情况
- 原始 JSON 数据

## 🗺️ 模型名称映射

| Cursor 发送 | 映射到 |
|-------------|--------|
| `gpt-4o` | `claude-3-5-sonnet-20241022` |
| `gpt-4o-mini` | `claude-3-5-haiku-20241022` |
| `gpt-4` | `claude-3-opus-20240229` |
| `gpt-4-turbo` | `claude-3-opus-20240229` |

自定义映射：
```env
EXTRA_MODEL_MAPPING='{"my-model":"actual-model"}'
```

## 🔌 API 端点

| 端点 | 说明 |
|------|------|
| `GET /` | 服务信息 |
| `GET /health` | 健康检查 |
| `GET /v1/models` | 获取模型列表 |
| `POST /v1/chat/completions` | 聊天补全 |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/embeddings` | 向量嵌入 |
| `GET /admin/` | 管理后台首页 |
| `GET /admin/requests` | 请求列表 |
| `GET /admin/request/{id}` | 请求详情 |

## 🛣️ 路线图

- [ ] Token 压缩算法 - 智能精简上下文
- [ ] 重复请求检测 - 避免重复计费
- [ ] 费用统计报表 - 按模型/时间段分析
- [ ] 告警机制 - Token 使用量超限提醒
- [ ] 缓存机制 - 相似请求复用响应

## 📄 许可证

MIT - 让 Token 费用回归理性！
