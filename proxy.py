#!/usr/bin/env python3
"""
🩴 TokenSlipper - 智能 API 代理服务器

给各大模型厂商 "敲警钟" 的 "拖鞋"！
精简 Token 请求、剔除冗余上下文，终结无效 Token 导致的费用飙升

特性:
- 兼容 OpenAI API 格式
- 完整的请求/响应日志记录
- 模型名称映射转换
- MySQL 数据库存储
- Web 管理后台
"""

import os
import json
import time
import uuid
import httpx
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc

from database import (
    init_db, get_db, SessionLocal,
    RequestLog, Message, ResponseLog
)

load_dotenv()

# ==================== 配置 ====================
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
VERIFY_CLIENT_AUTH = os.getenv("VERIFY_CLIENT_AUTH", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
ADMIN_PORT = int(os.getenv("ADMIN_PORT", "8080"))  # 管理后台端口

# ==================== 模型名称映射 ====================
MODEL_MAPPING = {
    "gpt-4o": "claude-3-5-sonnet-20241022",
    "gpt-4o-mini": "claude-3-5-haiku-20241022",
    "gpt-4": "claude-3-opus-20240229",
    "gpt-4-turbo": "claude-3-opus-20240229",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}

try:
    extra_mapping = json.loads(os.getenv("EXTRA_MODEL_MAPPING", "{}"))
    MODEL_MAPPING.update(extra_mapping)
except json.JSONDecodeError:
    pass

# ==================== 日志工具 ====================
def log_debug(msg: str, data: dict = None):
    if LOG_LEVEL in ["debug"]:
        _print_log("DEBUG", msg, data)

def log_info(msg: str, data: dict = None):
    if LOG_LEVEL in ["debug", "info"]:
        _print_log("INFO", msg, data)

def log_error(msg: str, data: dict = None):
    _print_log("ERROR", msg, data)

def _print_log(level: str, msg: str, data: dict = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {msg}")
    if data:
        json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        indented = "\n".join("    " + line for line in json_str.split("\n"))
        print(f"    Data:\n{indented}")

# ==================== 数据模型 ====================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None

# ==================== FastAPI 应用 ====================
app = FastAPI(title="TokenSlipper", version="2.0.0", description="让大模型 API 使用回归理性")
templates = Jinja2Templates(directory="templates")

request_counter = 0

def get_request_id() -> int:
    global request_counter
    request_counter += 1
    return request_counter

def map_model_name(cursor_model: str) -> str:
    """将 Cursor 的模型名映射到第三方实际的模型名"""
    original_model = cursor_model
    mapped_model = MODEL_MAPPING.get(cursor_model, cursor_model)
    if original_model != mapped_model:
        log_info(f"模型名映射: {original_model} -> {mapped_model}")
    return mapped_model

async def get_upstream_headers(client_auth: Optional[str] = None) -> dict:
    """构建转发到上游 API 的请求头"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    
    if VERIFY_CLIENT_AUTH and client_auth:
        headers["Authorization"] = client_auth
        log_debug("使用客户端提供的 Authorization")
    elif UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
        log_debug("使用配置的 UPSTREAM_API_KEY")
    
    return headers

def save_request_to_db(db: Session, request_id: str, headers: dict, body: dict, client_ip: str):
    """保存请求信息到数据库"""
    try:
        messages = body.get("messages", [])
        
        # 创建请求记录
        request_log = RequestLog(
            request_id=request_id,
            method="POST",
            path="/v1/chat/completions",
            client_ip=client_ip,
            user_agent=headers.get("user-agent", "")[:500],
            model_requested=body.get("model", ""),
            model_mapped=map_model_name(body.get("model", "")),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            stream=1 if body.get("stream", False) else 0,
            request_body=body,
            message_count=len(messages)
        )
        db.add(request_log)
        
        # 保存每条消息
        for idx, msg in enumerate(messages):
            message = Message(
                request_id=request_id,
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
                content_preview=msg.get("content", "")[:200],
                message_index=idx
            )
            db.add(message)
        
        db.commit()
        log_debug(f"请求已保存到数据库: {request_id}")
        return request_log
    except Exception as e:
        db.rollback()
        log_error(f"保存请求到数据库失败", {"error": str(e)})
        return None

def save_response_to_db(db: Session, request_id: str, status_code: int, 
                        response_data: dict, upstream_latency: float, 
                        total_latency: float, is_stream: bool = False,
                        chunk_count: int = None, error_msg: str = None):
    """保存响应信息到数据库"""
    try:
        # 提取响应内容
        content = ""
        if response_data and "choices" in response_data:
            choices = response_data.get("choices", [])
            if choices:
                # 尝试获取消息内容
                if "message" in choices[0]:
                    content = choices[0]["message"].get("content", "")
                elif "text" in choices[0]:
                    content = choices[0].get("text", "")
        
        # 获取 token 使用情况
        usage = response_data.get("usage", {}) if response_data else {}
        
        response_log = ResponseLog(
            request_id=request_id,
            status_code=status_code,
            upstream_latency=upstream_latency,
            total_latency=total_latency,
            model_responded=response_data.get("model") if response_data else None,
            finish_reason=response_data.get("choices", [{}])[0].get("finish_reason") if response_data else None,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            response_content=content,
            response_content_preview=content[:200] if content else None,
            is_stream=1 if is_stream else 0,
            chunk_count=chunk_count,
            error_message=error_msg,
            response_body=response_data
        )
        db.add(response_log)
        db.commit()
        log_debug(f"响应已保存到数据库: {request_id}")
    except Exception as e:
        db.rollback()
        log_error(f"保存响应到数据库失败", {"error": str(e)})

async def stream_response_with_capture(response: httpx.Response, req_id: str, db_request_id: str) -> AsyncGenerator[str, None]:
    """流式读取上游响应并 yield SSE 格式数据，同时捕获完整内容"""
    chunk_count = 0
    full_content_parts = []
    start_time = time.time()
    
    try:
        async for line in response.aiter_lines():
            if line:
                chunk_count += 1
                if LOG_LEVEL == "debug" and chunk_count <= 3:
                    log_debug(f"[Req {req_id}] SSE chunk #{chunk_count}", {"data": line[:200]})
                
                # 尝试解析内容
                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                full_content_parts.append(delta["content"])
                    except:
                        pass
                
                yield f"{line}\n\n"
        
        # 流结束，保存响应
        total_latency = time.time() - start_time
        full_content = "".join(full_content_parts)
        
        # 构造一个模拟的响应数据
        response_data = {
            "choices": [{"message": {"content": full_content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        }
        
        # 异步保存到数据库（不阻塞响应）
        db = SessionLocal()
        try:
            save_response_to_db(db, db_request_id, 200, response_data, 
                              total_latency, total_latency, True, chunk_count)
        finally:
            db.close()
        
        log_info(f"[Req {req_id}] 流式响应结束，共 {chunk_count} 个 chunks")
        
    except Exception as e:
        log_error(f"[Req {req_id}] 流式响应错误", {"error": str(e)})
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

# ==================== API 路由 ====================

@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    """获取可用模型列表"""
    req_id = get_request_id()
    log_info(f"[Req {req_id}] GET /v1/models")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            upstream_headers = await get_upstream_headers(authorization)
            response = await client.get(f"{UPSTREAM_BASE_URL}/models", headers=upstream_headers)
            
            if response.status_code == 200:
                return JSONResponse(content=response.json())
    except Exception as e:
        log_error(f"[Req {req_id}] 获取模型列表失败", {"error": str(e)})
    
    # 返回默认模型列表
    default_models = {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "created": 1677610602, "owned_by": "proxy"},
            {"id": "gpt-4o-mini", "object": "model", "created": 1677610602, "owned_by": "proxy"},
            {"id": "gpt-4", "object": "model", "created": 1677610602, "owned_by": "proxy"},
            {"id": "claude-3-5-sonnet", "object": "model", "created": 1677610602, "owned_by": "proxy"},
            {"id": "claude-3-opus", "object": "model", "created": 1677610602, "owned_by": "proxy"},
        ]
    }
    return JSONResponse(content=default_models)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    """聊天补全接口 - 转发到上游 API"""
    req_id = get_request_id()
    db_request_id = str(uuid.uuid4())[:16]
    start_time = time.time()
    
    # 读取原始请求体
    raw_body = await request.body()
    try:
        body_json = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    headers = dict(request.headers)
    client_ip = request.client.host if request.client else None
    
    # 打印日志
    log_info(f"[Req {req_id}] 收到请求: POST /v1/chat/completions", {
        "request_id": db_request_id,
        "模型": body_json.get("model"),
        "消息数": len(body_json.get("messages", [])),
        "stream": body_json.get("stream", False)
    })
    
    # 保存请求到数据库
    db = SessionLocal()
    try:
        save_request_to_db(db, db_request_id, headers, body_json, client_ip)
    finally:
        db.close()
    
    # 模型名称映射
    original_model = body_json.get("model", "")
    body_json["model"] = map_model_name(original_model)
    
    try:
        upstream_headers = await get_upstream_headers(authorization)
        is_stream = body_json.get("stream", False)
        
        log_info(f"[Req {req_id}] 转发到上游: {UPSTREAM_BASE_URL}/chat/completions")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            upstream_start = time.time()
            response = await client.post(
                f"{UPSTREAM_BASE_URL}/chat/completions",
                headers=upstream_headers,
                json=body_json,
                timeout=300.0
            )
            upstream_latency = time.time() - upstream_start
            
            log_info(f"[Req {req_id}] 上游响应: {response.status_code}, 首包耗时: {upstream_latency:.3f}s")
            
            if response.status_code != 200:
                error_text = await response.aread()
                error_str = error_text.decode()
                log_error(f"[Req {req_id}] 上游返回错误", {"status": response.status_code, "body": error_str[:500]})
                
                # 保存错误响应
                db = SessionLocal()
                try:
                    total_latency = time.time() - start_time
                    save_response_to_db(db, db_request_id, response.status_code, None,
                                      upstream_latency, total_latency, False, None, error_str)
                finally:
                    db.close()
                
                raise HTTPException(status_code=response.status_code, detail=error_str)
            
            if is_stream:
                # 流式请求
                return StreamingResponse(
                    stream_response_with_capture(response, req_id, db_request_id),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                # 非流式请求
                response_data = response.json()
                total_latency = time.time() - start_time
                
                # 打印响应摘要
                usage = response_data.get("usage", {})
                log_info(f"[Req {req_id}] 响应摘要", {
                    "模型": response_data.get("model"),
                    "finish_reason": response_data.get("choices", [{}])[0].get("finish_reason"),
                    "total_tokens": usage.get("total_tokens"),
                    "总耗时": f"{total_latency:.3f}s"
                })
                
                # 保存响应到数据库
                db = SessionLocal()
                try:
                    save_response_to_db(db, db_request_id, 200, response_data,
                                      upstream_latency, total_latency, False)
                finally:
                    db.close()
                
                return JSONResponse(content=response_data)
                
    except httpx.TimeoutException:
        log_error(f"[Req {req_id}] 上游超时")
        raise HTTPException(status_code=504, detail="Upstream timeout")
    except httpx.ConnectError as e:
        log_error(f"[Req {req_id}] 无法连接到上游", {"error": str(e)})
        raise HTTPException(status_code=502, detail="Cannot connect to upstream")
    except Exception as e:
        log_error(f"[Req {req_id}] 处理异常", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    """文本补全接口（旧版）- 转发到上游 API"""
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if "model" in body:
        body["model"] = map_model_name(body["model"])
    
    try:
        upstream_headers = await get_upstream_headers(authorization)
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{UPSTREAM_BASE_URL}/completions",
                headers=upstream_headers,
                json=body,
                timeout=300.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=(await response.aread()).decode())
            
            if body.get("stream", False):
                return StreamingResponse(response.aiter_text(), media_type="text/event-stream")
            else:
                return JSONResponse(content=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: Optional[str] = Header(None)):
    """向量嵌入接口 - 转发到上游 API"""
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if "model" in body:
        body["model"] = map_model_name(body["model"])
    
    try:
        upstream_headers = await get_upstream_headers(authorization)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{UPSTREAM_BASE_URL}/embeddings",
                headers=upstream_headers,
                json=body,
                timeout=60.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=(await response.aread()).decode())
            return JSONResponse(content=response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "proxy": "openai-compatible"}

@app.get("/")
async def root():
    return {
        "name": "TokenSlipper",
        "slogan": "让大模型 API 使用回归理性",
        "version": "2.0.0",
        "admin_panel": f"http://localhost:{ADMIN_PORT}/admin",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
    }

# ==================== 管理后台路由 ====================

@app.get("/admin")
async def admin_redirect():
    return RedirectResponse(url="/admin/")

@app.get("/admin/")
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    """管理后台首页 - 概览"""
    # 统计数据
    total_requests = db.query(RequestLog).count()
    total_messages = db.query(Message).count()
    
    # Token 统计
    token_stats = db.query(
        func.sum(ResponseLog.total_tokens).label("total_tokens"),
        func.avg(ResponseLog.total_tokens).label("avg_tokens")
    ).first()
    
    # 平均响应时间
    latency_stats = db.query(func.avg(ResponseLog.total_latency).label("avg_latency")).first()
    
    # 最近请求（最近20条）
    recent_requests = db.query(RequestLog).options(
        joinedload(RequestLog.response)
    ).order_by(desc(RequestLog.timestamp)).limit(20).all()
    
    # 模型使用统计
    model_stats = db.query(
        RequestLog.model_mapped,
        func.count(RequestLog.id).label("count")
    ).group_by(RequestLog.model_mapped).order_by(desc("count")).all()
    
    stats = {
        "total_requests": total_requests,
        "total_messages": total_messages,
        "total_tokens": int(token_stats.total_tokens) if token_stats.total_tokens else 0,
        "avg_latency": round(latency_stats.avg_latency, 2) if latency_stats.avg_latency else 0
    }
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "recent_requests": recent_requests,
        "model_stats": model_stats
    })

@app.get("/admin/requests")
async def admin_requests(
    request: Request, 
    page: int = 1, 
    per_page: int = 50,
    db: Session = Depends(get_db)
):
    """请求列表页面"""
    # 限制 per_page 的最大值，防止查询过慢
    per_page = min(max(per_page, 10), 200)
    
    offset = (page - 1) * per_page
    
    # 查询总数
    total = db.query(RequestLog).count()
    total_pages = (total + per_page - 1) // per_page
    
    # 确保页码有效
    if page < 1:
        page = 1
    if total_pages > 0 and page > total_pages:
        page = total_pages
        offset = (page - 1) * per_page
    
    # 查询分页数据
    requests = db.query(RequestLog).options(
        joinedload(RequestLog.response)
    ).order_by(desc(RequestLog.timestamp)).offset(offset).limit(per_page).all()
    
    return templates.TemplateResponse("requests.html", {
        "request": request,
        "requests": requests,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages
    })

@app.get("/admin/request/{request_id}")
async def admin_request_detail(request_id: str, request: Request, db: Session = Depends(get_db)):
    """请求详情页面"""
    request_log = db.query(RequestLog).options(
        joinedload(RequestLog.messages),
        joinedload(RequestLog.response)
    ).filter(RequestLog.request_id == request_id).first()
    
    if not request_log:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # 排序消息
    messages = sorted(request_log.messages, key=lambda m: m.message_index)
    
    # 准备 JSON 数据
    request_body_json = json.dumps(request_log.request_body, ensure_ascii=False, indent=2) if request_log.request_body else "{}"
    response_body_json = json.dumps(request_log.response.response_body, ensure_ascii=False, indent=2) if request_log.response and request_log.response.response_body else "{}"
    
    return templates.TemplateResponse("request_detail.html", {
        "request": request,
        "request_log": request_log,
        "messages": messages,
        "request_body_json": request_body_json,
        "response_body_json": response_body_json
    })

# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 初始化数据库
    print("正在初始化数据库...")
    init_db()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     🩴 TokenSlipper - 让大模型 API 使用回归理性                    ║
╠══════════════════════════════════════════════════════════════════╣
║  代理端口:        {PROXY_PORT:<49} ║
║  管理后台:        http://localhost:{ADMIN_PORT}/admin{' ' * (27 - len(str(ADMIN_PORT)))}║
╠══════════════════════════════════════════════════════════════════╣
║  上游 API:        {UPSTREAM_BASE_URL:<49} ║
║  日志级别:        {LOG_LEVEL:<49} ║
╠══════════════════════════════════════════════════════════════════╣
║  🗺️  模型映射配置:                                                ║
""")
    
    for cursor_model, actual_model in MODEL_MAPPING.items():
        print(f"║    {cursor_model:<25} -> {actual_model:<35} ║")
    
    print(f"""╚══════════════════════════════════════════════════════════════════╝

使用方式:
  export OPENAI_BASE_URL=http://localhost:{PROXY_PORT}/v1
  export OPENAI_API_KEY=你的密钥

管理后台:
  http://localhost:{ADMIN_PORT}/admin
""")
    
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
