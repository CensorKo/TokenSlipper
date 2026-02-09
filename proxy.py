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
import hashlib
import secrets
from typing import AsyncGenerator, Optional, List, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc

from database import (
    init_db, get_db, SessionLocal,
    RequestLog, Message, ResponseLog, User, ModelMapping, ApiProvider, ApiToken
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

# ==================== 管理后台认证配置 ====================
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # 默认密码，请及时修改
SESSION_COOKIE_NAME = "tokenslipper_session"
SESSION_MAX_AGE = 86400 * 7  # 7天

def hash_password(password: str) -> str:
    """哈希密码 (使用 SHA256 + salt)"""
    salt = secrets.token_hex(16)
    pwdhash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwdhash}"

def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    try:
        salt, stored_hash = hashed.split("$")
        pwdhash = hashlib.sha256((password + salt).encode()).hexdigest()
        return pwdhash == stored_hash
    except Exception:
        return False

def create_default_user():
    """创建默认管理员用户"""
    db = SessionLocal()
    try:
        # 检查是否已存在用户
        user = db.query(User).filter(User.username == ADMIN_USERNAME).first()
        if not user:
            user = User(
                username=ADMIN_USERNAME,
                password_hash=hash_password(ADMIN_PASSWORD),
                is_active=True
            )
            db.add(user)
            db.commit()
            print(f"✅ 创建默认管理员用户: {ADMIN_USERNAME}")
            print(f"⚠️  请使用默认密码登录后及时修改密码！")
    except Exception as e:
        print(f"❌ 创建默认用户失败: {e}")
    finally:
        db.close()

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """获取当前登录用户，未登录则重定向到登录页"""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
    
    # 简单的 session 验证：username|timestamp
    try:
        username, timestamp = session_token.split("|", 1)
        user = db.query(User).filter(User.username == username, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
        
        # 检查 session 是否过期
        if time.time() - float(timestamp) > SESSION_MAX_AGE:
            raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
        
        return user
    except Exception:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})

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
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

# ==================== FastAPI 应用 ====================
app = FastAPI(title="TokenSlipper", version="2.0.0", description="让大模型 API 使用回归理性")
templates = Jinja2Templates(directory="templates")

request_counter = 0

def get_request_id() -> int:
    global request_counter
    request_counter += 1
    return request_counter

def map_model_name(cursor_model: str, db: Session = None, provider_id: int = None) -> str:
    """将 Cursor 的模型名映射到第三方实际的模型名
    
    优先顺序:
    1. 查询数据库中指定厂商的动态映射
    2. 查询数据库中全局的动态映射（provider_id为NULL）
    3. 使用配置文件中的静态映射
    4. 原样返回
    
    Args:
        cursor_model: 原始模型名
        db: 数据库会话
        provider_id: 厂商ID，如果指定则优先查找该厂商的映射
    """
    original_model = cursor_model
    
    # 1. 先查询数据库中的动态映射
    if db:
        # 1.1 如果指定了厂商，先查该厂商的专属映射
        if provider_id:
            provider_mapping = db.query(ModelMapping).filter(
                ModelMapping.provider_id == provider_id,
                ModelMapping.source_model == cursor_model,
                ModelMapping.is_active == True
            ).first()
            if provider_mapping:
                log_info(f"模型厂商映射 [{provider_id}]: {original_model} -> {provider_mapping.target_model}")
                return provider_mapping.target_model
        
        # 1.2 查询全局映射（provider_id为NULL）
        global_mapping = db.query(ModelMapping).filter(
            ModelMapping.provider_id.is_(None),
            ModelMapping.source_model == cursor_model,
            ModelMapping.is_active == True
        ).first()
        if global_mapping:
            log_info(f"模型全局映射: {original_model} -> {global_mapping.target_model}")
            return global_mapping.target_model
    
    # 2. 使用配置文件中的静态映射
    mapped_model = MODEL_MAPPING.get(cursor_model, cursor_model)
    if original_model != mapped_model:
        log_info(f"模型静态映射: {original_model} -> {mapped_model}")
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
            model_mapped=map_model_name(body.get("model", ""), db),
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

# ==================== Token 验证 ====================

async def verify_client_token(authorization: Optional[str], db: Session) -> Tuple[bool, Optional[ApiToken], str]:
    """验证客户端 Token
    
    Returns:
        (是否有效, Token对象, 错误信息)
    """
    if not authorization:
        return False, None, "缺少 Authorization header"
    
    # 提取 token
    token_key = authorization
    if authorization.lower().startswith("bearer "):
        token_key = authorization[7:].strip()
    
    # 查询数据库
    token = db.query(ApiToken).filter(ApiToken.token_key == token_key).first()
    
    if not token:
        return False, None, "无效的 API Token"
    
    if not token.is_active:
        return False, None, "API Token 已被禁用"
    
    # 检查是否过期
    if token.expires_at and token.expires_at < datetime.now():
        return False, None, "API Token 已过期"
    
    # 更新使用信息
    token.use_count += 1
    token.last_used_at = datetime.now()
    db.commit()
    
    return True, token, ""


# ==================== API 路由 ====================

@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    """获取可用模型列表"""
    req_id = get_request_id()
    log_info(f"[Req {req_id}] GET /v1/models")
    
    # 验证 Token
    db = SessionLocal()
    try:
        is_valid, token, error_msg = await verify_client_token(authorization, db)
        if not is_valid:
            log_info(f"[Req {req_id}] Token 验证失败: {error_msg}")
            return JSONResponse(
                status_code=401,
                content={"error": {"message": error_msg, "type": "authentication_error"}}
            )
    finally:
        db.close()
    
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
    
    # 验证 Token
    db = SessionLocal()
    try:
        is_valid, token, error_msg = await verify_client_token(authorization, db)
        if not is_valid:
            log_info(f"[Req {req_id}] Token 验证失败: {error_msg}")
            return JSONResponse(
                status_code=401,
                content={"error": {"message": error_msg, "type": "authentication_error"}}
            )
        
        # 打印日志
        log_info(f"[Req {req_id}] 收到请求: POST /v1/chat/completions", {
            "request_id": db_request_id,
            "token": token.name,
            "模型": body_json.get("model"),
            "消息数": len(body_json.get("messages", [])),
            "stream": body_json.get("stream", False)
        })
        
        save_request_to_db(db, db_request_id, headers, body_json, client_ip)
        
        # 模型名称映射（使用数据库查询动态映射）
        original_model = body_json.get("model", "")
        body_json["model"] = map_model_name(original_model, db)
    finally:
        db.close()
    
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
    # 验证 Token
    db = SessionLocal()
    try:
        is_valid, token, error_msg = await verify_client_token(authorization, db)
        if not is_valid:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": error_msg, "type": "authentication_error"}}
            )
    finally:
        db.close()
    
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if "model" in body:
        # 查询数据库中的动态映射
        db = SessionLocal()
        try:
            body["model"] = map_model_name(body["model"], db)
        finally:
            db.close()
    
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
    # 验证 Token
    db = SessionLocal()
    try:
        is_valid, token, error_msg = await verify_client_token(authorization, db)
        if not is_valid:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": error_msg, "type": "authentication_error"}}
            )
    finally:
        db.close()
    
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if "model" in body:
        # 查询数据库中的动态映射
        db = SessionLocal()
        try:
            body["model"] = map_model_name(body["model"], db)
        finally:
            db.close()
    
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

# ==================== 管理后台认证路由 ====================

@app.get("/admin/login")
async def admin_login_page(request: Request, error: str = None):
    """登录页面"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error
    })

@app.post("/admin/login")
async def admin_login(request: Request, db: Session = Depends(get_db)):
    """登录处理"""
    form = await request.form()
    username = form.get("username", "").strip()
    password = form.get("password", "")
    
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "用户名或密码错误"
        })
    
    # 更新最后登录时间
    user.last_login = datetime.now()
    db.commit()
    
    # 创建 session token
    session_token = f"{user.username}|{time.time()}"
    
    response = RedirectResponse(url="/admin/", status_code=302)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax"
    )
    return response

@app.get("/admin/logout")
async def admin_logout():
    """登出"""
    response = RedirectResponse(url="/admin/login", status_code=302)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response

# ==================== 管理后台路由 ====================

def require_login(request: Request, db: Session = Depends(get_db)):
    """检查是否已登录"""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
    
    try:
        username, timestamp = session_token.split("|", 1)
        if time.time() - float(timestamp) > SESSION_MAX_AGE:
            raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
        
        user = db.query(User).filter(User.username == username, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=302, headers={"Location": "/admin/login"})
        
        return user
    except Exception:
        raise HTTPException(status_code=302, headers={"Location": "/admin/login"})

@app.get("/admin")
async def admin_redirect():
    return RedirectResponse(url="/admin/")

@app.get("/admin/")
async def admin_dashboard(
    request: Request, 
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
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
        "model_stats": model_stats,
        "user": user
    })

@app.get("/admin/requests")
async def admin_requests(
    request: Request, 
    page: int = 1, 
    per_page: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
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
        "total_pages": total_pages,
        "total": total,
        "user": user
    })

@app.get("/admin/request/{request_id}")
async def admin_request_detail(
    request_id: str, 
    request: Request, 
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
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
        "response_body_json": response_body_json,
        "user": user
    })


@app.get("/admin/profile")
async def admin_profile(request: Request, user: User = Depends(require_login)):
    """个人资料页面 - 修改密码"""
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": user,
        "success": None,
        "error": None
    })


@app.post("/admin/profile")
async def admin_profile_update(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """更新密码"""
    form = await request.form()
    current_password = form.get("current_password", "")
    new_password = form.get("new_password", "")
    confirm_password = form.get("confirm_password", "")
    
    error = None
    success = None
    
    # 验证当前密码
    if not verify_password(current_password, user.password_hash):
        error = "当前密码错误"
    # 验证新密码长度
    elif len(new_password) < 6:
        error = "新密码长度至少6位"
    # 验证两次输入是否一致
    elif new_password != confirm_password:
        error = "两次输入的新密码不一致"
    else:
        # 更新密码
        user.password_hash = hash_password(new_password)
        db.commit()
        success = "密码修改成功！下次登录请使用新密码。"
    
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": user,
        "success": success,
        "error": error
    })


# ==================== 模型映射管理 ====================

@app.get("/admin/models")
async def admin_models(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """模型映射管理页面"""
    # 获取所有动态映射（包括关联的厂商）
    dynamic_mappings = db.query(ModelMapping).options(
        joinedload(ModelMapping.provider)
    ).order_by(ModelMapping.created_at.desc()).all()
    
    # 获取所有厂商供选择
    providers = db.query(ApiProvider).filter(ApiProvider.is_active == True).all()
    
    # 配置文件中的静态映射
    static_mappings = MODEL_MAPPING
    
    return templates.TemplateResponse("models.html", {
        "request": request,
        "user": user,
        "dynamic_mappings": dynamic_mappings,
        "static_mappings": static_mappings,
        "providers": providers
    })


@app.post("/admin/models/add")
async def admin_model_add(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """添加模型映射"""
    form = await request.form()
    provider_id = form.get("provider_id", "").strip()
    source_model = form.get("source_model", "").strip()
    target_model = form.get("target_model", "").strip()
    description = form.get("description", "").strip()
    
    error = None
    
    # 转换 provider_id
    provider_id_int = int(provider_id) if provider_id and provider_id.isdigit() else None
    
    # 验证
    if not source_model or not target_model:
        error = "源模型和目标模型不能为空"
    elif source_model == target_model:
        error = "源模型和目标模型不能相同"
    else:
        # 检查是否已存在（同一厂商下源模型名唯一）
        existing_query = db.query(ModelMapping).filter(
            ModelMapping.source_model == source_model,
            ModelMapping.provider_id == provider_id_int
        )
        existing = existing_query.first()
        
        if existing:
            provider_name = "全局" if not provider_id_int else "该厂商"
            error = f"模型 '{source_model}' 在{provider_name}下的映射已存在"
        else:
            # 创建新映射
            mapping = ModelMapping(
                provider_id=provider_id_int,
                source_model=source_model,
                target_model=target_model,
                description=description,
                is_active=True
            )
            db.add(mapping)
            db.commit()
            
            return RedirectResponse(url="/admin/models", status_code=302)
    
    # 有错误，返回页面
    dynamic_mappings = db.query(ModelMapping).options(
        joinedload(ModelMapping.provider)
    ).order_by(ModelMapping.created_at.desc()).all()
    providers = db.query(ApiProvider).filter(ApiProvider.is_active == True).all()
    static_mappings = MODEL_MAPPING
    
    return templates.TemplateResponse("models.html", {
        "request": request,
        "user": user,
        "dynamic_mappings": dynamic_mappings,
        "static_mappings": static_mappings,
        "providers": providers,
        "error": error,
        "form_data": {
            "provider_id": provider_id,
            "source_model": source_model,
            "target_model": target_model,
            "description": description
        }
    })


@app.post("/admin/models/{mapping_id}/edit")
async def admin_model_edit(
    mapping_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """编辑模型映射"""
    form = await request.form()
    target_model = form.get("target_model", "").strip()
    description = form.get("description", "").strip()
    is_active = form.get("is_active") == "on"
    
    mapping = db.query(ModelMapping).filter(ModelMapping.id == mapping_id).first()
    
    if not mapping:
        raise HTTPException(status_code=404, detail="映射不存在")
    
    if not target_model:
        dynamic_mappings = db.query(ModelMapping).order_by(ModelMapping.created_at.desc()).all()
        static_mappings = MODEL_MAPPING
        return templates.TemplateResponse("models.html", {
            "request": request,
            "user": user,
            "dynamic_mappings": dynamic_mappings,
            "static_mappings": static_mappings,
            "error": "目标模型不能为空",
            "edit_id": mapping_id
        })
    
    # 更新
    mapping.target_model = target_model
    mapping.description = description
    mapping.is_active = is_active
    mapping.updated_at = datetime.now()
    db.commit()
    
    return RedirectResponse(url="/admin/models", status_code=302)


@app.get("/admin/models/{mapping_id}/delete")
async def admin_model_delete(
    mapping_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """删除模型映射"""
    mapping = db.query(ModelMapping).filter(ModelMapping.id == mapping_id).first()
    
    if not mapping:
        raise HTTPException(status_code=404, detail="映射不存在")
    
    db.delete(mapping)
    db.commit()
    
    return RedirectResponse(url="/admin/models", status_code=302)


# ==================== API 厂商管理 ====================

@app.get("/admin/providers")
async def admin_providers(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """API 厂商管理页面"""
    providers = db.query(ApiProvider).order_by(ApiProvider.created_at.desc()).all()
    return templates.TemplateResponse("providers.html", {
        "request": request,
        "user": user,
        "providers": providers
    })


@app.post("/admin/providers/add")
async def admin_provider_add(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """添加 API 厂商"""
    form = await request.form()
    name = form.get("name", "").strip()
    base_url = form.get("base_url", "").strip()
    api_key = form.get("api_key", "").strip()
    
    error = None
    
    # 验证
    if not name or not base_url or not api_key:
        error = "厂商名称、API地址和API Key不能为空"
    elif not base_url.startswith(("http://", "https://")):
        error = "API地址必须以 http:// 或 https:// 开头"
    else:
        # 检查名称是否已存在
        existing = db.query(ApiProvider).filter(ApiProvider.name == name).first()
        if existing:
            error = f"厂商 '{name}' 已存在"
        else:
            # 创建新厂商
            provider = ApiProvider(
                name=name,
                base_url=base_url,
                api_key=api_key,
                is_active=True,
                test_status="unknown"
            )
            db.add(provider)
            db.commit()
            return RedirectResponse(url="/admin/providers", status_code=302)
    
    # 有错误，返回页面
    providers = db.query(ApiProvider).order_by(ApiProvider.created_at.desc()).all()
    return templates.TemplateResponse("providers.html", {
        "request": request,
        "user": user,
        "providers": providers,
        "error": error,
        "form_data": {"name": name, "base_url": base_url, "api_key": api_key}
    })


@app.post("/admin/providers/{provider_id}/edit")
async def admin_provider_edit(
    provider_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """编辑 API 厂商"""
    form = await request.form()
    base_url = form.get("base_url", "").strip()
    api_key = form.get("api_key", "").strip()
    is_active = form.get("is_active") == "on"
    
    provider = db.query(ApiProvider).filter(ApiProvider.id == provider_id).first()
    
    if not provider:
        raise HTTPException(status_code=404, detail="厂商不存在")
    
    if not base_url or not api_key:
        providers = db.query(ApiProvider).order_by(ApiProvider.created_at.desc()).all()
        return templates.TemplateResponse("providers.html", {
            "request": request,
            "user": user,
            "providers": providers,
            "error": "API地址和API Key不能为空",
            "edit_id": provider_id
        })
    
    if not base_url.startswith(("http://", "https://")):
        providers = db.query(ApiProvider).order_by(ApiProvider.created_at.desc()).all()
        return templates.TemplateResponse("providers.html", {
            "request": request,
            "user": user,
            "providers": providers,
            "error": "API地址必须以 http:// 或 https:// 开头",
            "edit_id": provider_id
        })
    
    # 更新
    provider.base_url = base_url
    provider.api_key = api_key
    provider.is_active = is_active
    provider.updated_at = datetime.now()
    db.commit()
    
    return RedirectResponse(url="/admin/providers", status_code=302)


@app.get("/admin/providers/{provider_id}/delete")
async def admin_provider_delete(
    provider_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """删除 API 厂商"""
    provider = db.query(ApiProvider).filter(ApiProvider.id == provider_id).first()
    
    if not provider:
        raise HTTPException(status_code=404, detail="厂商不存在")
    
    db.delete(provider)
    db.commit()
    
    return RedirectResponse(url="/admin/providers", status_code=302)


@app.post("/admin/providers/{provider_id}/test")
async def admin_provider_test(
    provider_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """测试 API 厂商"""
    provider = db.query(ApiProvider).filter(ApiProvider.id == provider_id).first()
    
    if not provider:
        raise HTTPException(status_code=404, detail="厂商不存在")
    
    # 执行测试
    test_results = await test_api_provider(provider)
    
    # 更新测试结果
    provider.test_status = test_results["status"]
    provider.test_message = test_results["message"]
    provider.test_time = datetime.now()
    db.commit()
    
    return JSONResponse(content=test_results)


async def test_api_provider(provider: ApiProvider) -> dict:
    """测试 API 厂商是否可用
    
    发送两种测试：
    1. 非流式请求
    2. 流式请求
    """
    import httpx
    
    test_model = "gpt-3.5-turbo"  # 使用通用模型测试
    test_messages = [{"role": "user", "content": "Hello, this is a test. Reply with 'OK' only."}]
    
    results = {
        "provider_id": provider.id,
        "provider_name": provider.name,
        "status": "failed",
        "message": "",
        "tests": {}
    }
    
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json"
    }
    
    base_url = provider.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    
    # 测试 1: 非流式请求
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": test_model,
                    "messages": test_messages,
                    "max_tokens": 10,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    results["tests"]["non_stream"] = {
                        "status": "success",
                        "status_code": response.status_code,
                        "model": data.get("model", "unknown"),
                        "content": data["choices"][0].get("message", {}).get("content", "")[:50]
                    }
                else:
                    results["tests"]["non_stream"] = {
                        "status": "failed",
                        "status_code": response.status_code,
                        "error": "响应格式异常"
                    }
            else:
                results["tests"]["non_stream"] = {
                    "status": "failed",
                    "status_code": response.status_code,
                    "error": response.text[:200]
                }
    except Exception as e:
        results["tests"]["non_stream"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # 测试 2: 流式请求（只检查前几行）
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": test_model,
                    "messages": test_messages,
                    "max_tokens": 10,
                    "stream": True
                }
            ) as response:
                if response.status_code == 200:
                    chunk_count = 0
                    has_data = False
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            has_data = True
                            chunk_count += 1
                            if chunk_count >= 3:  # 收到3个chunk就认为成功
                                break
                    
                    if has_data:
                        results["tests"]["stream"] = {
                            "status": "success",
                            "status_code": response.status_code,
                            "chunks_received": chunk_count
                        }
                    else:
                        results["tests"]["stream"] = {
                            "status": "failed",
                            "status_code": response.status_code,
                            "error": "未收到流式数据"
                        }
                else:
                    results["tests"]["stream"] = {
                        "status": "failed",
                        "status_code": response.status_code,
                        "error": (await response.aread()).decode()[:200]
                    }
    except Exception as e:
        results["tests"]["stream"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # 汇总结果
    non_stream_ok = results["tests"].get("non_stream", {}).get("status") == "success"
    stream_ok = results["tests"].get("stream", {}).get("status") == "success"
    
    if non_stream_ok and stream_ok:
        results["status"] = "success"
        results["message"] = "✅ 非流式和流式测试均通过"
    elif non_stream_ok:
        results["status"] = "partial"
        results["message"] = "⚠️ 非流式测试通过，流式测试失败"
    elif stream_ok:
        results["status"] = "partial"
        results["message"] = "⚠️ 流式测试通过，非流式测试失败"
    else:
        results["status"] = "failed"
        non_stream_error = results["tests"].get("non_stream", {}).get("error", "未知错误")
        results["message"] = f"❌ 测试失败: {non_stream_error[:100]}"
    
    return results


# ==================== Token 管理 ====================

def generate_api_token() -> str:
    """生成 OpenAI 兼容格式的 API Token"""
    import secrets
    import string
    # 生成随机字符串
    random_part = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    return f"sk-ts-{random_part}"


@app.get("/admin/tokens")
async def admin_tokens(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """Token 管理页面"""
    tokens = db.query(ApiToken).order_by(ApiToken.created_at.desc()).all()
    
    # 构建 API 基础地址
    host = request.headers.get('host', 'www.tokenslipper.com')
    # 如果是 IP 地址或 localhost，使用 www.tokenslipper.com
    if ':' in host or host in ['localhost', '127.0.0.1']:
        api_base_url = "http://www.tokenslipper.com/v1"
    else:
        api_base_url = f"http://{host}/v1"
    
    return templates.TemplateResponse("tokens.html", {
        "request": request,
        "user": user,
        "tokens": tokens,
        "api_base_url": api_base_url
    })


@app.post("/admin/tokens/add")
async def admin_token_add(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """创建 API Token"""
    form = await request.form()
    name = form.get("name", "").strip()
    description = form.get("description", "").strip()
    expires_days = form.get("expires_days", "").strip()
    
    error = None
    
    # 验证
    if not name:
        error = "令牌名称不能为空"
    else:
        # 生成唯一 Token
        token_key = generate_api_token()
        
        # 计算过期时间
        expires_at = None
        if expires_days and expires_days.isdigit():
            expires_at = datetime.now() + timedelta(days=int(expires_days))
        
        # 创建 Token
        token = ApiToken(
            name=name,
            token_key=token_key,
            description=description,
            is_active=True,
            expires_at=expires_at
        )
        db.add(token)
        db.commit()
        
        return RedirectResponse(url="/admin/tokens", status_code=302)
    
    # 有错误，返回页面
    tokens = db.query(ApiToken).order_by(ApiToken.created_at.desc()).all()
    host = request.headers.get('host', 'www.tokenslipper.com')
    # 如果是 IP 地址或 localhost，使用 www.tokenslipper.com
    if ':' in host or host in ['localhost', '127.0.0.1']:
        api_base_url = "http://www.tokenslipper.com/v1"
    else:
        api_base_url = f"http://{host}/v1"
    
    return templates.TemplateResponse("tokens.html", {
        "request": request,
        "user": user,
        "tokens": tokens,
        "api_base_url": api_base_url,
        "error": error,
        "form_data": {"name": name, "description": description, "expires_days": expires_days}
    })


@app.post("/admin/tokens/{token_id}/toggle")
async def admin_token_toggle(
    token_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """启用/禁用 Token"""
    token = db.query(ApiToken).filter(ApiToken.id == token_id).first()
    
    if not token:
        raise HTTPException(status_code=404, detail="Token 不存在")
    
    # 切换状态
    token.is_active = not token.is_active
    db.commit()
    
    status = "启用" if token.is_active else "禁用"
    return JSONResponse(content={"success": True, "status": status, "is_active": token.is_active})


@app.get("/admin/tokens/{token_id}/delete")
async def admin_token_delete(
    token_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_login)
):
    """删除 Token"""
    token = db.query(ApiToken).filter(ApiToken.id == token_id).first()
    
    if not token:
        raise HTTPException(status_code=404, detail="Token 不存在")
    
    db.delete(token)
    db.commit()
    
    return RedirectResponse(url="/admin/tokens", status_code=302)


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 初始化数据库
    print("正在初始化数据库...")
    init_db()
    
    # 创建默认管理员用户
    print("正在检查管理员账户...")
    create_default_user()
    
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
