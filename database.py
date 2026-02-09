"""
数据库配置和模型
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, BigInteger, Boolean, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()

# MySQL 配置
DB_HOST = os.getenv("DB_HOST", "192.168.1.22")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "censor")
DB_PASSWORD = os.getenv("DB_PASSWORD", "jhshgU2-madksjhuv-sisidmtud")
DB_NAME = os.getenv("DB_NAME", "tokenslipper")

# 创建数据库连接 URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4&use_unicode=1&collation=utf8mb4_unicode_ci"

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
    connect_args={
        'charset': 'utf8mb4',
        'use_unicode': True
    }
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()


class RequestLog(Base):
    """请求日志表"""
    __tablename__ = "request_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), unique=True, index=True, comment="请求唯一ID")
    timestamp = Column(DateTime, default=datetime.now, index=True, comment="请求时间")
    method = Column(String(10), comment="HTTP方法")
    path = Column(String(255), comment="请求路径")
    client_ip = Column(String(50), comment="客户端IP")
    user_agent = Column(String(500), comment="User-Agent")
    
    # 请求内容
    model_requested = Column(String(100), comment="请求的模型名(Cursor发送的)")
    model_mapped = Column(String(100), comment="映射后的模型名")
    temperature = Column(Float, nullable=True, comment="temperature参数")
    max_tokens = Column(Integer, nullable=True, comment="max_tokens参数")
    stream = Column(Integer, default=0, comment="是否流式请求 0/1")
    
    # 完整请求体（JSON格式）
    request_body = Column(JSON, comment="完整请求体")
    
    # 关联的消息
    messages = relationship("Message", back_populates="request", cascade="all, delete-orphan")
    
    # 关联的响应
    response = relationship("ResponseLog", back_populates="request", uselist=False, cascade="all, delete-orphan")
    
    # 统计
    message_count = Column(Integer, default=0, comment="消息数量")
    
    def __repr__(self):
        return f"<RequestLog(id={self.id}, request_id={self.request_id}, model={self.model_mapped})>"


class Message(Base):
    """消息表（存储每次对话的上下文）"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), ForeignKey("request_logs.request_id"), index=True)
    role = Column(String(20), comment="角色: system/user/assistant")
    content = Column(Text, comment="消息内容")
    content_preview = Column(String(200), comment="内容前200字符预览")
    message_index = Column(Integer, comment="消息序号")
    
    request = relationship("RequestLog", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, request_id={self.request_id})>"


class User(Base):
    """用户表（管理后台登录）"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    is_active = Column(Boolean, default=True, comment="是否激活")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    last_login = Column(DateTime, nullable=True, comment="最后登录时间")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class ApiToken(Base):
    """API Token 表（供用户使用）"""
    __tablename__ = "api_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, comment="令牌名称")
    token_key = Column(String(100), unique=True, index=True, nullable=False, comment="Token 值")
    description = Column(String(255), nullable=True, comment="描述")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    expires_at = Column(DateTime, nullable=True, comment="过期时间")
    last_used_at = Column(DateTime, nullable=True, comment="最后使用时间")
    use_count = Column(Integer, default=0, comment="使用次数")
    
    def __repr__(self):
        return f"<ApiToken(id={self.id}, name={self.name}, active={self.is_active})>"


class ApiProvider(Base):
    """API 厂商表"""
    __tablename__ = "api_providers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, comment="厂商名称")
    base_url = Column(String(500), nullable=False, comment="API 基础 URL")
    api_key = Column(String(500), nullable=False, comment="API Key")
    is_active = Column(Boolean, default=True, comment="是否启用")
    test_status = Column(String(20), default="unknown", comment="测试状态: success/failed/unknown")
    test_message = Column(Text, nullable=True, comment="测试结果信息")
    test_time = Column(DateTime, nullable=True, comment="最后测试时间")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    def __repr__(self):
        return f"<ApiProvider(id={self.id}, name={self.name}, status={self.test_status})>"


class ModelMapping(Base):
    """模型映射表（动态配置）"""
    __tablename__ = "model_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("api_providers.id"), nullable=True, comment="关联厂商ID，NULL表示全局映射")
    source_model = Column(String(100), index=True, nullable=False, comment="源模型名（如cursor发送的）")
    target_model = Column(String(100), nullable=False, comment="目标模型名（映射后的）")
    description = Column(String(255), nullable=True, comment="描述说明")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    # 关联厂商
    provider = relationship("ApiProvider")
    
    __table_args__ = (
        # 同一厂商下源模型名唯一
        UniqueConstraint('provider_id', 'source_model', name='uix_provider_source_model'),
    )
    
    def __repr__(self):
        provider_name = self.provider.name if self.provider else "全局"
        return f"<ModelMapping(id={self.id}, {provider_name}: {self.source_model} -> {self.target_model})>"


class ResponseLog(Base):
    """响应日志表"""
    __tablename__ = "response_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), ForeignKey("request_logs.request_id"), unique=True, index=True)
    
    # 响应时间和状态
    response_timestamp = Column(DateTime, default=datetime.now, comment="响应时间")
    status_code = Column(Integer, comment="HTTP状态码")
    upstream_latency = Column(Float, comment="上游API响应耗时(秒)")
    total_latency = Column(Float, comment="总耗时(秒)")
    
    # 响应内容
    model_responded = Column(String(100), comment="实际响应的模型名")
    finish_reason = Column(String(50), nullable=True, comment="结束原因")
    
    # Token 使用情况
    prompt_tokens = Column(Integer, nullable=True, comment="Prompt tokens")
    completion_tokens = Column(Integer, nullable=True, comment="Completion tokens")
    total_tokens = Column(Integer, nullable=True, comment="Total tokens")
    
    # 完整响应内容
    response_content = Column(Text, comment="完整的响应内容（非流式）或合并后的内容（流式）")
    response_content_preview = Column(String(200), comment="响应内容前200字符预览")
    
    # 流式响应统计
    is_stream = Column(Integer, default=0, comment="是否流式响应 0/1")
    chunk_count = Column(Integer, nullable=True, comment="流式响应的chunk数量")
    
    # 错误信息
    error_message = Column(Text, nullable=True, comment="错误信息")
    
    # 完整响应体（JSON格式）
    response_body = Column(JSON, nullable=True, comment="完整响应体")
    
    request = relationship("RequestLog", back_populates="response")
    
    def __repr__(self):
        return f"<ResponseLog(id={self.id}, request_id={self.request_id}, status={self.status_code})>"


# 创建所有表
def init_db():
    """初始化数据库，创建所有表"""
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表已创建/更新")


# 获取数据库会话
def get_db():
    """获取数据库会话，用于依赖注入"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
