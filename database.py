"""
数据库配置和模型
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, BigInteger
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
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False
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
