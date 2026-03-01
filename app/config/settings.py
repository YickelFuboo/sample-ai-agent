import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from app.utils.common import get_project_meta, get_project_base_directory


# 定义全局配置常量
_meta = get_project_meta()
APP_NAME = _meta["name"]
APP_VERSION = _meta["version"]
APP_DESCRIPTION = _meta["description"]

PROJECT_BASE_DIR = get_project_base_directory()

class Settings(BaseSettings):
    """应用配置类 - 平铺结构"""
    
    # 应用基础配置
    service_host: str = Field(default="0.0.0.0", description="服务主机地址", env="SERVICE_HOST")
    service_port: int = Field(default=8000, description="服务端口", env="SERVICE_PORT")
    debug: bool = Field(default=False, description="调试模式", env="DEBUG")
    app_log_level: str = Field(default="INFO", description="日志级别", env="APP_LOG_LEVEL")

    # =============================================================================
    # Agent配置 - Agent 会话存储
    # =============================================================================
    agent_session_storage_dir: str = Field(default="data/sessions", description="本地会话文件目录(仅本地存储时生效)", env="AGENT_SESSION_STORAGE_DIR")

    class Config:
        env_file = os.path.join(PROJECT_BASE_DIR, "env")
        env_file_encoding = "utf-8"
        extra = "ignore"

# 全局配置实例
settings = Settings() 