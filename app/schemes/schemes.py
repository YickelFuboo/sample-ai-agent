from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Optional, List


class UserRequest(BaseModel):
    """用户请求"""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    session_id: str
    message: str

class tool_result(BaseModel):
    """工具结果"""
    name: str = ""
    success: bool = False
    output: str = ""

class UserResponse(BaseModel):
    """用户响应"""
    session_id: str
    response: str = ""
    status: str = ""
    tool_results: List[tool_result] = Field(default_factory=list)
    timestamp: int = 0
    duration_ms: int = 0

