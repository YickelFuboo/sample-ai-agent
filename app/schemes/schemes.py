from stringprep import in_table_d1
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List


class UserRequest(BaseModel):
    """用户请求"""
    model_id: str
    session_id: str
    message: str

class tool_result(BaseModel):
    """工具结果"""
    name: str = ""
    success: str = ""
    output: str = ""

class UserResponse(BaseModel):
    """用户响应"""
    session_id: str
    resopne: str = ""
    status: str = ""
    tool_results: List[tool_result] = Field(default_factory=list)
    timestamp: int = 0
    duration_ms: int = 0

