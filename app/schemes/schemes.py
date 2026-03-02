from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Optional, List
import json


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

# 房源项目使用
class HouseRespone(BaseModel):
    """房源项目使用"""
    message: str = ""
    houses: List[str] = []

    def to_json(self) -> str:
        """将房源项目使用转换为JSON字符串"""
        return json.dumps(self.model_dump(), ensure_ascii=False)