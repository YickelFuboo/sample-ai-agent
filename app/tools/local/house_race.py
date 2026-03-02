import logging
from typing import Any
import httpx
from ..base import BaseTool
from ..schemes import ToolResult, ToolSuccessResult, ToolErrorResult

LANDMARKS_BASE_URL = "http://7.225.29.233:8080"


class GetLandMarks(BaseTool):
    """调用远程接口获取地标信息。"""

    @property
    def name(self) -> str:
        return "get_landmarks_list"

    @property
    def description(self) -> str:
        return "调用远程接口获取地标信息，需要提供用户ID用于请求头 X-User-ID。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户ID，将作为请求头 X-User-ID 传递给远程接口",
                },
            },
            "required": ["user_id"],
        }

    async def execute(self, user_id: str, **kwargs: Any) -> ToolResult:
        try:
            url = f"{LANDMARKS_BASE_URL.rstrip('/')}/api/landmarks"
            headers = {"X-User-ID": user_id}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            return ToolSuccessResult(data)
        except httpx.HTTPStatusError as e:
            logging.error("GetLandMarks HTTP error: %s %s", e.response.status_code, e.response.text)
            return ToolErrorResult(f"接口返回错误: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logging.error("GetLandMarks error: %s", e)
            return ToolErrorResult(f"获取地标失败: {str(e)}")
