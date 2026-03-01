from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .message import Message


class Session(BaseModel):
    """会话数据模型：仅负责会话元数据与消息列表，不包含压缩逻辑。"""

    session_id: str
    description: Optional[str] = None
    session_type: str
    user_id: str

    llm_provider: str
    llm_model: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # 会话历史信息
    messages: List[Message] = Field(default_factory=list)  # 历史会话记录

    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def model_dump(self) -> Dict[str, Any]:
        """序列化。"""
        return {
            "session_id": self.session_id,
            "description": self.description,
            "session_type": self.session_type,
            "user_id": self.user_id,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "metadata": self.metadata,
            "messages": [msg.model_dump() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    def add_message(self, message: Message) -> None:
        """追加一条消息，不执行压缩。需压缩时由调用方使用 context_compressor.get_context_for_llm。"""
        self.messages.append(message)
        self.last_updated = datetime.now()

    def get_messages(self) -> List[Message]:
        """返回会话消息列表。"""
        return self.messages

    def get_context(self, max_messages: int = 500) -> List[Dict[str, Any]]:
        """返回最近若干条消息的上下文格式，供 LLM 使用。"""
        return [s.to_context() for s in self.messages[-max_messages:]]

    def clear(self) -> None:
        """清空会话历史。"""
        self.messages.clear()
        self.last_updated = datetime.now()

    def to_information(self) -> Dict[str, Any]:
        """会话关键信息，供 API 列表等使用。"""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "description": self.description,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "metadata": self.metadata,
        }

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)
