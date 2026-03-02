from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal
from .schemes import ChatResponse, AskToolResponse


class LLM(ABC):
    """LLM基础抽象类"""
    def __init__(self,
                 model_name: str,
                 model_type: str,
                 api_base: str,
                 api_key: str,
                 model_id: str = "",
                 session_id: str = "",
                 **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.api_base = api_base
        self.api_key = api_key
        self.configs = kwargs
        self.model_id = model_id
        self.session_id = session_id
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        """初始化模型，包括设置 max_tokens"""
        pass

    @abstractmethod
    def chat(self,
             system_prompt: str,
             user_prompt: str,
             user_question: str,
             history: List[Dict[str, str]] = None,
             **kwargs) -> ChatResponse:
        """统一的对话方法（同步）"""
        pass

    @abstractmethod
    def ask_tools(self,
                 system_prompt: str,
                 user_prompt: str,
                 user_question: str,
                 history: List[Dict[str, str]] = None,
                 tools: Optional[List[dict]] = None,
                 tool_choice: Literal["none", "auto", "required"] = "auto",
                 **kwargs) -> AskToolResponse:
        """统一的工具调用方法（同步）"""
        pass


