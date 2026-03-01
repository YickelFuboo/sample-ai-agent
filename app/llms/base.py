from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal, Union, AsyncGenerator
from .schemes import ChatResponse, AskToolResponse


class LLM(ABC):
    """LLM基础抽象类"""
    def __init__(self, 
                 model_name: str,
                 model_type: str,
                 api_base: str,
                 api_key: str,
                 **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.api_base = api_base
        self.api_key = api_key
        self.model_params = kwargs
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        """初始化模型，包括设置 max_tokens"""
        pass
    @abstractmethod
    async def chat(self, 
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  stream: bool = False,
                  history: List[Dict[str, str]] = None,
                  **kwargs) -> Union[AsyncGenerator[str, None], ChatResponse]:
        """统一的对话方法"""
        pass

    @abstractmethod
    async def ask_tools(self,
                       system_prompt: str,
                       user_prompt: str,
                       user_question: str,
                       history: List[Dict[str, str]] = None,
                       stream: bool = False,
                       tools: Optional[List[dict]] = None,
                       tool_choice: Literal["none", "auto", "required"] = "auto",
                       **kwargs) -> Union[AsyncGenerator[Union[str, AskToolResponse], None], AskToolResponse]:
        """统一的工具调用方法"""
        pass


