import json
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from openai import AsyncOpenAI
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo

MAX_LLM_CACHE_SIZE = 256


def _chat_cache_key(model: str, messages: List[Dict[str, Any]], temperature: Any, max_tokens: Any) -> Tuple:
    """chat 请求的缓存键：model + messages + 主要参数。"""
    return ("chat", model, json.dumps(messages, sort_keys=True), temperature, max_tokens)


def _ask_tools_cache_key(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[dict]],
    tool_choice: str,
    temperature: Any,
    max_tokens: Any,
) -> Tuple:
    """ask_tools 请求的缓存键。"""
    return (
        "ask_tools",
        model,
        json.dumps(messages, sort_keys=True),
        json.dumps(tools or [], sort_keys=True),
        tool_choice,
        temperature,
        max_tokens,
    )


class OpenAIStyleLLM(LLM):
    """OpenAI风格的API实现（使用异步客户端），相同 messages 与参数时返回缓存结果。"""
    def _initialize_model(self):
        if self.model_id:  # 大赛使用：model_id 为 IPv4 地址，端口固定 8888
            default_headers = {"Session-ID": self.session_id} if self.session_id else None
            base_url = f"http://{self.model_id}:8888/v1"
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=60.0,
                default_headers=default_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=60.0
            )
        self._response_cache: Dict[Tuple, Union[ChatResponse, AskToolResponse]] = {}
        self._max_cache_size = MAX_LLM_CACHE_SIZE

    def _format_openai_message(
        self,
        system_prompt: str,
        user_prompt: str,
        user_question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """格式化消息为 OpenAI API 所需的格式

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            user_question: 用户问题
            history: 历史会话记录，使用 Message 类型

        Returns:
            List[Dict[str, Any]]: OpenAI 格式的消息列表
        """
        try:
            # 添加信息信息
            messages = [{"role": "system", "content": system_prompt}]

            # 处理历史记录
            if history:
                messages.extend(history)

            # 如果有单独的用户问题信息，则添加用户问题信息
            if user_question:
                user_message = f"{user_prompt}\n{user_question}" if user_prompt else user_question
                messages.append({"role": "user", "content": user_message})

            # 如果messages为空
            if not messages:
                logging.error("Messages are empty")
                raise ValueError("Messages are empty")

            return messages
        except Exception as e:
            logging.error(f"Error in _format_openai_message: {e}")
            raise e

    def _cache_put(self, key: Tuple, value: Union[ChatResponse, AskToolResponse]) -> None:
        """写入缓存并限制容量。"""
        if self._max_cache_size and len(self._response_cache) >= self._max_cache_size:
            oldest = next(iter(self._response_cache))
            del self._response_cache[oldest]
        self._response_cache[key] = value

    async def chat(self,
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  **kwargs) -> ChatResponse:
        """OpenAI风格的聊天实现（异步），messages 与参数相同时返回缓存。"""
        try:
            messages = self._format_openai_message(
                system_prompt, user_prompt, user_question, history
            )
            model = self.model_name or self.configs.get("default_model", "default")
            params = {
                "stream": False,
                "temperature": kwargs.get("temperature", self.configs.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.configs.get("max_tokens", 2048)),
            }
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            cache_key = _chat_cache_key(
                model, messages, params["temperature"], params["max_tokens"]
            )
            if cache_key in self._response_cache:
                logging.info("OpenAIStyleLLM chat cache hit")
                return self._response_cache[cache_key]

            response = await self.client.chat.completions.create(
                model=model, messages=messages, **params
            )
            result = ChatResponse(
                content=response.choices[0].message.content.strip(),
                success=True
            )
            self._cache_put(cache_key, result)
            return result

        except Exception as e:
            logging.error("Error in chat: %s", e)
            raise e

    async def ask_tools(self,
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  tools: Optional[List[dict]] = None,
                  tool_choice: Literal["none", "auto", "required"] = "auto",
                  **kwargs) -> AskToolResponse:
        """OpenAI风格的工具调用实现（异步）"""
        try:
            if tool_choice == "required" and not tools:
                raise ValueError("tool_choice 为 'required' 时必须提供 tools")

            messages = self._format_openai_message(
                system_prompt, user_prompt, user_question, history
            )
            model = self.model_name or self.configs.get("default_model", "default")
            params = {
                "stream": False,
                "temperature": kwargs.get("temperature", self.configs.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.configs.get("max_tokens", 2048))
            }
            if tools and tool_choice != "none":
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            cache_key = _ask_tools_cache_key(
                model, messages, tools, tool_choice,
                params.get("temperature"), params.get("max_tokens")
            )
            if cache_key in self._response_cache:
                logging.info("OpenAIStyleLLM ask_tools cache hit")
                return self._response_cache[cache_key]

            response = await self.client.chat.completions.create(
                model=model, messages=messages, **params
            )
            if not response.choices or not response.choices[0].message:
                return AskToolResponse(content="Invalid response structure", success=False)

            msg = response.choices[0].message
            tool_calls = []
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    arguments = tool_call.function.arguments
                    try:
                        args = json.loads(arguments)
                    except json.JSONDecodeError:
                        args = arguments
                    tool_calls.append(ToolInfo(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=args
                    ))

            result = AskToolResponse(
                content=msg.content or "",
                tool_calls=tool_calls,
                success=True
            )
            self._cache_put(cache_key, result)
            return result

        except Exception as e:
            logging.error(f"Error in ask_tools: {e}")
            raise e
