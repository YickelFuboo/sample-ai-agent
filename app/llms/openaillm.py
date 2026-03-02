import json
import logging
from typing import Dict, Optional, List, Literal, Any
from openai import OpenAI
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo

class OpenAIStyleLLM(LLM):
    """OpenAI风格的API实现（使用同步客户端）"""
    def _initialize_model(self):
        if self.model_id:  # 大赛使用：model_id 为 IPv4 地址，端口固定 8888
            default_headers = {"Session-ID": self.session_id} if self.session_id else None
            base_url = f"http://{self.model_id}:8888/v1"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=60.0,
                default_headers=default_headers
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=60.0
            )

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

    def chat(self,
             system_prompt: str,
             user_prompt: str,
             user_question: str,
             history: List[Dict[str, Any]] = None,
             **kwargs) -> ChatResponse:
        """OpenAI风格的聊天实现（同步）"""
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

            response = self.client.chat.completions.create(
                model=model, messages=messages, **params
            )
            return ChatResponse(
                content=response.choices[0].message.content.strip(),
                success=True
            )

        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise e

    def ask_tools(self,
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  tools: Optional[List[dict]] = None,
                  tool_choice: Literal["none", "auto", "required"] = "auto",
                  **kwargs) -> AskToolResponse:
        """OpenAI风格的工具调用实现（同步）"""
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

            response = self.client.chat.completions.create(
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

            return AskToolResponse(
                content=msg.content or "",
                tool_calls=tool_calls,
                success=True
            )

        except Exception as e:
            logging.error(f"Error in ask_tools: {e}")
            raise e
