import json
import logging
from typing import Dict, Optional, List, Literal, Any
from anthropic import AsyncAnthropic
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo


class AnthropicStyleLLM(LLM):
    """Anthropic风格的API实现（异步）"""

    def _initialize_model(self):
        self.client = AsyncAnthropic(api_key=self.api_key)

    def _format_anthropic_prompt(self,
                               system_prompt: str,
                               user_prompt: str,
                               user_question: str,
                               history: List[Dict[str, Any]] = None) -> str:
        """格式化 Anthropic 风格的提示词

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            user_question: 用户问题
            history: 历史会话记录

        Returns:
            str: 格式化后的 Anthropic 格式消息

        Note:
            Anthropic 的消息格式要求:
            1. 必须以 "Human:" 开头
            2. "Human:" 和 "Assistant:" 必须严格交替
            3. 系统提示需要包含在第一个 Human 消息中
        """
        try:
            # 构建第一条消息，包含系统提示和首次用户输入
            messages = []

            # 第一条消息包含系统提示
            first_message = f"{system_prompt}\n\n作为 AI 助手，请按照以上要求回答我的问题。"
            messages.append(f"Human: {first_message}")
            messages.append("Assistant: 好的，我会按照要求为您提供帮助。")

            # 添加历史记录
            if history:
                for msg in history:
                    prefix = "Human: " if msg["role"] == "user" else "Assistant: "
                    messages.append(f"{prefix}{msg['content']}")
                    if msg["tool_calls"]:
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(msg["tool_calls"])
                        })
                    if msg["name"]:
                        messages.append({
                            "role": "tool",
                            "content": msg["name"]
                        })
                    if msg["tool_call_id"]:
                        messages.append({
                            "role": "tool",
                            "content": msg["tool_call_id"]
                        })
            # 添加当前问题
            current_question = f"{user_prompt}\n{user_question}" if user_prompt else user_question
            messages.append(f"Human: {current_question}")
            messages.append("Assistant:")  # Anthropic 要求以 Assistant: 结尾

            # 用换行符连接所有消息
            return "\n\n".join(messages)
        except Exception as e:
            logging.error(f"Error in _format_anthropic_prompt: {e}")
            raise e

    async def chat(self,
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  **kwargs) -> ChatResponse:
        """Anthropic风格的聊天实现（异步）"""
        try:
            prompt = self._format_anthropic_prompt(
                system_prompt, user_prompt, user_question, history
            )
            params = {
                "model": self.model_name,
                "max_tokens": self.configs.get("max_tokens", 2048),
                "temperature": self.configs.get("temperature", 0.7),
                "stream": False,
                **kwargs
            }
            response = await self.client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            content = response.content
            if isinstance(content, list) and content:
                content = getattr(content[0], "text", content[0]) if hasattr(content[0], "text") else str(content[0])
            return ChatResponse(content=content, success=True)

        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise e

    async def ask_tools(self,
                        system_prompt: str,
                        user_prompt: str,
                        user_question: str,
                        history: List[Dict[str, Any]] = None,
                        tools: Optional[List[dict]] = None,
                        tool_choice: Literal["none", "auto", "required"] = "auto",
                        **kwargs) -> AskToolResponse:
        """Anthropic风格的工具调用实现（异步）"""
        try:
            prompt = self._format_anthropic_prompt(
                system_prompt, user_prompt, user_question, history
            )
            params = {
                "model": self.model_name,
                "max_tokens": self.configs.get("max_tokens", 2048),
                "temperature": self.configs.get("temperature", 0.7),
                "stream": False,
                **kwargs
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice

            response = await self.client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            content = response.content
            if isinstance(content, list) and content:
                content = getattr(content[0], "text", content[0]) if hasattr(content[0], "text") else str(content[0])
            try:
                result = json.loads(content)
                if isinstance(result, dict) and "tool_name" in result:
                    tool_calls = [ToolInfo(
                        id="",
                        name=result["tool_name"],
                        args=result.get("tool_args", {})
                    )]
                    return AskToolResponse(
                        content=result.get("thought"),
                        tool_calls=tool_calls,
                        success=True
                    )
            except json.JSONDecodeError:
                pass
            return AskToolResponse(content=content, success=True)

        except Exception as e:
            logging.error(f"Error in ask_tools: {e}")
            raise e
