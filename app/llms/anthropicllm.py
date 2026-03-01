from typing import Dict, Optional, List, Literal, Union, AsyncGenerator, Any
import json
from anthropic import AsyncAnthropic
import logging
from .base import LLM
from .schemes import ChatResponse, AskToolResponse


class AnthropicStyleLLM(LLM):
    """Anthropic风格的API实现"""

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
                  stream: bool = False,
                  history: List[Dict[str, Any]] = None,
                  **kwargs) -> Union[AsyncGenerator[str, None], ChatResponse]:
        """Anthropic风格的聊天实现"""
        try:
            prompt = self._format_anthropic_prompt(
                system_prompt, user_prompt, user_question, history
            )

            params = {
                "model": self.model_name,
                "max_tokens": self.configs.get("max_tokens", 2048),
                "temperature": self.configs.get("temperature", 0.7),
                "stream": stream,
                **kwargs
            }

            if stream:
                response = await self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                async def stream_response():
                    try:
                        async for chunk in response:
                            if chunk.content:
                                yield chunk.content
                    except Exception as e:
                        logging.error(f"Error in stream response: {e}")
                        if response and hasattr(response, 'close'):
                            await response.close()
                        raise
                return stream_response()

            response = await self.client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return ChatResponse(
                content=response.content,
                success=True
            )

        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise e

    async def ask_tools(self,
                       system_prompt: str,
                       user_prompt: str,
                       user_question: str,
                       history: List[Dict[str, Any]] = None,
                       stream: bool = False,
                       tools: Optional[List[dict]] = None,
                       tool_choice: Literal["none", "auto", "required"] = "auto",
                       **kwargs) -> Union[AsyncGenerator[Union[str, AskToolResponse], None], AskToolResponse]:
        """Anthropic风格的工具调用实现

        Note:
            工具格式示例:
            tools = [{
                "name": "search",
                "description": "搜索信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }]

            返回格式示例:
            {
                "tool_name": "search",
                "tool_args": {"query": "Python单例模式"},
                "thought": "需要搜索相关资料"
            }
        """
        try:
            prompt = self._format_anthropic_prompt(
                system_prompt, user_prompt, user_question, history
            )

            params = {
                "model": self.model_name,
                "max_tokens": self.configs.get("max_tokens", 2048),
                "temperature": self.configs.get("temperature", 0.7),
                "stream": stream,
                **kwargs
            }

            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice

            if stream:
                response = await self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                async def stream_response():
                    collected_content = []
                    try:
                        async for chunk in response:
                            if chunk.content:
                                collected_content.append(chunk.content)
                                yield chunk.content

                        # 处理完整响应
                        full_content = "".join(collected_content)
                        try:
                            result = json.loads(full_content)
                            if isinstance(result, dict) and "tool_name" in result:
                                yield AskToolResponse(
                                    content=result.get("thought"),
                                    tool_name=result["tool_name"],
                                    tool_args=result["tool_args"],
                                    success=True
                                )
                                return
                        except json.JSONDecodeError:
                            pass

                        yield AskToolResponse(
                            content=full_content,
                            success=True
                        )
                    except Exception as e:
                        logging.error(f"Error in stream response: {e}")
                        if response and hasattr(response, 'close'):
                            await response.close()
                        raise
                return stream_response()

            # 非流式响应
            response = await self.client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            content = response.content

            # 尝试解析工具调用
            try:
                result = json.loads(content)
                if isinstance(result, dict) and "tool_name" in result:
                    return AskToolResponse(
                        content=result.get("thought"),
                        tool_name=result["tool_name"],
                        tool_args=result["tool_args"],
                        success=True
                    )
            except json.JSONDecodeError:
                pass

            return AskToolResponse(
                content=content,
                success=True
            )

        except Exception as e:
            logging.error(f"Error in ask_tools: {e}")
            raise e
