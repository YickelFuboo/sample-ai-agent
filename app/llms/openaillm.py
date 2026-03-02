from typing import Dict, Optional, List, Literal, Union, AsyncGenerator, Any
import json
from openai import AsyncOpenAI
import logging
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo

class OpenAIStyleLLM(LLM):
    """OpenAI风格的API实现"""
    def _initialize_model(self):
        if self.model_id: # 大赛使用
            default_headers = {"Session-ID": self.session_id} if self.session_id else None
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=f"{self.model_id}/v1",
                timeout=60.0,
                default_headers=default_headers
            )
        else:
            self.client = AsyncOpenAI(
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

    async def chat(self,
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  stream: bool = False,
                  history: List[Dict[str, Any]] = None,
                  **kwargs) -> Union[AsyncGenerator[str, None], ChatResponse]:
        """OpenAI风格的聊天实现"""
        try:
            messages = self._format_openai_message(
                system_prompt, user_prompt, user_question, history
            )

            params = {
                "stream": stream,
                "temperature": kwargs.get("temperature", self.configs.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.configs.get("max_tokens", 2048)),
            }
            # 添加其他参数，避免重复
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            # 流式响应
            if stream:
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
                async def stream_response():
                    try:
                        async for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                yield chunk.choices[0].delta.content
                    except Exception as e:
                        logging.error(f"Error in stream response: {e}")
                        if hasattr(response, 'close'):
                            await response.close()
                        raise
                return stream_response()

            # 非流式响应
            response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
            return ChatResponse(
                content=response.choices[0].message.content.strip(),
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
        """OpenAI风格的工具调用实现"""
        try:
            # 参数验证
            if tool_choice == "required" and not tools:
                raise ValueError("tool_choice 为 'required' 时必须提供 tools")

            messages = self._format_openai_message(
                system_prompt, user_prompt, user_question, history
            )

            params = {
                "stream": stream,
                "temperature": kwargs.get("temperature", self.configs.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.configs.get("max_tokens", 2048))
            }

            if tools and tool_choice != "none":
                params["tools"] = tools
                params["tool_choice"] = tool_choice

            # 添加其他参数，避免重复
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            if stream:
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
                async def stream_response():
                    collected = {
                        "content": [],
                        "tools": []  # 收集工具调用信息
                    }
                    try:
                        async for chunk in response:
                            if chunk.choices[0].delta.tool_calls:
                                # 处理工具调用
                                tool_call = chunk.choices[0].delta.tool_calls[0]
                                if tool_call.function:
                                    # 确保 arguments 是有效的 JSON 字符串
                                    arguments = tool_call.function.arguments or "{}"
                                    try:
                                        args = json.loads(arguments)
                                    except json.JSONDecodeError:
                                        args = arguments

                                    tool_info = ToolInfo(
                                        name=tool_call.function.name,
                                        args=args
                                    )
                                    if tool_info not in collected["tools"]:
                                        collected["tools"].append(tool_info)
                            elif chunk.choices[0].delta.content:
                                # 处理普通内容
                                content = chunk.choices[0].delta.content
                                collected["content"].append(content)
                                yield content

                        # 最后返回完整结果
                        if collected["tools"]:
                            yield AskToolResponse(
                                content="".join(collected["content"]),
                                tool_calls=collected["tools"],
                                success=True
                            )
                    except Exception as e:
                        logging.error(f"Error in stream response: {e}")
                        if hasattr(response, 'close'):
                            await response.close()
                        raise
                return stream_response()

            # 非流式响应
            response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
            # 检查响应结构是否有效
            if (not response.choices or not response.choices[0].message):
                return AskToolResponse(
                    content="Invalid response structure",
                    success=False
                )

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

            # 比赛打印用
            if response.usage.total_tokens and response.usage.total_tokens > 300:
                logging.error(f"=======Total tokens exceeded 300: {response.usage.total_tokens}")

            return AskToolResponse(
                content=msg.content or "",
                tool_calls=tool_calls,
                success=True
            )

        except Exception as e:
            logging.error(f"Error in ask_tools: {e}")
            raise e
