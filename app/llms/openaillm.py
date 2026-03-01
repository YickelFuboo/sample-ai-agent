from typing import Dict, Optional, List, Literal, Union, AsyncGenerator, Any
import json
from openai import AsyncOpenAI
from app.logger import logger
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo

class OpenAIStyleLLM(LLM):
    """OpenAI风格的API实现"""    
    def _initialize_model(self):
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
 
            # 添加当前问题
            messages.append({"role": "user", "content": f"{user_prompt}\n{user_question}"})
        
            return messages
        except Exception as e:
            logger.error(f"Error in _format_openai_message: {e}")
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
                "model": self.model_name,
                "messages": messages,
                "stream": stream,
                "temperature": self.model_params.get("temperature", 0.7),
                "max_tokens": self.model_params.get("max_tokens", 2048),
                **kwargs
            }

            # 流式响应
            if stream:
                response = await self.client.chat.completions.create(**params)
                async def stream_response():
                    try:
                        async for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                yield chunk.choices[0].delta.content
                    except Exception as e:
                        logger.error(f"Error in stream response: {e}")
                        if hasattr(response, 'close'):
                            await response.close()
                        raise
                return stream_response()
            
            # 非流式响应
            response = await self.client.chat.completions.create(**params)
            return ChatResponse(
                content=response.choices[0].message.content,
                success=True
            )

        except Exception as e:
            logger.error(f"Error in chat: {e}")
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
                "model": self.model_name,
                "messages": messages,
                "stream": stream,
                "temperature": self.model_params.get("temperature", 0.7),
                "max_tokens": self.model_params.get("max_tokens", 2048)
            }

            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            params.update(kwargs)

            if stream:
                response = await self.client.chat.completions.create(**params)
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
                        logger.error(f"Error in stream response: {e}")
                        if hasattr(response, 'close'):
                            await response.close()
                        raise
                return stream_response()

            # 非流式响应
            response = await self.client.chat.completions.create(**params)
            msg = response.choices[0].message
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # 处理工具调用
                tool_calls = []
                for tool_call in msg.tool_calls:
                    # 确保 arguments 是有效的 JSON 字符串
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
            
            return AskToolResponse(
                content=msg.content,
                success=True
            )

        except Exception as e:
            logger.error(f"Error in ask_tools: {e}")
            raise e
