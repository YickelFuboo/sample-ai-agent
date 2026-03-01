from typing import Dict, Optional, List, Literal, Union, AsyncGenerator, Any
import json
import requests
import httpx
from app.logger import logger
from .base import LLM
from .schemes import ChatResponse, AskToolResponse

 
class SiliconStyleLLM(LLM):
    """Silicon风格的API实现（如百川、智谱等）"""
     
    def _initialize_model(self):
        """初始化模型客户端"""
        pass
 
    def _format_silicon_message(self, 
                               system_prompt: str,
                               user_prompt: str,
                               user_question: str,
                               history: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """格式化 Silicon 风格的消息
         
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            user_question: 用户问题
            history: 历史会话记录
             
        Returns:
            List[Dict[str, str]]: 格式化后的消息列表
             
        Note:
            Silicon 风格API通常使用类似OpenAI的消息格式，
            但可能有一些特殊字段要求
        """
        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # 处理历史记录
            if history:
                messages.extend(history)
            
            # 合并用户提示和问题
            user_message = f"{user_prompt}\n{user_question}" if user_prompt else user_question
            messages.append({"role": "user", "content": user_message})
         
            return messages
        except Exception as e:
            logger.error(f"Error in _format_silicon_message: {e}")
            raise e
 
 
    async def chat(self, 
                   system_prompt: str,
                   user_prompt: str,
                   user_question: str,
                   stream: bool = False,
                   history: List[Dict[str, Any]] = None,
                   **kwargs) -> Union[AsyncGenerator[str, None], ChatResponse]:
        """Silicon风格的聊天实现"""
        try:
            message = self._format_silicon_message(
                system_prompt, user_prompt, user_question, history
            )
            
            payload = {
                "model": self.model_name,
                "messages": message,
                "stream": stream,
                "temperature": self.model_params.get("temperature", 0.7),
                "max_tokens": self.model_params.get("max_tokens", 2048),
                "stop": ["null"],
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"},
                **kwargs
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            if stream:
                async def stream_response():
                    response = None
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            response = await client.post(self.api_base, json=payload, headers=headers, timeout=None)
                            async for line in response.aiter_lines():
                                if line:
                                    if line.startswith("data: "):
                                        if line.strip() == "data: [DONE]":
                                            break
                                        try:
                                            data = json.loads(line[6:])
                                            if content := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                                yield content
                                        except json.JSONDecodeError:
                                            continue
                    except Exception as e:
                        logger.error(f"Error in stream response: {e}")
                        if response:
                            await response.aclose()
                        raise
                return stream_response()

            # 非流式响应
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.api_base, json=payload, headers=headers, timeout=None)
                response.raise_for_status()
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return ChatResponse(
                        content=data['choices'][0]['message']['content'],
                        success=True
                    )
                else:
                    return ChatResponse(
                        content="Invalid response format from API",
                        success=False
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
        """Silicon风格的工具调用实现
         
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
            if tool_choice == "required" and not tools:
                raise ValueError("tool_choice 为 'required' 时必须提供 tools")
            
            message = self._format_silicon_message(
                system_prompt, user_prompt, user_question, history
            )
            
            payload = {
                "model": self.model_name,
                "messages": message,
                "stream": stream,
                "temperature": self.model_params.get("temperature", 0.7),
                "max_tokens": self.model_params.get("max_tokens", 2048),
                "stop": ["null"],
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"},
                **kwargs
             }
            
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            if stream:
                async def stream_response():
                    response = None
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            response = await client.post(self.api_base, json=payload, headers=headers, timeout=None)
                            collected = {"content": [], "tool": None}
                            async for line in response.aiter_lines():
                                if line:
                                    if line.startswith("data: "):
                                        if line.strip() == "data: [DONE]":
                                            break
                                        try:
                                            data = json.loads(line[6:])
                                            if tool_calls := data.get("choices", [{}])[0].get("delta", {}).get("tool_calls"):
                                                # 处理工具调用
                                                tool_call = tool_calls[0]
                                                if not collected["tool"]:
                                                    collected["tool"] = {
                                                        "name": tool_call.get("function", {}).get("name"),
                                                        "arguments": tool_call.get("function", {}).get("arguments", "{}")
                                                    }
                                            elif content := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                                # 处理普通内容
                                                collected["content"].append(content)
                                                yield content
                                        except json.JSONDecodeError:
                                            continue
                            # 最后返回完整的工具调用结果
                            if collected["tool"]:
                                yield AskToolResponse(
                                    content="".join(collected["content"]),
                                    tool_name=collected["tool"]["name"],
                                    tool_args=json.loads(collected["tool"]["arguments"]),
                                    success=True
                                )
                    except Exception as e:
                        logger.error(f"Error in stream response: {e}")
                        if response:
                            await response.aclose()
                        raise
                return stream_response()

            # 非流式响应
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.api_base, json=payload, headers=headers, timeout=None)
                response.raise_for_status()
                data = response.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                if tool_calls := msg.get("tool_calls"):
                    tool_call = tool_calls[0]
                    return AskToolResponse(
                        content=msg.get("content", ""),
                        tool_name=tool_call.get("function", {}).get("name"),
                        tool_args=json.loads(tool_call.get("function", {}).get("arguments", "{}")),
                        success=True
                    )
                return AskToolResponse(
                    content=msg.get("content", ""),
                    success=True
                )
        except Exception as e:
            logger.error(f"Error in ask_tools: {e}")
            raise e