import json
import logging
from typing import Dict, Optional, List, Literal, Any
import httpx
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo


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
            logging.error(f"Error in _format_silicon_message: {e}")
            raise e


    def chat(self,
             system_prompt: str,
             user_prompt: str,
             user_question: str,
             stream: bool = False,
             history: List[Dict[str, Any]] = None,
             **kwargs) -> ChatResponse:
        """Silicon风格的聊天实现（同步，不支持 stream）"""
        try:
            message = self._format_silicon_message(
                system_prompt, user_prompt, user_question, history
            )
            payload = {
                "model": self.model_name,
                "messages": message,
                "stream": False,
                "temperature": self.configs.get("temperature", 0.7),
                "max_tokens": self.configs.get("max_tokens", 2048),
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
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.api_base, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get("choices"):
                return ChatResponse(
                    content=data["choices"][0]["message"]["content"],
                    success=True
                )
            return ChatResponse(content="Invalid response format from API", success=False)

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
        """Silicon风格的工具调用实现（同步）"""
        try:
            if tool_choice == "required" and not tools:
                raise ValueError("tool_choice 为 'required' 时必须提供 tools")

            message = self._format_silicon_message(
                system_prompt, user_prompt, user_question, history
            )
            payload = {
                "model": self.model_name,
                "messages": message,
                "stream": False,
                "temperature": self.configs.get("temperature", 0.7),
                "max_tokens": self.configs.get("max_tokens", 2048),
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
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.api_base, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            tool_calls_raw = msg.get("tool_calls")
            if tool_calls_raw:
                tool_calls = []
                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolInfo(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        args=args
                    ))
                return AskToolResponse(content=content, tool_calls=tool_calls, success=True)
            return AskToolResponse(content=content, success=True)

        except Exception as e:
            logging.error(f"Error in ask_tools: {e}")
            raise e