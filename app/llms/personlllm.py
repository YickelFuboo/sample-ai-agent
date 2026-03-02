import json
import logging
from typing import Dict, Optional, List, Literal, Any
import httpx
from .base import LLM
from .schemes import ChatResponse, AskToolResponse, ToolInfo

PERSON_LLM_PORT = 8888


class PersonLLM(LLM):
    """使用 HTTP 同步 POST 发送模型请求，URL: http://{model_id}:8888/V2/chat/completions"""

    def _initialize_model(self):
        pass

    def _format_openai_message(
        self,
        system_prompt: str,
        user_prompt: str,
        user_question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        if user_question:
            user_message = f"{user_prompt}\n{user_question}" if user_prompt else user_question
            messages.append({"role": "user", "content": user_message})
        if not messages:
            raise ValueError("Messages are empty")
        return messages

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.session_id:
            headers["Session-ID"] = self.session_id
        return headers

    def _get_chat_url(self) -> str:
        return f"http://{self.model_id}:{PERSON_LLM_PORT}/V2/chat/completions"

    def chat(self,
             system_prompt: str,
             user_prompt: str,
             user_question: str,
             history: List[Dict[str, Any]] = None,
             **kwargs) -> ChatResponse:
        """同步聊天"""
        try:
            messages = self._format_openai_message(system_prompt, user_prompt, user_question, history)
            url = self._get_chat_url()
            body = {
                "model": self.model_name or "default",
                "messages": messages,
                "stream": False,
                "temperature": kwargs.get("temperature", self.configs.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.configs.get("max_tokens", 2048)),
            }
            for key, value in kwargs.items():
                if key not in body:
                    body[key] = value
            headers = self._build_headers()

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = ""
            if data.get("choices"):
                msg = data["choices"][0].get("message", {})
                content = (msg.get("content") or "").strip()
            return ChatResponse(content=content, success=True)

        except httpx.HTTPStatusError as e:
            logging.error("PersonLLM chat HTTP error: %s %s", e.response.status_code, e.response.text)
            return ChatResponse(content=f"HTTP error: {e.response.status_code}", success=False)
        except Exception as e:
            logging.error("PersonLLM chat error: %s", e)
            return ChatResponse(content=str(e), success=False)

    def ask_tools(self,
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  tools: Optional[List[dict]] = None,
                  tool_choice: Literal["none", "auto", "required"] = "auto",
                  **kwargs) -> AskToolResponse:
        try:
            if tool_choice == "required" and not tools:
                raise ValueError("tool_choice 为 'required' 时必须提供 tools")

            messages = self._format_openai_message(system_prompt, user_prompt, user_question, history)
            url = self._get_chat_url()
            body = {
                "model": self.model_name or "default",
                "messages": messages,
                "stream": False,
                "temperature": kwargs.get("temperature", self.configs.get("temperature", 0.7)),
                "max_tokens": kwargs.get("max_tokens", self.configs.get("max_tokens", 2048)),
            }
            if tools and tool_choice != "none":
                body["tools"] = tools
                body["tool_choice"] = tool_choice
            for key, value in kwargs.items():
                if key not in body:
                    body[key] = value
            headers = self._build_headers()

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices") or []
            if not choices:
                return AskToolResponse(content="Invalid response structure", success=False)

            msg = choices[0].get("message", {})
            content = msg.get("content") or ""
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
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

        except httpx.HTTPStatusError as e:
            logging.error("PersonLLM ask_tools HTTP error: %s %s", e.response.status_code, e.response.text)
            return AskToolResponse(content=f"HTTP error: {e.response.status_code}", success=False)
        except Exception as e:
            logging.error("PersonLLM ask_tools error: %s", e)
            return AskToolResponse(content=str(e), success=False)
