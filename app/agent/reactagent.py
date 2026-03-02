import asyncio
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import List, Any, Optional, Tuple
from app.tools.factory import ToolsFactory
from app.agent.base import BaseAgent, AgentState
from app.sessions.message import Role, Message, ToolCall, Function
from app.llms.factory import llm_factory
from app.tools.schemes import ToolResult, ToolResultStatus
from app.tools.local.shell import ExecTool
from app.tools.local.house_race import GetLandMarks
from app.prompts.prompt_template_load import get_prompt_template
from app.schemes.schemes import tool_result


class ToolChoice(str, Enum):
    """工具调用模式：none=不暴露工具，auto=由模型决定，required=必须调用工具。"""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


PROMPT_PATH = Path(__file__).parent.parent / "prompts"


class ReActAgent(BaseAgent):
    """ReAct 执行类，属性仅在 __init__ 内通过 self 赋值。"""

    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        agent_type: str,
        session_id: str,
        workspace_index: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        next_step_prompt: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        memory_window: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_duplicate_steps: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            agent_name=agent_name,
            agent_description=agent_description,
            agent_type=agent_type,
            session_id=session_id,
            workspace_index=workspace_index,
            user_id=user_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            next_step_prompt=next_step_prompt,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            memory_window=memory_window,
            max_steps=max_steps,
            max_duplicate_steps=max_duplicate_steps,
            **kwargs,
        )

        # 提示词信息
        self.system_prompt = get_prompt_template(str(PROMPT_PATH), "system_prompt.md") or ""
        self.user_prompt = get_prompt_template(str(PROMPT_PATH), "user_prompt.md") or ""

        # 工具信息
        self.available_tools = ToolsFactory()
        self.tool_choices = ToolChoice.AUTO
        self.special_tool_names = []
        self._register_default_tools()
        self.tool_results = []

    def reset(self):
        """重置 agent 状态到初始状态
        - 工具调用状态清空
        """
        super().reset()
        self.tool_choices = ToolChoice.AUTO
        self.special_tool_names = []
        self.tool_results = []
        logging.info(f"ReActAgent state reset to IDLE")

    def _register_default_tools(self) -> None:
        # 如果没有指定工具，则注册默认工具
        if self.available_tools:
            self.available_tools.register_tools(
                #ExecTool(),
                GetLandMarks(),
            )

    def _strip_think(text: str | None) -> str | None:
        """去掉回复中的 <think>...</think> 块（部分思考模型会内嵌），避免把思考过程当正文返回。"""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    async def run(self, question: str) -> Tuple[str, List[tool_result]]:
        """Run the agent

        Args:
            question: Input question

        Returns:
            Tuple[str, List[tool_result]]: Execution result and tool results
        """
        logging.info(f"Running agent {self.agent_name} with question: {question}")

        if not self.session_id or not self.workspace_index:
            raise ValueError("Session ID and workspace_index are required")

        # 检查并重置状态
        if self.state != AgentState.IDLE:
            logging.warning(f"Agent is busy with state {self.state}, resetting...")
            self.reset()

        try:
            # 设置运行状态
            self.state = AgentState.RUNNING
            logging.info(f"Agent state set to RUNNING")

            # 设置添加用户消息到history标志
            had_push_user_message = False

            while (self.current_step < self.max_steps and self.state != AgentState.FINISHED):
                self.current_step += 1
                logging.info(f"Executing step {self.current_step}/{self.max_steps}")

                # 模型思考和工具调度
                content, tool_calls = await self.think(question)
                if not tool_calls or self._has_special_tool(tool_calls):
                    if not had_push_user_message:
                        await self.push_history_message(self.session_id, Message.user_message(question))
                        had_push_user_message = True
                    await self.push_history_message(self.session_id, Message.assistant_message(content))
                    break
                else:
                    if not had_push_user_message:
                        await self.push_history_message(self.session_id, Message.user_message(question))
                        had_push_user_message = True
                    await self.push_history_message(self.session_id, Message.tool_call_message(content, tool_calls))
                    await self.act(tool_calls)

                # 检查模型是否进行死循环
                if await self.is_stuck():
                    self.handle_stuck_state()

                # 继续下一步
                question = self.next_step_prompt

            # 检查终止原因并重置状态
            if self.current_step >= self.max_steps:
                content += f"\n\n Terminated: Reached max steps ({self.max_steps})"

            # 统一重置状态
            # self.reset()
            return content, self.tool_results

        except Exception as e:
            # 发生错误时设置错误状态
            self.state = AgentState.ERROR
            await self.push_history_message(self.session_id, Message.assistant_message(f"Error in agent execution: {str(e)}"))
            return str(e), self.tool_results

    async def think(self, question: str) -> Tuple[str, bool]:
        """Think about the question"""
        # 获取当前会话历史
        history = await self.get_history_context(self.session_id)
        llm = llm_factory.create_llm_instance(
            self.llm_provider, self.llm_model,
            model_id=self.model_id, session_id=self.session_id
        )

        response = None
        tool_calls = []
        try:
            if self.tool_choices == ToolChoice.NONE:
                response = await llm.chat(
                    self.system_prompt,
                    self.user_prompt,
                    question,
                    history,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                if not response.success:
                    raise Exception(response.content)
            else:
                response = await llm.ask_tools(
                    self.system_prompt,
                    self.user_prompt,
                    question,
                    history,
                    self.available_tools.to_params(),
                    self.tool_choices.value,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                # 处理工具调用
                if response.tool_calls:
                    # 处理工具调用列表
                    for i, tool_info in enumerate(response.tool_calls):
                        tool_call = ToolCall(
                            id=tool_info.id,
                            function=Function(
                                name=tool_info.name,
                                arguments=json.dumps(tool_info.args, ensure_ascii=False)
                            )
                        )
                        tool_calls.append(tool_call)

                # 结果信息打印
                logging.info(f"{self.agent_name}'s thoughts: {response.content}")
                logging.info(f"{self.agent_name} selected {len(tool_calls)} tools to use")

                if not tool_calls and self.tool_choices == ToolChoice.REQUIRED:
                    raise ValueError("Tool calls required but none provided")

            return response.content, tool_calls

        except Exception as e:
            logging.error(f"Error in {self.agent_name}'s thinking process: {str(e)}")
            raise RuntimeError(str(e))

    async def act(self, tool_calls: List[ToolCall]) -> None:
        """Execute tool calls and handle their results"""
        try:
            for toolcall in tool_calls:
                # 执行工具
                result = await self.execute_tool(toolcall)
                # 记录工具执行历史
                await self.push_history_message(self.session_id, Message.tool_result_message(
                    result.result, toolcall.function.name, toolcall.id)
                )
                # 记录工具执行结果，用于返回用户
                self.tool_results.append(tool_result(
                    name=toolcall.function.name,
                    success=result.status == ToolResultStatus.EXECUTE_SUCCESS,
                    output=result.result or ""
                ))
                logging.info(f"Tool '{toolcall.function.name}' completed! Result: {result}")

        except Exception as e:
            logging.error(f"Error in {self.agent_name}'s act process: {str(e)}")
            raise RuntimeError(str(e))

    async def execute_tool(self, toolcall: ToolCall) -> ToolResult:
        """Execute a single tool call with robust error handling"""
        if not toolcall or not toolcall.function:
            raise ValueError("Invalid tool call format")

        name = toolcall.function.name
        if not self.available_tools.get_tool(name):
            raise ValueError(f"Unknown tool '{name}'")

        try:
            # Parse arguments
            args = json.loads(toolcall.function.arguments or "{}")
            tool_result = await self.available_tools.execute(tool_name=name, tool_params=args)
            return tool_result

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON arguments for tool '{name}'")
            raise ValueError(f"Invalid JSON arguments for tool '{name}'")
        except Exception as e:
            logging.error(f"Tool({name}) execution error: {str(e)}")
            raise RuntimeError(f"Tool({name}) execution error: {str(e)}")

    def _has_special_tool(self, tool_calls: Optional[List[ToolCall]]) -> bool:
        """检查 tool_calls 中是否包含特殊工具"""
        if not self.special_tool_names or not tool_calls:
            return False
        special = [n.lower() for n in self.special_tool_names]
        return any(tc.function.name.lower() in special for tc in tool_calls)

    def get_available_tools(self) -> List[str]:
        """Get available tools list

        Returns:
            List[str]: List of available tools
        """
        return list(self.available_tools.keys())
