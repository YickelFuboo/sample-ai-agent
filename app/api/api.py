from fastapi import APIRouter, HTTPException
import logging
import time
from app.schemes.schemes import UserRequest, UserResponse
from app.agent.reactagent import ReActAgent
from app.sessions.manager import SESSION_MANAGER


router = APIRouter()


@router.post("/chat", response_model=UserResponse)
async def chat(request: UserRequest):
    """聊天接口"""
    try:
        start_time = time.time()
        
        # 检查session_id有效性，如果不存在则创建会话
        session_id = request.session_id
        if session_id is None or session_id == "" or not await SESSION_MANAGER.get_session(session_id):
            session_id = await SESSION_MANAGER.create_session(session_id=session_id)

        # 获取会话
        session = await SESSION_MANAGER.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=400, detail="Session not found")

        agent = ReActAgent(
            agent_name="ReActAgent", 
            agent_description="A ReAct agent", 
            agent_type="default",
            channel_type="",
            channel_id=session_id,
            session_id=session_id, 
            workspace_index=session_id,
            user_id="",
            model_id=request.model_id,
        )

        # 运行Agent
        respone, tool_results = await agent.run(request.message)

        end_time = time.time()
        timestamp = int(end_time)
        duration_ms = int((end_time - start_time) * 1000)

        return UserResponse(session_id=session_id, resopne=respone, tool_results=tool_results, timestamp=timestamp, duration_ms=duration_ms)
    
    except Exception as e:
        logging.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
