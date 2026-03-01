from fastapi import APIRouter
from typing import List
from .models import LLMInfo
from .factory import llm_factory


router = APIRouter()

@router.get("/available", 
    summary="获取支持的模型列表",
    response_model=List[LLMInfo])
async def get_supported_models():
    """获取所有支持的LLM模型列表"""
    return llm_factory.get_supported_llms() 