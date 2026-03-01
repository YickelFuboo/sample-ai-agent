from pydantic import BaseModel


class LLMInfo(BaseModel):
    """LLM模型信息"""
    provider: str  # 提供者
    model_name: str  # 模型名称
    type: str  # 模型类型(如 openai/anthropic/silicon)
    description: str  # 模型描述
    max_tokens: int  # 最大token数