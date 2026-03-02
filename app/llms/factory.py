import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import logging
from app.utils.common import get_project_base_directory
from .base import LLM
from .models import LLMInfo
from .schemes import ChatResponse, AskToolResponse
from .openaillm import OpenAIStyleLLM
from .anthropicllm import AnthropicStyleLLM
from .siliconllm import SiliconStyleLLM

_LLM_MODELS_JSON = Path(get_project_base_directory()) / "llm_models.json"

def _load_llm_configs_from_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """从 llm_models.json 加载并扁平化为 name -> config。"""
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models") or {}
    result: Dict[str, Dict[str, Any]] = {}
    for provider, prov_cfg in models.items():
        base_url = prov_cfg.get("base_url", "")
        api_key = prov_cfg.get("api_key", "")
        api_style = prov_cfg.get("api_style", "")
        instances = prov_cfg.get("instances") or {}
        for instance_name, inst_cfg in instances.items():
            name = f"{provider}/{instance_name}"
            result[name] = {
                "provider": provider,
                "model_name": instance_name,
                "api_style": api_style,
                "api_base": base_url,
                "api_key": api_key,
                "description": (inst_cfg or {}).get("description", ""),
                "max_tokens": 4096,
                "temperature": 0.7,
            }
    return result

def _load_default_llm_from_json(path: Path) -> tuple:
    """从 llm_models.json 加载默认 LLM，返回 (provider, model_name)。"""
    if not path.is_file():
        return "", ""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    default = data.get("default") or {}
    provider = default.get("provider", "")
    model_name = default.get("model", "") or default.get("model_name", "")
    return provider, model_name

class LLMFactory:
    """LLM工厂类"""

    API_STYLES = {
        "openai": OpenAIStyleLLM,
        "anthropic": AnthropicStyleLLM,
        "silicon": SiliconStyleLLM,
    }

    def __init__(self):
        self.llmconfigs = _load_llm_configs_from_json(_LLM_MODELS_JSON)
        self.default_provider, self.default_model_name = _load_default_llm_from_json(_LLM_MODELS_JSON)

    # 获取指定模型的指定参数值
    def get_llm_config(self, provider: str, model_name: str, param_name: str) -> Any:
        """获取指定模型的指定参数值"""
        return self.llmconfigs[f"{provider}/{model_name}"][param_name]

    def _validate_llm(self, provider: str, model_name: str) -> Dict[str, Any]:
        """验证LLM是否存在并返回配置"""
        full_name = f"{provider}/{model_name}"
        if full_name not in self.llmconfigs:
            raise ValueError(f"未知的LLM: {provider}/{model_name}")

        config = self.llmconfigs[full_name]
        api_style = config["api_style"]

        if api_style not in self.API_STYLES:
            raise ValueError(f"不支持的API风格: {api_style}")

        return config

    def create_llm_instance(self, provider: str, model_name: str, model_id: str = "", session_id: str = "", *kwargs: Any) -> LLM:
        """创建LLM实例"""
        if not provider or not model_name or f"{provider}/{model_name}" not in self.llmconfigs:
            if not self.default_provider or not self.default_model_name:
                raise ValueError("未配置默认LLM")
            active_provider = self.default_provider
            active_model_name = self.default_model_name
        else:
            active_provider = provider
            active_model_name = model_name

        full_name = f"{active_provider}/{active_model_name}"
        if full_name not in self.llmconfigs:
            raise ValueError(f"未知的LLM: {full_name}")

        config = self.llmconfigs[full_name]
        params = {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
        }
        params.update(kwargs)

        return self.API_STYLES[config["api_style"]](
            model_name=active_model_name,
            model_type=config["api_style"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_id=model_id, # 大赛使用
            session_id=session_id, # 大赛使用
            **params
        )

    def get_supported_llms(self) -> List[LLMInfo]:
        """获取支持的模型列表"""
        llm_list = []
        for name, llm in self.llmconfigs.items():
            llm_list.append(LLMInfo(
                provider=llm["provider"],
                model_name=llm["model_name"],
                type=llm["api_style"],
                description=llm.get("description") or "",
                max_tokens=llm.get("max_tokens", 4096),
            ))
        return llm_list


# 全局工厂实例
llm_factory = LLMFactory()