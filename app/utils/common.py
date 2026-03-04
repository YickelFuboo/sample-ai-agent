import os
from pathlib import Path
import tomllib
from typing import Any, Dict, List

def get_project_meta(package_name: str = "knowledge-service"):
    """从 pyproject.toml 读取项目元数据"""
    toml_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if not toml_path.exists():
        return {
            "name": "unknown-project",
            "version": "",
            "description": "",
        }

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    poetry = data.get("tool", {}).get("poetry", {})
    return {
        "name": poetry.get("name", "unknown-project"),
        "version": poetry.get("version", "0.0.0"),
        "description": poetry.get("description", ""),
    }

def get_project_base_directory():
    # 通过查找包含pyproject.toml的目录来确定项目根目录
    current_dir = os.path.dirname(__file__)

    project_root = current_dir
    while project_root != os.path.dirname(project_root):  # 直到到达文件系统根目录
        if os.path.exists(os.path.join(project_root, "pyproject.toml")):
            break
        project_root = os.path.dirname(project_root)

    return project_root


def _format_value(v: Any) -> str:
    """递归格式化值：若为 Dict 则呈现为 (key:value, key:value)。"""
    if isinstance(v, dict):
        inner = ", ".join(f"{k}: {_format_value(val)}" for k, val in v.items())
        return f"({inner})"
    return str(v)


def dict_to_str(d: Dict[str, Any]) -> str:
    """将 Dict 转为一行字符串，格式为 key:value, key:value；若 value 为 Dict 则 value 呈 key:value, key:(key:value, ...)。"""
    return ", ".join(f"{k}: {_format_value(v)}" for k, v in d.items())


def dict_to_str_by_keys(d: Dict[str, Any], keys: List[str]) -> str:
    """仅取指定 key 列表中的数据，转为一行字符串，格式同 dict_to_str。"""
    return ", ".join(f"{k}: {_format_value(d[k])}" for k in keys if k in d)


def dict_list_to_str(lst: List[Dict[str, Any]]) -> str:
    """将 Dict 列表转为字符串，每个 Dict 一行，格式同 dict_to_str，行间用换行分割。"""
    return "\n".join(dict_to_str(d) for d in lst if isinstance(d, dict))


def dict_list_to_str_by_keys(lst: List[Dict[str, Any]], keys: List[str]) -> str:
    """将 Dict 列表按指定 key 转为字符串，每个 Dict 一行，行间用换行分割。"""
    return "\n".join(dict_to_str_by_keys(d, keys) for d in lst if isinstance(d, dict))