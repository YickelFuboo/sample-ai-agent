"""长期记忆：将已拼装好的内容追加写入当前文件同目录下的 long_memory.md。"""
import logging
import os


HEADER = "#记忆\n\n\n"


def _get_long_memory_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_memory.md")


def append_shell_result(output: str) -> None:
    """将已拼装好的内容追加到 long_memory.md，文件头部为 #记忆 加两个空行。"""
    path = _get_long_memory_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(HEADER)
        with open(path, "a", encoding="utf-8") as f:
            f.write(output)
    except Exception as e:
        logging.warning("Failed to append to long_memory: %s", e)
