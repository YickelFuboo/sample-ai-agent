"""会话存储：本地文件存储。"""
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from .message import Message
from .session import Session
from app.config.settings import settings


class LocalFileSessionStore(ABC):
    """本地文件存储：目录下 {session_id}.json，用 _cache 存 load 结果。"""

    def __init__(self) -> None:
        self.storage_dir = settings.agent_session_storage_dir
        self._cache: Dict[str, Session] = {}

    def _load_one(self, session_id: str) -> Optional[Session]:
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session(**data)
        except Exception as e:
            logging.error("Error loading session %s: %s", session_id, e)
            return None

    async def get(self, session_id: str) -> Optional[Session]:
        if session_id in self._cache:
            return self._cache[session_id]
        data = await asyncio.to_thread(self._load_one, session_id)
        if data:
            self._cache[session_id] = data
        return data

    async def save(self, session: Session) -> None:
        path = os.path.join(self.storage_dir, f"{session.session_id}.json")
        try:
            data = session.model_dump()

            def write_file():
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            await asyncio.to_thread(write_file)
            self._cache[session.session_id] = session
        except Exception as e:
            logging.error("Error saving session %s: %s", session.session_id, e)

    async def delete(self, session_id: str) -> bool:
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        self._cache.pop(session_id, None)
        if not os.path.isfile(path):
            return False
        try:
            await asyncio.to_thread(os.remove, path)
            logging.info("Deleted session file: %s", session_id)
            return True
        except Exception as e:
            logging.error("Error deleting session %s: %s", session_id, e)
            return False

    async def get_all(self) -> List[Session]:
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
            logging.info("Created sessions directory: %s", self.storage_dir)
            return []
        self._cache.clear()
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue
            session_id = filename[:-5]
            path = os.path.join(self.storage_dir, filename)
            try:
                def read_file():
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                data = await asyncio.to_thread(read_file)
                self._cache[session_id] = Session(**data)
            except Exception as e:
                logging.error("Error loading session %s: %s", session_id, e)
        return list(self._cache.values())