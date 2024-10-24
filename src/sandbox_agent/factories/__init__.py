"""Factories for the sandbox agent."""

# Import factories from other modules as needed
from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sandbox_agent.ai.chat_models import ChatModelFactory
from sandbox_agent.ai.document_loaders import DocumentLoaderFactory
from sandbox_agent.ai.embedding_models import EmbeddingModelFactory
from sandbox_agent.ai.evaluators import EvaluatorFactory
from sandbox_agent.ai.key_value_stores import KeyValueStoreFactory
from sandbox_agent.ai.memory import MemoryFactory
from sandbox_agent.ai.retrievers import RetrieverFactory
from sandbox_agent.ai.text_splitters import TextSplitterFactory
from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.ai.vector_stores import VectorStoreFactory


READ_ONLY = "read_only"
COERCE_TO = "coerce_to"


@dataclass
class SerializerFactory:
    def as_dict(self) -> dict:
        d: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if READ_ONLY not in f.metadata:
                if COERCE_TO in f.metadata:
                    value = f.metadata[COERCE_TO](value)
                d[f.name] = value
        return d
