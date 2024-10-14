"""Factories for the sandbox agent."""

# Import factories from other modules as needed
from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from sandbox_agent.ai.chat_models import ChatModelFactory
from sandbox_agent.ai.document_loaders import DocumentLoaderFactory
from sandbox_agent.ai.embedding_models import EmbeddingModelFactory
from sandbox_agent.ai.key_value_stores import KeyValueStoreFactory
from sandbox_agent.ai.retrievers import RetrieverFactory
from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.ai.vector_stores import VectorStoreFactory


READ_ONLY = "read_only"
COERCE_TO = "coerce_to"


@dataclass
class SerializerFactory:
    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for f in dataclasses.fields(self):
            if READ_ONLY in f.metadata.keys():
                del d[f.name]
            if COERCE_TO in f.metadata.keys():
                d[f.name] = f.metadata[COERCE_TO](d[f.name])
        return d
