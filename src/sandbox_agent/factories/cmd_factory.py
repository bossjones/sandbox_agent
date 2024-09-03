"""sandbox_agent.factories.cmd_factory."""

# pylint: disable=no-value-for-parameter
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from pydantic import ValidationError, validate_call

from sandbox_agent.factories import SerializerFactory


# SOURCE: https://stackoverflow.com/questions/54863458/force-type-conversion-in-python-dataclass-init-method
@validate_call
@dataclass
class CmdSerializer(SerializerFactory):
    name: str
    cmd: Optional[str]
    uri: Optional[str]

    @staticmethod
    def create(d: dict) -> CmdSerializer:
        return CmdSerializer(name=d["name"])


# SOURCE: https://stackoverflow.com/questions/54863458/force-type-conversion-in-python-dataclass-init-method
