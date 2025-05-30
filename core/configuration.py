import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass



class PlannerProvider(Enum):
    OPENAI = "openai"



class WriterProvider(Enum):
    OPENAI = "openai"



class FilterProvider(Enum):
    OPENAI = "openai"

class PipelineMode(Enum):
    S2 = "s2"
    S1 = "s1"

@dataclass(kw_only=True)
class Configuration:
    # planner_provider: PlannerProvider = PlannerProvider.ANTHROPIC
    # planner_model: str = "claude-3-7-sonnet-latest"
    planner_provider: PlannerProvider = PlannerProvider.OPENAI
    planner_model: str = "gpt-4.1-mini"
    # writer_provider: WriterProvider = WriterProvider.XAI
    # writer_model: str = "grok-2-latest"
    writer_provider: WriterProvider = WriterProvider.OPENAI
    writer_model: str = "gpt-4.1-mini"
    filter_provider: FilterProvider = FilterProvider.OPENAI
    filter_model: str = "gpt-4.1-mini"
    pipelineMode: PipelineMode = PipelineMode.S2

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})