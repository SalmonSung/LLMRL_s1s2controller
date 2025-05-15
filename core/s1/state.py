from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field

class S1Input(BaseModel):
    task: str = Field(description="Question that needs to be answered")
    solution: str = Field(description="The default answer")

class S1Output(BaseModel):
    answer: str
    pass_or_fail: str

class S1AgentFormat(BaseModel):
    task: str = Field(description="Question that needs to be answered")
    answer: str = Field(description="The answer")

class S1State(TypedDict):
    task: str
    answer: str
    solution: str
    pass_or_fail: Literal['pass', 'fail']