from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field


class S2Input(BaseModel):
    task: str = Field(description="Question that needs to be answered")
    solution: str = Field(description="The default answer")


class S2Output(BaseModel):
    task: str
    answer: str
    solution: str
    pass_or_fail: str


class S2AgentFormat(BaseModel):
    task: str = Field(description="Question that needs to be answered")
    answer: str = Field(description="The answer")


class GradeAnswerFormat(BaseModel):
    answer: str = Field(description="The answer from the user")
    solution: str = Field(description="The default answer")
    pass_or_fail: str = Field(description="Whether user's answer is correct")


class S2State(TypedDict):
    task: str
    answer: str
    solution: str
    pass_or_fail: Literal['pass', 'fail']
