from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

class SectionInput(BaseModel):
    task: str = Field(description="Question that needs to be answered")
    solution: str = Field(description="The default answer")

class SectionOutput(BaseModel):
    task: str
    answer: str
    solution: str
    pass_or_fail: str

class SectionState(TypedDict):
    task: str
    answer: str
    solution: str
    pass_or_fail: Literal['1', '0']
