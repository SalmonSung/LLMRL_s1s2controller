from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from core.s2.graph import s2_agent_builder
from core.s1.graph import s1_agent_builder
from core.configuration import Configuration
from core.state import *

def controller_front(state: SectionState, config: RunnableConfig):

    return state

def router(state: SectionState, config: RunnableConfig) -> Command[Literal["s2_agent", "s1_agent"]]:
    configurable = Configuration.from_runnable_config(config)
    mode = configurable.pipelineMode
    if mode == "s2":
        return Command(goto="s2_agent", update={"task": state["task"], "solution": state["solution"]})
    else:
        return Command(goto="s1_agent", update={"task": state["task"], "solution": state["solution"]})

def controller_rear(state: SectionState, config: RunnableConfig):
    #PlaceHolder for further development
    return state


builder = StateGraph(SectionState, input=SectionInput, output=SectionOutput, config_schema=Configuration)
builder.add_node("controller_front", controller_front)
builder.add_node("router", router)
builder.add_node("s2_agent", s2_agent_builder.compile())
builder.add_node("s1_agent", s1_agent_builder.compile())
builder.add_node("controller_rear", controller_rear)

builder.add_edge(START, "controller_front")
builder.add_edge("controller_front", "router")
builder.add_edge("router", "controller_rear")
builder.add_edge("controller_rear", END)

graph = builder.compile()
