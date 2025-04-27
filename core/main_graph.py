from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

from core.s2.graph import s2_agent_builder
from core.configuration import Configuration
from core.state import *

def controller_front(state: SectionState, config: RunnableConfig):
    #Placeholder for further development
    return state

def router(state: SectionState, config: RunnableConfig) -> Command[Literal["s2_agent"]]:
    return Command(goto="s2_agent", update={"task": state["task"], "solution": state["solution"]})

def controller_rear(state: SectionState, config: RunnableConfig):
    #PlaceHolder for further development
    return state


builder = StateGraph(SectionState, input=SectionInput, output=SectionOutput, config_schema=Configuration)
builder.add_node("controller_front", controller_front)
builder.add_node("router", router)
builder.add_node("s2_agent", s2_agent_builder.compile())
builder.add_node("controller_rear", controller_rear)

builder.add_edge(START, "controller_front")
builder.add_edge("controller_front", "router")
builder.add_edge("router", "controller_rear")
builder.add_edge("controller_rear", END)

graph = builder.compile()
