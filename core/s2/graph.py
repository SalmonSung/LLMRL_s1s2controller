from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig

from core.s2.state import *
from core.s2.prompts import *
from core.s2.tool import tools
from core.configuration import Configuration


def s2_agent(state: S2State, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = configurable.writer_model
    if writer_model == "claude-3-7-sonnet-latest":
        writer_llm = init_chat_model(model=writer_model,
                                     max_tokens=20_000,
                                     thinking={"type": "disabled", "budget_tokens": 16_000})
    else:
        writer_llm = init_chat_model(model=writer_model)
    agent = create_react_agent(model=writer_llm, tools=tools, prompt=s2Agent_prompt, response_format=S2AgentFormat)
    inputs = {"messages": state["task"]}
    for output in agent.stream(inputs, stream_mode="updates"):
        if output.get("structured_response"):
            return {"answer": output["structured_response"].answer}
            # return Command(goto=END, update={"answer": output["structured_response"].answer})

def grade_answer(state: S2State, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = configurable.writer_model
    if writer_model == "claude-3-7-sonnet-latest":
        writer_llm = init_chat_model(model=writer_model,
                                     max_tokens=20_000,
                                     thinking={"type": "disabled", "budget_tokens": 16_000}
                                    )
    else:
        writer_llm = init_chat_model(model=writer_model)
    structured_llm = writer_llm.with_structured_output(GradeAnswerFormat)
    system_prompt = grade_answer_prompt.format(default_solution=state["solution"])
    output = structured_llm.invoke([AIMessage(content=system_prompt), HumanMessage(content=state["answer"])])
    return {"pass_or_fail": output.pass_or_fail}

s2_agent_builder = StateGraph(S2State, input=S2Input, output=S2Output)
s2_agent_builder.add_node("s2_agent", s2_agent)
s2_agent_builder.add_node("grade_answer", grade_answer)

s2_agent_builder.add_edge(START, "s2_agent")
s2_agent_builder.add_edge( "s2_agent", "grade_answer")
s2_agent_builder.add_edge("grade_answer", END)