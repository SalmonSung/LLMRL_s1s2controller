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
        writer_llm = init_chat_model(
            model=writer_model,
            max_tokens=20000,
            thinking={"type": "disabled", "budget_tokens": 16000}
        )
    else:
        writer_llm = init_chat_model(model=writer_model)

    structured_llm = writer_llm.with_structured_output(S2AgentFormat)

    full_prompt = s2Agent_prompt.strip() + f"\n\nQuestion: {state['task'].strip()}"

    output = structured_llm.invoke(full_prompt)

    return {"answer": output.answer}

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
    pass_or_fail = "1" if output.pass_or_fail.strip().lower() == "pass" else "0"
    return {"pass_or_fail": pass_or_fail}


s2_agent_builder = StateGraph(S2State, input=S2Input, output=S2Output)
s2_agent_builder.add_node("s2_agent", s2_agent)
s2_agent_builder.add_node("grade_answer", grade_answer)

s2_agent_builder.add_edge(START, "s2_agent")
s2_agent_builder.add_edge( "s2_agent", "grade_answer")
s2_agent_builder.add_edge("grade_answer", END)