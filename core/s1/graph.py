from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig

from core.s1.state import *
from core.s1.prompts import s1Agent_prompt
from core.configuration import Configuration
from core.s2.graph import grade_answer

def s1_agent(state: S1State, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = configurable.writer_model
    if writer_model == "claude-3-7-sonnet-latest":
        writer_llm = init_chat_model(
            model=writer_model,
            max_tokens=20000,
            thinking={"type": "disabled", "budget_tokens": 16000},
        )
    else:
        writer_llm = init_chat_model(model=writer_model)

    structured_llm = writer_llm.with_structured_output(S1AgentFormat)

    full_prompt = (
        s1Agent_prompt.strip()
        + f"\n\nQuestion: {state['task'].strip()}"
    )

    output = structured_llm.invoke(full_prompt)

    return{"answer": output.answer}



s1_agent_builder = StateGraph(S1State, input=S1Input, output=S1Output, config_schema=Configuration)
s1_agent_builder.add_node("s1_agent", s1_agent)
s1_agent_builder.add_node("grade_answer", grade_answer)

s1_agent_builder.add_edge(START, "s1_agent")
s1_agent_builder.add_edge("s1_agent", "grade_answer")
s1_agent_builder.add_edge("grade_answer", END)