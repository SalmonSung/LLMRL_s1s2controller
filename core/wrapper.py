from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from core.configuration import PipelineMode
from core.main_graph import builder
import uuid

load_dotenv()

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# thread = {"configurable": {"thread_id": str(uuid.uuid4()),
#                            "writer_provider": "openai",
#                            "writer_model": "gpt-4o",
#                             "pipelineMode": "s1"
#                            }}

def nlp_wrapper(task: str, solution: str, pipelineMode: str = "s1") -> dict:
    thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                               "writer_provider": "openai",
                               "writer_model": "gpt-4o",
                                "pipelineMode": pipelineMode
                               }}
    event = {"task": task, "solution": solution}
    s_lists = []
    step = 0
    for s in graph.stream(event, thread, stream_mode="updates"):
        print(f"====step: {step}=====")
        print(s)
        s_lists.append(s)
        step += 1
        if '__interrupt__' in s:
            interrupt_value = s['__interrupt__'][0].value
            # display(Markdown(interrupt_value))
    # print(type(s_lists[-1]["generate_arg_for_def_collect"]))
    # print(s_lists[-1]["generate_arg_for_def_collect"])
    print(s_lists[-1])
    # print(s_lists[-1]["generate_arg_for_def_collect"]["tp"])
    if pipelineMode == "s2":
        return s_lists[-1]["s2_agent"]
    else:
        return s_lists[-1]["s1_agent"]

if __name__ == "__main__":
    output = nlp_wrapper("A pizza and a toy together cost $13. The pizza costs $10 more than the toy. How much does the toy cost?", "1.5")
    # command_line_format = ""
    # for key, item in output.items():
    #     if item is None:
    #         continue
    #     command_line_format += f"-{key} {item} "
    print("=====FINAL OUTPUT=====")
    print(output)