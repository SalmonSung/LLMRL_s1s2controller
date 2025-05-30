from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import time
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
    start_time = time.time()

    thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                               "writer_provider": "openai",
                               "writer_model": "gpt-4.1-mini",
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
    end_time = time.time()
    time_token = round(end_time-start_time,5)
    print(f"cost time is: {time_token} s")
    if pipelineMode == "s2":
        return {
            "s2_agent": s_lists[-1]["s2_agent"],
            "time_token":time_token
        }
    else:
        return {
            "s1_agent": s_lists[-1]["s1_agent"],
            "time_token":time_token
        }
if __name__ == "__main__":
    output = nlp_wrapper("A pizza and a toy together cost $13. The pizza costs $10 more than the toy. How much does the toy cost?", "1.5")
    # command_line_format = ""
    # for key, item in output.items():
    #     if item is None:
    #         continue
    #     command_line_format += f"-{key} {item} "
    print("=====FINAL OUTPUT=====")
    print(output)