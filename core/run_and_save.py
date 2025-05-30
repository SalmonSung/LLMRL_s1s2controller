from wrapper import nlp_wrapper
from dataset.crt import crt3_not_hostile
import json

results = []

for item in crt3_not_hostile:
    task = item["task"]
    solution = item["correct"]
    number = item.get("number", "?")

    print(f"TASK #{number}")
    output = nlp_wrapper(task, solution, pipelineMode="s2")


    results.append({
        "task": task,
        "result": output,
    })

# 保存结果为 JSON
with open("s2_result_3/results_crt3_not_hostile.csv", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("All tasks are finished")