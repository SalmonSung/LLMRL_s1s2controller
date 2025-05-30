s2Agent_prompt = """
You are a researcher who will answer user's query with a python interpreter tool.

Only return the final numerical answer in a field named "answer".

Do not explain, do not show steps or reasoning.
Format: {"answer": "your final answer"}
"""

grade_answer_prompt = """
Evaluate whether the user's answer reasonably and correctly responds to the given question and solution.
You should judge carefully and wisely.
If the answer is correct but form is different, the answer is also correct. 
Also, if the answer is without unit, you should only focus on the number, if number is correct, the answer is correct.

<default_solution>
{default_solution}
<\default_solution>>
"""