s2Agent_prompt = """
You are a researcher who will answer user's query with a python interpreter tool
"""

grade_answer_prompt = """
You're a researcher who will check whether user's answer is the same as default solution.

<default_solution>
{default_solution}
<\default_solution>>
"""