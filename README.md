# LLMRL_s1s2controller  
This project is dedicated to developing a controller that can immediately decide whether to use the s1 or s2 system to answer a query upon receiving it.  
![image](https://github.com/user-attachments/assets/6e1fda48-a561-4852-8847-dcb2d862a096)
## Before You Start
1. create ```.env``` file from ```.env.example```   
2. ```pip install -r requirements.txt```
## FAQ  
**1. How to use langgraph studio?(Interface to visualise the process)**  
terminal: ```langgraph dev```  
This will runn a server locally and automatically pop up a web.  
**2. How to use this in my project?**  
```Python
from wrapper import nlp_wrapper

def nlp_wrapper(task: str, solution: str, pipelineMode: str = "s1"):
    return <some_dict>
```
Rather than Langgraph Studio, you can call this method and collect the data more easily.  
**3. What is the algorithm used to eliminate the difference between agent's output and actual solution?**  
I discovered the database and realise the data types are too mess up. Eventually I designed a state as follows:  
```Python
class S2State(TypedDict):
    task: str
    answer: str
    solution: str
    pass_or_fail: Literal['pass', 'fail']
```
After the agent return the answer, another LLM will judge whether the answer is aligned with solution.  
## RoadMap  
1. Finish ```def s2_agent_wrapper()```
2. will probably start to prepare TextGradient System  
