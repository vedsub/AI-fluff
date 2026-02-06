from concurrent.futures import ThreadPoolExecutor

from asyncio import tools

from litellm import query #3 worker threads created
# executor.map(process_message, a):
# # Submits the process_message function on each element of list a.

#### Routing

#Breakdown of Routing Code
#input_question = "My car is damaged and I want to buy a car insurance policy?"
#The routing logic is encoded in the selection_prompt using a list of predefined categories:

#options = ['Life Insurance','Health Insurance','Car Insurance','Home Insurance','General Insurance']


#Extracting Sturctured Data

def extract_option(result):
  return result.content.split("<answer>")[1].split("</answer>")[0].strip()

@tool
def bmi_calculator(weight:float , height:float) -> str:
  return

llm_with_tool = llm.bind_tools(tools)


message = [HumanMessage(content=query)]
message.append(ToolMessage(content = tool_output , tool_call_id = i['id']))


query = "What is the BMI for a person weighing 70 kg and 1.75 meters tall?"
ai_message = llm_with_tool.invoke(message)

for i in ai_message.tool_calls:
  selected_tool = {"bmi_calculator" : bmi_calculator }[i['name'].lower()]
  tool_output = selected_tool.invoke(i['args'])
  message.append(ToolMessage(content = tool_output , tool_call_id = i['id']))
  
llm_with_tool.invoke(message)

functions = [
    {
        "name": "bmi_calculator",
        "description": "Calculate the BMI category",
        "parameters": {
            "type": "object",
            "properties": {
                "weight": {"type": "number"},
                "height": {"type": "number"}
            },
            "required": ["weight", "height"]
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[...],
    functions=functions,
    function_call="auto"
)