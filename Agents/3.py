from concurrent.futures import ThreadPoolExecutor #3 worker threads created
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