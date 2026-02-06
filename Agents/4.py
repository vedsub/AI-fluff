## Evaluator and Optimiser with feedback loop

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

client = MongoClient(uri, server_api=ServerApi('1'))


from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

list_of_complaints = [...]
category_of_complaints = [...]

main_prompt = f"""
You are a customer service agent...
"""
classification_response = llm.invoke(main_prompt)
category = classification_response.content.strip("```xml\n<").strip("</category>").strip("\n")

evaluator_prompt = f"""
Context-
The previous LLM call has classified...
"""
evaluation_response = llm.invoke(evaluator_prompt)
feedback = evaluation_response.content

if feedback == "False":
    complaints_collection = db['complaints_unclassified']
else:
    complaints_collection = db['complaints']