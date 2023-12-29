from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
# Token Key Tracking
os.environ["OPENAI_API_KEY"] = "sk"
llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)


# Tokens Used: 44
# 	Prompt Tokens: 4
# 	Completion Tokens: 40
# Successful Requests: 1
# Total Cost (USD): $0.00088