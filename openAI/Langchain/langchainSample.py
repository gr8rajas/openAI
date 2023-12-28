# Before executing the following code, save your OpenAI key in the environment variable using the following key OPENAI_API_KEY

import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "sk"
llm = OpenAI(model="text-davinci-003", temperature=0.2)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
    # Used to Store Q & A in Memory
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)
