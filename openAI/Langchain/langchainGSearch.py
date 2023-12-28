
import os
from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper


os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDR5TrAvDaQvmdKhGxAaUNzRX4"
os.environ["GOOGLE_CSE_ID"] = "a74f2e92"
llm = OpenAI(model="text-davinci-003", temperature=0)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        # The "zero-shot-react-description" type of an Agent uses the ReAct framework to decide which tool to use based only on the tool's description.
                         verbose=True,
                         max_iterations=6)


response = agent("What's the latest news about Donald Trump?")
print(response['output'])