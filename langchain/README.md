# In LangChain, agents are high-level components that use language models (LLMs) to determine which actions to take and in what order.

The zero-shot-react-description agent uses the ReAct framework to decide which tool to employ based purely on the tool's description. It necessitates a description of each tool.

The react-docstore agent engages with a docstore through the ReAct framework. It needs two tools: a Search tool and a Lookup tool. The Search tool finds a document, and the Lookup tool searches for a term in the most recently discovered document.

The self-ask-with-search agent employs a single tool named Intermediate Answer, which is capable of looking up factual responses to queries. It is identical to the original self-ask with the search paper, where a Google search API was provided as the tool.

The conversational-react-description agent is designed for conversational situations. It uses the ReAct framework to select a tool and uses memory to remember past conversation interactions.

# LangChain provides an expansive toolkit that integrates various functions to improve the functionality of conversational agents. Here are some examples:

SerpAPI: This tool is an interface for the SerpAPI search engine, allowing the agent to perform robust online searches to pull in relevant data for a conversation or task.

PythonREPLTool: This unique tool enables the writing and execution of Python code within an agent. This opens up a wide range of possibilities for advanced computations and interactions within the conversation.# openAI
