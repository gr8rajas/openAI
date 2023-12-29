from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
import os

# Few-shot learning is a remarkable ability that allows LLMs to learn and generalize from limited examples.
# This approach involves using the FewShotPromptTemplate class, which takes in a PromptTemplate and a list of a few shot examples
# create our examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

os.environ["OPENAI_API_KEY"] = "sk"
# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
op = chain.run("What's the meaning of life?")
print(op)