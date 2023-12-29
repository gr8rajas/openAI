from langchain import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "What is the capital city of France?"

############################################


from langchain import HuggingFaceHub, LLMChain
import os

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
print(llm_chain.run(question))


qa = [
    {'question': "What is the capital city of USA?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]
res = llm_chain.generate(qa)
print( res )

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = LLMChain(
    prompt=long_prompt,
    llm=llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)
print(llm_chain.run(qs_str))