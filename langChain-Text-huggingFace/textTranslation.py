import os
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "l"

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

source_language = "English"
target_language = "Telugu"
text = "My Name is RAJA"
translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)
print(translated_text)


# https://colab.research.google.com/drive/1DRpBmq6i8fREqIWItHdpjECdYoPWNrIP?usp=sharing