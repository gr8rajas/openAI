import os
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Initialize language model
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="/Users/rajasekharkalamata/PycharmProjects/openAI/LangChain/langChainPDF/BlueLabs.pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])

# Airflow is an option for data transfer from SFTP to S3 Bucket/ADLS Container, with the ability to apply
# business rules/transformations in tasks and write results into a reporting database.
# Slack can be integrated into all DAGs, and for transformations, Rivery/DBT can be used.
# When selecting a product to replace their Vertica architecture, scalability, cost-effectiveness, ease of migration, data security, integration, development, and customer/community support should be considered.
# To ensure a smooth cutover experience with minimal downtime, data should be replicated in the new system and updated in parallel, a rollback plan should be in place,
# stakeholders should be informed, and metrics should be monitored.




################################# Prompt use case ######################################

# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
#
# # Before executing the following code, make sure to have
# # your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
# chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#
# template = "You are an assistant that helps users find information about movies."
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template = "Find information about the movie {movie_title}."
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
#
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#
# response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())
#
# print(response.content)



# QA chain example:
# We can also use LangChain to manage prompts for asking general questions from the LLMs.
# These models are proficient in addressing fundamental inquiries. Nevertheless, it is crucial to remain mindful of the potential issue of hallucinations, where the models may generate non-factual information
