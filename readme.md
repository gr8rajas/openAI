`Create a virtual environment using the command python -m venv my_venv_name.
Activate the virtual environment by executing source my_venv_name/bin/activate.
Install the required libraries and run the code snippets from the lessons within the virtual environment.

pip install langchain==0.0.208 
pip install deeplake==3.6.5
pip install openai==0.27.8
pip install tiktoken==0.4.0
pip install selenium==4.15.2


To deactivate the virtual environment, simply run deactivate.
`

LLM's Issue - Hallucination--(Use Retreiver to get data from Web to Mitigate)
Efficient retrievers are built using embedding models that map texts to vectors. These vectors are then stored in specialized databases called vector stores.

LLMs and Chat Models each have their advantages and disadvantages. LLMs are powerful and flexible, capable of generating text for a wide range of tasks. However, their API is less structured compared to Chat Models.

On the other hand, Chat Models offer a more structured API and are better suited for conversational tasks. Also, they can remember previous exchanges with the user, making them more suitable for engaging in meaningful conversations. Additionally, they benefit from reinforcement learning from human feedback, which helps improve their responses.


_The LangChain’s Chat API offers several advantages:

Context preservation: By maintaining a list of messages in the conversation, the API ensures that the context is preserved throughout the interaction. This allows the GPT-4 model to generate relevant and coherent responses based on the provided information.

Memory: The class’s message history acts as a short-term memory for the chatbot, allowing it to refer back to previous messages and provide more accurate and contextual responses.

Modularity: The combination of MessageTemplate and ChatOpenAI classes offers a modular approach to designing conversation applications. This makes it easier to develop, maintain, and extend the functionality of the chatbot.
Improved performance: GPT-4, as an advanced language model, is more adept at understanding complex prompts and generating better responses than its predecessors. 

Flexibility: The Chat API can be adapted to different domains and tasks, making it a versatile solution for various chatbot applications._ 

## Popular LLM models accessible to LangChain via API

### Cohere Command
The Cohere service provides a variety of models such as Command (command) for dialogue-like interactions, Generation (base) for generative tasks, Summarize (summarize-xlarge) for generating summaries, and more. You can get free, rate-limited usage for learning and prototyping. This means that usage is free until you go into production, however some of the models may be a bit more expensive than OpenAI APIs when you do—for example, $2.5 for generating 1K tokens. However, since Cohere offers more customized models for each task, this could lead to a more use case-specific model having improved outcomes in downstream tasks. The LangChain’s Cohere class makes it easy to access these models. Cohere(model="<MODEL_NAME>", cohere_api_key="<API_KEY>")
You might see deprecated model names in the LangChain documentation. (like command-xlarge-20221108) Please refer to the Cohere documentation for the latest naming convention.

### GPT-3.5
GPT-3.5 is a language model developed by OpenAI. Its turbo version (recommended by OpenAI over other variants) offers a more affordable option for generating human-like text through an API accessible via OpenAI endpoints. The model is optimized for chat applications while remaining powerful on other generative tasks and can process 96 languages. GPT-3.5-turbo has a up to 16K tokens context length and is the most cost-effective option from the OpenAI collection with only $0.002 per 1000 tokens. It is possible to access this model’s API by using the gpt-3.5-turbo key while initializing either ChatOpenAI or OpenAI classes.


### GPT-4
OpenAI's GPT-4 is a competent multimodal model with an undisclosed number of parameters or training procedures. It is the latest and most powerful model published by OpenAI, and the multi-modality enables the model to process both text and image as input. Unfortunately, It is not publicly available; however, it can be accessed by submitting your early access request through the OpenAI platform. The two variants of the model are gpt-4 and gpt-4-32k with different context lengths, 8192 and 32768 tokens, respectively.


### Jurassic-2
The AI21’s Jurassic-2 is a language model with three sizes and different price points: Jumbo, Grande, and Large. The model sizes are not publicly available, but their documentation marks the Jumbo version as the most powerful model. They describe the models as general-purpose with excellent capability on every generative task. Their J2 model understands seven languages and can be fine-tuned on custom datasets. Getting your API key from the AI21 platform and using the AI21()class to access these models is possible.


### StableLM
StableLM Alpha is a language model developed by Stable Diffusion, which can be accessed via HuggingFace Hub (with the following id stabilityai/stablelm-tuned-alpha-3b) to host locally or Replicate API with a rate from $0.0002 to $0.0023 per second. So far, it comes in two sizes, 3 billion and 7 billion parameters. The weights for StableLM Alpha are available under CC BY-SA 4.0 license with commercial use access. The context length of StableLM is 4096 tokens.


### Dolly-v2-12B
Dolly-v2-12B is a language model created by Databricks, which can be accessed via HuggingFace Hub (with the following id databricks/dolly-v2-3b) to host locally or Replicate API with the same price range as mentioned in the previous subsection. It has 12 billion parameters and is available under an open source license for commercial use. The base model used for Dolly-v2-12B is Pythia-12B. 


### GPT4ALL
GPT4ALL is based on meta’s LLaMA model with 7B parameters. It is a language model developed by Nomic-AI that can be accessed through GPT4ALL and Hugging Face Local Pipelines. The model is published with a GPL 3.0 open-source license. However, it is not free to use for commercial applications. It is available for researchers to use for their projects and experiments. We went through this model’s capability and usage process in the previous lesson.


## LLM Platforms that can integrate into LangChain

### Cohere

Cohere is a Canadian-based startup specializing in natural language processing models that help companies enhance human-machine interactions. Cohere provides access to their Cohere xlarge model through API, which has 52 billion parameters. Their API pricing is based on embeddings and is $1 for every 1000 embeddings. Cohere provides an easy-to-follow installation process for their package, which is required to access their API. Using LangChain, developers can easily interact with Cohere models by creating prompts incorporating input variables, which can then be passed to the Cohere API to generate responses. 


### OpenAI

OpenAI platform is one of the biggest companies focusing on large language models. By introducing their conversational model, ChatGPT, they were the first service to catch mainstream media attention on the potency of LLMs. They also provide a large variety of API endpoints for different NLP tasks with different price points. The LangChain library provides multiple classes for convenient access, examples of which we saw in previous lessons, like ChatGPT and GPT4 classes.


### Hugging Face Hub

Hugging Face is a company that develops natural language processing (NLP) technologies, including pre-trained language models, and offers a platform for developing and deploying NLP models. The platform hosts over 120k models and 20k datasets. They offer the Spaces service for researchers and developers to create a demo and showcase their model’s capabilities quickly. The platform hosts large-scale models such as StableLM by Stability AI, Dolly by DataBricks, or Camel by Writer. The HuggingFaceHub class takes care of downloading and initializing the models.

This integration provides access to many models that are optimized for Intel CPUs using Intel® Extension for PyTorch library. The mentioned package can be applied to models with minimal code change. It enables the networks to take advantage of Intel®’s advanced architectural designs to significantly enhance CPU and GPU lines' performance. For example, the reports reveal a 3.8 speed up while running the BLOOMZ model (text-to-image) on the Intel® Xeon® 4s CPU compared to the previous generation with no changes in architecture/weights. When the mentioned optimization library was used alongside the 4th generation of Intel® Xeon® CPU, the inference speed rate increased nearly twofold to 6.5 times its original value. (online demo) Whisper and GPT-J are two other examples of widely recognized models that leverage these efficiency gains.


### Amazon SageMakerEndpoint

The Amazon SageMaker infrastructure enables users to train and host their machine-learning models easily. It is a high-performance and low-cost environment for experimenting and using large-scale models. The LangChain library provides a simple-to-use interface that simplifies the process of querying the deployed models. So, There is no need to write API codes for accessing the model. It is possible to load a model by using the endpoint_name which is the model’s unique name from SageMaker, followed by credentials_profile_name which is the name of the profile you want to use for authentication.


### Hugging Face Local Pipelines

Hugging Face Local Pipelines is a powerful tool that allows users to run Hugging Face models locally using the HuggingFacePipeline class. The Hugging Face Model Hub is home to an impressive collection of more than 120,000 models, 20,000 datasets, and 50,000 demo apps (Spaces) that are all publicly available and open source, making it easy for individuals to collaborate and build machine learning models together. To access these models, users can either utilize the local pipeline wrapper or call the hosted inference endpoints via the HuggingFaceHub class. Before getting started, the Transformers Python package must be installed. Once installed, users can load their desired model using the model_id and task and any additional model arguments. Finally, the model can be integrated into an LLMChain by creating a PromptTemplate and LLMChain object and running the input through it.

### Azure OpenAI

OpenAI’s models can also be accessed via Microsoft’s Azure platform. 

### AI21

AI21 is a company that offers access to their powerful Jurassic-2 large language models through their API. The API provides access to their Jurassic-2 model, which has an impressive 178 billion parameters. The API comes at quite a reasonable cost of only $0.01 for every 1k tokens. Developers can easily interact with the AI21 models by creating prompts with LangChain that incorporate input variables. With this simple process, developers can take advantage of their powerful language processing capabilities.

### Aleph Alpha

Aleph Alpha is a company that offers a family of large language models known as the Luminous series. The Luminous family includes three models, namely Luminous-base, Luminous-extended, and Luminous-supreme, which vary in terms of complexity and capabilities. Aleph Alpha's pricing model is token-based, and the table provides the base prices per model for every 1000 input tokens. The Luminous-base model costs 0.03€ per 1000 input tokens, Luminous-extended costs 0.045€ per 1000 input tokens, Luminous-supreme costs 0.175€ per 1000 input tokens, and Luminous-supreme-control costs 0.21875€ per 1000 input tokens.

### Banana

Banana is a machine learning infrastructure-focused company that provides developers with the tools to build machine learning models. Using LangChain, one can interact with Banana models by installing the Banana package, including an SDK for Python. Next, two following tokens are required: the BANANA_API_KEY and the YOUR_MODEL_KEY, which can be obtained from their platform. After setting the keys, we can create an object by providing the YOUR_MODEL_KEY. It is then possible to integrate the Banana model into an LLMChain by creating a PromptTemplate and LLMChain object and running the desired input through it.

CerebriumAI

CerebriumAI is an excellent alternative to AWS Sagemaker, providing access to several LLM models through its API. The available pre-trained LLM models include Whisper, MT0, FlanT5, GPT-Neo, Roberta, Pygmalion, Tortoise, and GPT4All. Developers create an instance of CerebriumAI by providing the endpoint URL and other relevant parameters such as max length, temperature, etc.

### DeepInfra

DeepInfra is a unique API that offers a range of LLMs, such as distilbert-base-multilingual-cased, bert-base, whisper-large, gpt2, dolly-v2-12b, and more. It is connected to LangChain via API and runs on A100 GPUs that are optimized for inference performance and low latency. Compared to Replicate, DeepInfra's pricing is much more affordable, at $0.0005/second and $0.03/minute. With DeepInfra, we are given a 1-hour free trial of serverless GPU computing to experiment with different models.

### ForefrontAI

ForefrontAI is a platform that allows users to fine-tune and utilize various open-source large language models like GPT-J, GPT-NeoX, T5, and more. The platform offers different pricing plans, including the Starter plan for $29/month, which comes with 5 million serverless tokens, 5 fine-tuned models, 1 user, and Discord support. With ForefrontAI, developers have access to various models that can be fine-tuned to suit our specific needs.

### GooseAI

GooseAI is a fully managed NLP-as-a-Service platform that offers access to various models, including GPT-Neo, Fairseq, and GPT-J. The pricing for GooseAI is based on different model sizes and usage. For the 125M model, the base price for up to 25 tokens is $0.000035 per request, with an additional fee of $0.000001. To use GooseAI with LangChain, you need to install the openai package and set the Environment API Key, which can be obtained from GooseAI. Once you have the API key, you can create a GooseAI instance and define a Prompt Template for Question and Answer. The LLMChain can then be initiated, and you can provide a question to run the LLMChain.

### Llama-cpp

Llama-cpp, a Python binding for llama.cpp, has been seamlessly integrated into the LangChain framework. This integration allows users to access a variety of LLM (Large Language Model) models offered by Llama-cpp, including LLaMA 🦙, Alpaca, GPT4All, Chinese LLaMA / Alpaca, Vigogne (French), Vicuna, Koala, OpenBuddy 🐶 (Multilingual), Pygmalion 7B, and Metharme 7B. With this integration, users have a wide range of options to choose from based on their specific language processing needs. By integrating Llama-cpp into LangChain, users can benefit from the powerful language models and generate humanistic and step-by-step responses to their input questions.

### Manifest

Manifest is an integration tool that enhances the capabilities of LangChain, making it more powerful and user-friendly for language processing tasks. It acts as a bridge between LangChain and local Hugging Face models, allowing users to access and utilize these models within LangChain easily. Manifest has been seamlessly integrated into LangChain, providing users with enhanced capabilities for language processing tasks. To utilize Manifest within LangChain, users can follow the provided instructions, which involve installing the manifest-ml package and configuring the connection settings. Once integrated, users can leverage Manifest's functionalities alongside LangChain for a comprehensive language processing experience.

### Modal

Modal is seamlessly integrated into LangChain, adding powerful cloud computing capabilities to the language processing workflow. While Modal does not provide any specific language models (LLMs), it serves as the infrastructure enabling LangChain to leverage serverless cloud computing. By integrating Modal into LangChain, users can directly harness the benefits of on-demand access to cloud resources from their Python scripts on their local computers. By installing the Modal client library and generating a new token, users can authenticate and establish a connection to the Modal server. In the LangChain example, a Modal LLM is instantiated using the endpoint URL, and a PromptTemplate is defined to structure the input. LangChain then executes the LLMChain with the specified prompt and runs a language processing task, such as answering a question.

### NLP Cloud

NLP Cloud seamlessly integrates with LangChain, providing a comprehensive suite of high-performance pre-trained and custom models for a wide range of natural language processing (NLP) tasks. These models are designed for production use and can be accessed through a REST API. By executing the LLMChain with the specified prompt, users can seamlessly perform NLP tasks like answering questions.

### Petals

Petals are seamlessly integrated into LangChain, enabling the utilization of over 100 billion language models within a decentralized architecture similar to BitTorrent. This notebook provides guidance on incorporating Petals into the LangChain workflow. Petals offer a diverse range of language models, and its integration with LangChain enhances natural language understanding and generation capabilities. Petals operate under a decentralized model, providing users with powerful language processing capabilities in a distributed environment.

### PipelineAI

PipelineAI is seamlessly integrated into LangChain, allowing users to scale their machine-learning models in the cloud. Additionally, PipelineAI offers API access to a range of LLM (Large Language Model) models. It includes GPT-J, Stable Diffusion, ESRGAN, DALL·E, GPT-2, and GPT-Neo, each with its own specific model parameters and capabilities. PipelineAI empowers users to leverage the scalability and power of the cloud for their machine-learning workflows within the LangChain ecosystem.

### PredictionGuard

PredictionGuard is seamlessly integrated into LangChain, providing users with a powerful wrapper for their language model usage. To begin using PredictionGuard within the LangChain framework, the predictionguard and LangChain libraries need to be installed. PredictionGuard can also be seamlessly integrated into LangChain's LLMChain for more advanced tasks. PredictionGuard enhances the LangChain experience by providing an additional layer of control and safety to language model outputs.

### PromptLayer OpenAI

PredictionGuard is seamlessly integrated into LangChain, offering users enhanced control and management of their GPT prompt engineering. PromptLayer acts as a middleware between users' code and OpenAI's Python library, enabling the recording, tracking, and exploration of OpenAI API requests through the PromptLayer dashboard. To utilize PromptLayer with OpenAI, the 'promptlayer' package needs to be installed. Users can attach templates to requests, enabling the evaluation of different templates and models within the PromptLayer dashboard.

### Replicate

Replicate is seamlessly integrated into LangChain, providing a wide range of LLM models for various applications. Some of the LLM models offered by Replicate include vicuna-13b, bark, speaker-transcription, stablelm-tuned-alpha-7b, Kandinsky-2, and stable-diffusion. These models cover diverse areas such as language generation, generative audio, speaker transcription, language modeling, and text-to-image generation. Each model has specific parameters and capabilities, enabling users to choose the most suitable model for their needs. Replicate provides flexible pricing options based on the computational resources required for running the models. Replication simplifies the deployment of custom machine-learning models at scale. Users can integrate Replicate into LangChain to interact with these models effectively.

### Runhouse

Runhouse is seamlessly integrated into LangChain, providing powerful remote compute and data management capabilities across different environments and users. Runhouse offers the flexibility to host models on your own GPU infrastructure or leverage on-demand GPUs from cloud providers such as AWS, GCP, and Azure. Runhouse provides several LLM models that can be utilized within LangChain, such as gpt2 and google/flan-t5-small. Users can specify the desired hardware configuration. By combining Runhouse and LangChain, users can easily create advanced language model workflows, enabling efficient model execution and collaboration across different environments and users.

### StochasticAI

StochasticAI aims to simplify the workflow of deep learning models within LangChain, providing users with an efficient and user-friendly environment for model interaction and deployment. It provides a streamlined process for the lifecycle management of Deep Learning models. StochasticAI's Acceleration Platform simplifies tasks such as model uploading, versioning, training, compression, and acceleration, ultimately facilitating the deployment of models into production. Within LangChain, users can interact with StochasticAI models effortlessly. The available LLM models from StochasticAI include FLAN-T5, GPT-J, Stable Diffusion 1, and Stable Diffusion 2. These models offer diverse capabilities for various language-related tasks.

### Writer

The writer is seamlessly integrated into LangChain, providing users with a powerful platform for generating diverse language content. With Writer integration, LangChain users can effortlessly interact with a range of LLM models to meet their language generation needs. The available LLM models provided by Writer include Palmyra Small (128m), Palmyra 3B (3B), Palmyra Base (5B), Camel 🐪 (5B), Palmyra Large (20B), InstructPalmyra (30B), Palmyra-R (30B), Palmyra-E (30B), and Silk Road. These models offer different capacities for improving language understanding, generative pre-training, following instructions, and retrieval-augmented generation. 