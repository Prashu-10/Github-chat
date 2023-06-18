
import os
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

chat = ChatOpenAI(temperature=0)
template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
chain = LLMChain(llm=chat, prompt=chat_prompt)

print(chain.run(input_language="English", output_language="", text="I love programming."))

