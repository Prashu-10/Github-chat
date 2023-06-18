import os
import requests
import fnmatch
import argparse
import base64

from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

print(GITHUB_TOKEN)

def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo


def get_files_from_github_repo(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")


def fetch_md_contents(files):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.md"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ", file['path'])
                md_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents

def fetch_py_contents(files):
    py_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.py"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ",file['path'])
                py_contents.append(Document(page_content=decoded_content, metadata={"source":file['path']}))
            else:
                print(f"Error downloading file {file['path']}:{response.status_code}")
    return py_contents



def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks_text = []
    source_chunks_code = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in fetch_md_contents(files):
        for chunk in splitter.split_text(source.page_content):
            source_chunks_text.append(Document(page_content=chunk, metadate=source.metadata))
    # for source in fetch_py_contents(files):
    #     for chunk in splitter.split_text(source.page_content):
    #         source_chunks_code.append(Document(page_content=chunk, metadate=source.metadata)) 
    return source_chunks_text



def main():
    parser = argparse.ArgumentParser(description="Fetch all *.md files from a GitHub repository.")
    parser.add_argument("url", help="GitHub repository URL")
    args = parser.parse_args()

    GITHUB_OWNER, GITHUB_REPO = parse_github_url(args.url)
    
    all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)

    CHROMA_DB_PATH = f'./chroma_{GITHUB_REPO}_1/{os.path.basename(GITHUB_REPO)}'

    chroma_db = None

    if not os.path.exists(CHROMA_DB_PATH):
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks_text = get_source_chunks(all_files) 
        # with open('source_chunks.json', 'w') as source:
        #     source.write(str(source_chunks))
        #     source.close()
        chroma_db = Chroma.from_documents(source_chunks_text, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        chroma_db.persist()
    else:
        print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())
    
    while True:
        print('\n\n\033[31m' + 'Ask a question' + '\033[m')
        user_input = input()
        print('\033[31m' + qa.run(user_input) + '\033[m')
        input_answer = qa.run(user_input)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and "
                "provides lots of specific details from its context. If the AI does not know the answer to a "
                "question, it truthfully says it does not know."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferMemory(return_messages=True)
        conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
        # conversation.predict(qa.run(user_input))
        print('\033[31m' + 'AI Assitant: ' + conversation.predict(input = input_answer) + '\033[m')

        new_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and "
                "provides lots of specific details from its context. If the AI does not know the answer to a "
                "question, it truthfully says it does not know."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        llm_1 = ChatOpenAI(temperature=0)
        memory_1 = ConversationBufferMemory(return_messages=True)
        conversation_1 = ConversationChain(memory=memory_1, prompt=new_prompt, llm=llm_1)
        print('\033[31m' + 'Human Assitant: ' + conversation_1.predict(input = conversation.predict(input = input_answer)) + '\033[m')

    
if __name__ == "__main__":
    main()