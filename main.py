import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)

embeddings = OllamaEmbeddings(model="bge-m3:latest")

vector_store = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
)


def main():
    print("Hello from langchain-course!")

    response = llm.invoke("你能为我做什么")
    print(response.content)


if __name__ == "__main__":
    main()
