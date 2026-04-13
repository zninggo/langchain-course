import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(override=True)

# for key, val in os.environ.items():
#     print(f"{key}: {val}")


if __name__ == "__main__":
    document = TextLoader("./mediumblog1.txt", encoding="utf-8").load()
    print(document)

    text_spliter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    print("create text_spliter")

    chunks = text_spliter.split_documents(document)
    print(f"分块长度{len(chunks)}")

    # bge-m3:latest 是1024的向量维度
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    print("创建ollama嵌入模型")

    # 要花钱 放弃
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # PineconeVectorStore.from_documents(
    #     documents=chunks, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    # )

    vector_store = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    print(vector_store)

    print("finish")
