import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
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

# llm = ChatOpenAI(
#     model=os.getenv("GLM_MODEL"),
#     base_url=os.getenv("GLM_BASE_URL"),
#     api_key=os.getenv("GLM_API_KEY"),
# )

prompt = ChatPromptTemplate.from_template(
    """
    仅根据以下上下文回答问题: {context}
    
    问题: {question}
    
    提供详细解答: 
    """
)


def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])


def retrieval_chain_without_lcel(question: str):
    """
    无需 LCEL 的简单检索链。
    手动检索文档、格式化文档并生成响应。

    限制：
    - 手动逐步执行
    - 没有内置流媒体支持
    - 没有额外的代码就没有异步支持
    - 更难与其他链组合
    - 更冗长且容易出错
    """
    result = vector_store.similarity_search(question, k=3)

    docs_content = format_docs(result)

    messages = prompt.format_messages(question=question, context=docs_content)

    response = llm.invoke(messages)
    return response.content

def main():
    print("Hello from langchain-course!")

    question = '机器学习中的 Pinecone 是什么?'

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=question)])
    print("\nAnswer:")
    # print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(question)
    print("\nAnswer:")
    print(result_without_lcel)



if __name__ == "__main__":
    main()
