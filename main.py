import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

prompt = ChatPromptTemplate.from_template("""
    仅根据以下上下文回答问题: {context}
    
    问题: {question}
    
    提供详细解答: 
    """)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


retriever = vector_store.as_retriever(search_kwargs={"k": 3})


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


def create_retrieval_chain_with_lcel():
    """
    使用 LCEL（LangChain 表达式语言）创建检索链。
        返回一个可以通过 {"question": "..."} 调用的链

        相对于非 LCEL 方法的优点：
        - 声明性和可组合性：易于使用管道运算符 (|) 进行链接操作
        - 内置流：chain.stream() 开箱即用
        - 内置异步：可用 chain.ainvoke() 和 chain.astream()
        - 批处理：chain.batch() 用于多个输入
        - 类型安全：与LangChain的类型系统更好地集成
        - 更少的代码：更简洁和可读
        - 可重复使用：链可以保存、共享并与其他链组合
        - 更好的调试：LangChain提供更好的可观察性工具
    """

    retrieval_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retriever | format_docs
            )
            | prompt
            | llm
            | StrOutputParser()
    )
    return retrieval_chain


def main():
    print("Hello from langchain-course!")

    question = "机器学习中的 Pinecone 是什么?"

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
    # result_without_lcel = retrieval_chain_without_lcel(question)
    # print("\nAnswer:")
    # print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": question})
    print("\nAnswer:")
    print(result_with_lcel)


if __name__ == "__main__":
    main()
