import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv(".env", override=True)
print("Loading Environment Variables...")


@tool
def network_search(query: str) -> str | dict:
    """
    这是一个自定义的网络搜索工具
    args:
        query: 要搜索的字符串
    returns:
        查询的结果
    """
    tavily_client = TavilyClient()
    print(f"query 内容是: {query}")

    response = tavily_client.search(query)
    # if response.get("results"):
    #     return " ".join([content.get("content") for content in response["results"]])

    return response


def ollama_llm() -> ChatOllama:
    print("开始启动ollama...")
    ollama = ChatOllama(
        model="gemma-4-E4B-it-GGUF:Q4_K_M",
        temperature=0,
    )
    print("ollama启动成功...")
    return ollama


def openai_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        temperature=0,
        base_url=base_url,
        api_key=api_key,
    )


# llm = ChatOllama(temperature=0, model="gemma-4-E4B-it-GGUF:Q4_K_M")
# print("ollama启动完毕...")

tools = [TavilySearch()]


def main():
    print("Hello from langchain-course!")

    llm = openai_llm()

    agent = create_agent(llm, tools=tools)
    response = agent.invoke(
        {
            "messages": HumanMessage(
                "我想要在领英上搜索三条关于ai agent 应用工程师的岗位要求与 langchain 有关并列出相关的详细信息"
            )
        }
    )
    print(response.get("messages")[-1].content)


if __name__ == "__main__":
    main()
