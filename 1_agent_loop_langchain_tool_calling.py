import os
from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

load_dotenv(override=True)


base_url = os.getenv("OPENAI_BASE_URL")


MAX_ITERATIONS = 10
MODEL_NAME = "gemma-4-E4B-it-GGUF:Q4_K_M"


# ------------tools---------------
@tool
def get_product_price(product: str) -> float:
    """查询商品的价格 需要使用英文 商品有 laptop headphones keyboard

    Args:
        product: 商品名称
    Returns:
        查询到的商品价格, 不存在的商品返回的价格为0
    """

    print(f"\t>> 正在执行工具函数 >> get_product_price(product={product})")

    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """根据折扣等级返回折扣后的价格 等级使用英文 等级有 bronze silver gold

    Args:
        price: 折扣前的价格
        discount_tier: 折扣的等级 有 金 银 铜
    Returns:
        折扣后的价格
    """
    print(
        f"\t>> 正在执行工具函数 >> apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --------agent loop------------


def run_agent(question: str):
    tools = [get_product_price, apply_discount]

    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(model="ollama:{}".format(MODEL_NAME), temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print("question: {question}".format(question=question), end=f"\n{'-'*20}\n")

    messages: List[HumanMessage | SystemMessage | AIMessage | ToolMessage] = [
        SystemMessage(content="""
        现在你是一名购物助理, 你有两项技能分别可以获取商品的价格和商品折扣后的价格
        
        规则:
            1. 你不准自己猜测或假设商品的价格
            2. 你只有调用 get_product_price 工具才能获取准确的商品价格
            3. 只有拿到商品价格才能使用 apply_discount 工具计算最终折扣后的价格
            4. 不允许直接使用 get_product_price 计算的价格
            5. 客户没有指定折扣等级你就要主动询问使用哪个折扣
            6. 任何私自的行为都是不允许的
        """),
        HumanMessage(content=question),
    ]

    for count in range(1, MAX_ITERATIONS + 1):
        print(f"{'-'*10} 第 {count:02} 次迭代 {'-'*10}")

        llm_msg = llm_with_tools.invoke(messages)

        # 没有工具调用了返回 content
        if not llm_msg.tool_calls:
            print(f'\t>> 已经没有再调用tool了 返回最终结果~')
            return llm_msg.content

        use_tool = llm_msg.tool_calls[0]
        tool_name = use_tool.get("name")

        tool_args = use_tool.get("args", {})
        tool_id = use_tool.get("id")

        print(f"\t>> llm 选择了工具函数 >> {tool_name} 参数是: {tool_args}")

        if tools_dict.get(tool_name) is None:
            raise ValueError(f"没有找到{tool_name} 工具函数")

        tool_result = tools_dict.get(tool_name).invoke(tool_args)
        print(f"\t>> llm 调用工具函数的结果是 >> {tool_result}")

        messages.append(llm_msg)

        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

    print(f'\t>> 达到了agent loop最大 {MAX_ITERATIONS} 次数')
    return None

"""
“你是一位乐于助人的购物助理。”
“您可以使用产品目录工具”
“还有一个折扣工具。\n\n”
“严格规则 - 您必须严格遵守这些规则：\n”
“1. 切勿猜测或假设任何产品价格。”
“您必须先调用 get_product_price 才能获取实际价格。\n”
“2. 仅在收到后致电 apply_discount”
“来自 get_product_price 的价格。传递确切的价格”
“由 get_product_price 返回 — 不要传递虚构的数字。\n”
“3.永远不要自己用数学计算折扣。”
“始终使用 apply_discount 工具。\n”
“4.如果用户未指定折扣等级，”
“询问他们使用哪一层——不要假设是一层。”
"""

if __name__ == "__main__":

    print(f"hello agent loop!", end="\n\n")

    result = run_agent("笔记本黄金折扣的电脑价格是多少?")
    print(f'llm 的回答是: {result}')
