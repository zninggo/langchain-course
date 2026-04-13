import os

import ollama
from dotenv import load_dotenv

load_dotenv(override=True)


base_url = os.getenv("OPENAI_BASE_URL")

MAX_ITERATIONS = 10
MODEL_NAME = "gemma-4-E4B-it-GGUF:Q4_K_M"


# ------------tools---------------
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


ollama_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


def ollama_chat(messages):
    return ollama.chat(model=MODEL_NAME, tools=ollama_tools, messages=messages, think=True, options={'temperature': 0})


# --------agent loop------------


def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print("question: {question}".format(question=question), end=f"\n{'-'*20}\n")

    messages = [
        {
            "role": "system",
            "content": """
                        现在你是一名购物助理, 你有两项技能分别可以获取商品的价格和商品折扣后的价格
                        
                        规则:
                            1. 你不准自己猜测或假设商品的价格
                            2. 你只有调用 get_product_price 工具才能获取准确的商品价格
                            3. 只有拿到商品价格才能使用 apply_discount 工具计算最终折扣后的价格
                            4. 不允许直接使用 get_product_price 计算的价格
                            5. 客户没有指定折扣等级你就要主动询问使用哪个折扣
                            6. 任何私自的行为都是不允许的
                        """,
        },
        {"role": "user", "content": question},
    ]

    for count in range(1, MAX_ITERATIONS + 1):
        print(f"{'-'*10} 第 {count:02} 次迭代 {'-'*10}")

        llm_response = ollama_chat(messages)
        llm_msg = llm_response.message

        # 没有工具调用了返回 content
        if not llm_msg.tool_calls:
            print(f"\t>> 已经没有再调用tool了 返回最终结果~")
            return llm_msg.content

        use_tool = llm_msg.tool_calls[0]

        use_func = use_tool.function
        tool_name = use_func.get("name")

        tool_args = use_func.get("arguments", {})

        print(f"\t>> llm 选择了工具函数 >> {tool_name} 参数是: {tool_args}")

        if tools_dict.get(tool_name) is None:
            raise ValueError(f"没有找到{tool_name} 工具函数")

        tool_result = tools_dict.get(tool_name)(**tool_args)


        print(f"\t>> llm 调用工具函数的结果是 >> {tool_result}")

        messages.append(llm_msg)

        messages.append({'role': 'tool',  'tool_name': tool_name, 'content': str(tool_result)})

    print(f"\t>> 达到了agent loop最大 {MAX_ITERATIONS} 次数")
    return None


if __name__ == "__main__":

    print(f"hello agent loop!", end="\n\n")

    result = run_agent("笔记本黄金折扣的电脑价格是多少?")
    print(f"llm 的回答是: {result}")
