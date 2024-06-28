from operator import itemgetter
from typing import Dict, List, Union

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_zhipuai import ChatZhipuAI

from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base ** exponent


def test_tool_use():
    llm = ChatZhipuAI(model="glm-4-alltools", streaming=True)

    tools = [multiply, exponentiate, add]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}

    def call_tools(msg: AIMessage) -> Runnable:
        """Simple sequential tool calling helper."""
        tool_map = {tool.name: tool for tool in tools}
        tool_calls = msg.tool_calls.copy()
        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        return tool_calls

    chain = llm_with_tools | call_tools
    out = chain.invoke(
        "What's 23 times 7, and what's five times 18 and add a million plus a billion and cube thirty-seven"
    )
    print(out)