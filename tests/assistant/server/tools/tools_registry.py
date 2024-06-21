import json
import re
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from langchain.agents import tool
from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Extra, Field

from langchain_zhipuai.agent_toolkits.all_tools.tool import BaseToolOutput


@tool
def calculate(text: str = Field(description="a math expression")) -> BaseToolOutput:
    """
    Useful to answer questions about simple calculations.
    translate user question to a math expression that can be evaluated by numexpr.
    """
    import numexpr

    try:
        ret = str(numexpr.evaluate(text))
    except Exception as e:
        ret = f"wrong: {e}"

    return BaseToolOutput(ret)
