# LangChain 的 Shell 工具
from pydantic.v1 import Field

from langchain_zhipuai.tools.tools_registry import BaseToolOutput, regist_tool


@regist_tool(title="系统命令")
def code_interpreter(query: str = Field(description="The command to execute")):

    return BaseToolOutput( )
