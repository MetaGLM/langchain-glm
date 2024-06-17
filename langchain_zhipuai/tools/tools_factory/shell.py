# LangChain 的 Shell 工具
from langchain.tools.shell import ShellTool
from pydantic.v1 import Field

from .tools_registry import BaseToolOutput, regist_tool


@regist_tool(title="系统命令")
def shell(query: str = Field(description="The command to execute")):
    """Use Shell to execute system shell commands"""
    tool = ShellTool()
    return BaseToolOutput(tool.run(tool_input=query))
