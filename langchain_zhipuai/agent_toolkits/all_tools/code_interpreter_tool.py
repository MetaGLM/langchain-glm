from dataclasses import dataclass

from langchain_core.agents import AgentAction

from langchain_zhipuai.agent_toolkits import AdapterAllTool

from typing import TYPE_CHECKING, Any, Optional, Union, Dict, Tuple, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun, AsyncCallbackManagerForChainRun,
)

from langchain_zhipuai.agent_toolkits.all_tools.tool import BaseToolOutput, AllToolExecutor


class CodeInterpreterToolOutput(BaseToolOutput):
    platform_params: Dict[str, Any]

    def __init__(
            self,
            data: Any,
            platform_params: Dict[str, Any],
            **extras: Any,
    ) -> None:
        super().__init__(data, "", "", **extras)
        self.platform_params = platform_params


@dataclass
class CodeInterpreterAllToolExecutor(AllToolExecutor):
    """platform adapter tool for code interpreter tool"""
    name: str

    def run(
            self,
            tool: str,
            tool_input: str,
            log: str,
            outputs: List[Union[str, dict]] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CodeInterpreterToolOutput:

        if outputs is None or str(outputs).strip() == "":
            if 'auto' == self.platform_params.get('sandbox', 'auto'):
                raise ValueError(f"Tool {self.name} sandbox is auto , but log is None, is server error")
            elif 'none' == self.platform_params.get('sandbox', 'auto'):
                raise NotImplementedError(f"Tool {self.name} sandbox not auto , implement it")

        return CodeInterpreterToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )

    async def arun(
            self,
            tool: str,
            tool_input: str,
            log: str,
            outputs: List[Union[str, dict]] = None,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> CodeInterpreterToolOutput:
        """Use the tool asynchronously."""
        if outputs is None or str(outputs).strip() == "":
            if 'auto' == self.platform_params.get('sandbox', 'auto'):
                raise ValueError(f"Tool {self.name} sandbox is auto , but log is None, is server error")
            elif 'none' == self.platform_params.get('sandbox', 'auto'):
                raise NotImplementedError(f"Tool {self.name} sandbox not auto , implement it")

        return CodeInterpreterToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )


class CodeInterpreterAdapterAllTool(AdapterAllTool[CodeInterpreterAllToolExecutor]):

    @classmethod
    def get_type(cls) -> str:
        return "CodeInterpreterAdapterAllTool"

    def _build_adapter_all_tool(self, platform_params: Dict[str, Any]) -> CodeInterpreterAllToolExecutor:
        return CodeInterpreterAllToolExecutor(name="code_interpreter", platform_params=platform_params)
