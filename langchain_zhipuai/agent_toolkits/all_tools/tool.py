"""Tool for interacting with a single API with natural language definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, Dict, Tuple
from langchain_core.callbacks.manager import (
    Callbacks,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import json
import logging

logger = logging.getLogger(__name__)


class BaseToolOutput:
    """
    LLM 要求 Tool 的输出为 str，但 Tool 用在别处时希望它正常返回结构化数据。
    只需要将 Tool 返回值用该类封装，能同时满足两者的需要。
    基类简单的将返回值字符串化，或指定 format="json" 将其转为 json。
    用户也可以继承该类定义自己的转换方法。
    """

    def __init__(
            self,
            data: Any,
            format: str = "",
            data_alias: str = "",
            **extras: Any,
    ) -> None:
        self.data = data
        self.format = format
        self.extras = extras
        if data_alias:
            setattr(self, data_alias, property(lambda obj: obj.data))

    def __str__(self) -> str:
        if self.format == "json":
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        else:
            return str(self.data)


class AdapterAllTool(BaseTool):
    """platform adapter tool for all tools."""

    name: str
    platform_params: Dict[str, Any]

    @classmethod
    def from_platform_dict(
            cls, name: str, platform_params: Dict[str, Any], callbacks: Callbacks = None
    ) -> "AdapterAllTool":
        """Convert a platform dict to a tool."""

        return cls(name=name, platform_params=platform_params, callbacks=callbacks)

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if tool_input is None:
            return (), {}
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            # for tool defined with `*args` parameters
            # the args_schema has a field named `args`
            # it should be expanded to actual *args
            # e.g.: test_tools
            #       .test_named_tool_decorator_return_direct
            #       .search_api
            if "args" in tool_input:
                args = tool_input["args"]
                if args is None:
                    tool_input.pop("args")
                    return (), tool_input
                elif isinstance(args, tuple):
                    tool_input.pop("args")
                    return args, tool_input
            return (), tool_input

    def _run(
            self,
            tool: str,
            tool_input: str,
            log: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> BaseToolOutput:
        """Use the tool."""

        return BaseToolOutput(
            f"""Access：{tool}, Message: {tool_input},{log}"""
        )

    async def _arun(
            self,
            tool: str,
            tool_input: str,
            log: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> BaseToolOutput:
        """Use the tool asynchronously."""

        return BaseToolOutput(
            f"""Access：{tool}, Message: {tool_input},{log}"""
        )
