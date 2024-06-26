import json
import logging
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.utils.json import (
    parse_partial_json,
)
from zhipuai.core import BaseModel

from langchain_zhipuai.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_zhipuai.agents.output_parsers.base import (
    AllToolsMessageToolCall,
    AllToolsMessageToolCallChunk,
)
from langchain_zhipuai.agents.output_parsers.code_interpreter import (
    _best_effort_parse_code_interpreter_tool_calls,
    _paser_code_interpreter_chunk_input,
)
from langchain_zhipuai.agents.output_parsers.drawing_tool import (
    _best_effort_parse_drawing_tool_tool_calls,
    _paser_drawing_tool_chunk_input,
)
from langchain_zhipuai.agents.output_parsers.web_browser import (
    _best_effort_parse_web_browser_tool_calls,
    _paser_web_browser_chunk_input,
)
from langchain_zhipuai.chat_models.all_tools_message import ALLToolsMessageChunk

logger = logging.getLogger(__name__)


def parse_ai_message_to_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    # TODO: parse platform tools built-in @langchain_zhipuai.agents.zhipuai_all_tools.base._get_assistants_tool
    #   type in the future "function" or "code_interpreter"
    #   for @ToolAgentAction from langchain.agents.output_parsers.tools
    #   import with langchain.agents.format_scratchpad.tools.format_to_tool_messages
    actions: List = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get("tool_calls"):
            return AgentFinish(
                return_values={"output": message.content}, log=str(message.content)
            )
        # Best-effort parsing allready parsed tool calls
        tool_calls = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            if "function" == tool_call["type"]:
                function = tool_call["function"]
                function_name = function["name"]
                try:
                    args = json.loads(function["arguments"] or "{}")
                    tool_calls.append(
                        ToolCall(
                            name=function_name,
                            args=args,
                            id=tool_call["id"] if tool_call["id"] else "abc",
                        )
                    )
                except JSONDecodeError:
                    raise OutputParserException(
                        f"Could not parse tool input: {function} because "
                        f"the `arguments` is not valid JSON."
                    )
            elif tool_call["type"] in AdapterAllToolStructType.__members__.values():
                adapter_tool = tool_call[tool_call["type"]]

                tool_calls.append(
                    ToolCall(
                        name=tool_call["type"],
                        args=adapter_tool if adapter_tool else {},
                        id=tool_call["id"] if tool_call["id"] else "abc",
                    )
                )

    code_interpreter_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            code_interpreter_chunk = _best_effort_parse_code_interpreter_tool_calls(
                message.tool_call_chunks
            )
    else:
        code_interpreter_chunk = _best_effort_parse_code_interpreter_tool_calls(
            tool_calls
        )

    if code_interpreter_chunk and len(code_interpreter_chunk) > 1:
        actions.append(
            _paser_code_interpreter_chunk_input(message, code_interpreter_chunk)
        )

    drawing_tool_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            drawing_tool_chunk = _best_effort_parse_drawing_tool_tool_calls(
                message.tool_call_chunks
            )
    else:
        drawing_tool_chunk = _best_effort_parse_drawing_tool_tool_calls(tool_calls)

    if drawing_tool_chunk and len(drawing_tool_chunk) > 1:
        actions.append(_paser_drawing_tool_chunk_input(message, drawing_tool_chunk))

    web_browser_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            web_browser_chunk = _best_effort_parse_web_browser_tool_calls(
                message.tool_call_chunks
            )
    else:
        web_browser_chunk = _best_effort_parse_web_browser_tool_calls(tool_calls)

    if web_browser_chunk and len(web_browser_chunk) > 1:
        actions.append(_paser_web_browser_chunk_input(message, web_browser_chunk))

    # TODO: parse platform tools built-in @langchain_zhipuai
    # delete AdapterAllToolStructType from tool_calls
    tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call["name"] not in AdapterAllToolStructType.__members__.values()
    ]

    for tool_call in tool_calls:
        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        function_name = tool_call["name"]
        _tool_input = tool_call["args"]
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = f"responded: {message.content}\n" if message.content else "\n"
        log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"

        actions.append(
            ToolAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
                tool_call_id=tool_call["id"] if tool_call["id"] else "abc",
            )
        )
    return actions
