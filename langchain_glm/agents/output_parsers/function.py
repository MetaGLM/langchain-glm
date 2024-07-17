import json
import logging
from collections import deque
from typing import Any, Dict, List, Deque, Union

from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    BaseMessage,
    ToolCall,
)
from langchain_core.utils.json import parse_partial_json

from langchain_glm.agent_toolkits.all_tools.struct_type import AdapterAllToolStructType
from langchain_glm.agents.output_parsers._utils import find_object_positions, concatenate_segments
from langchain_glm.agents.output_parsers.base import AllToolsMessageToolCall, AllToolsMessageToolCallChunk

logger = logging.getLogger(__name__)


def _best_effort_parse_function_tool_calls(
        tool_call_chunks: List[dict],
) -> List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]]:
    function_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    # Best-effort parsing allready parsed tool calls
    for function in tool_call_chunks:
        if function["name"] not in AdapterAllToolStructType.__members__.values():
            if isinstance(function["args"], str):
                args_ = parse_partial_json(function["args"])
            else:
                args_ = function["args"]
            if not isinstance(args_, dict):
                raise ValueError("Malformed args.")

            if len(args_.keys()) > 0:
                function_chunk.append(
                    AllToolsMessageToolCall(
                        name=function["name"],
                        args=args_,
                        id=function["id"],
                    )
                )
            else:
                function_chunk.append(
                    AllToolsMessageToolCallChunk(
                        name=function["name"],
                        args=args_,
                        id=function["id"],
                        index=function.get("index"),
                    )
                )

    return function_chunk


def _paser_function_chunk_input(
        message: BaseMessage,
        function_chunk: List[
            Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
        ],
) -> Deque[ToolAgentAction]:

    try:
        function_action_result_stack: Deque[ToolAgentAction] = deque()
        for _chunk in function_chunk:
            if isinstance(_chunk, AllToolsMessageToolCall):

                function_name = _chunk.name
                _tool_input = _chunk.args
                tool_call_id = _chunk.id if _chunk.id else "abc"
                if "__arg1" in _tool_input:
                    tool_input = _tool_input["__arg1"]
                else:
                    tool_input = _tool_input

                content_msg = f"responded: {message.content}\n" if message.content else "\n"
                log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"

                function_action_result_stack.append(
                    ToolAgentAction(
                        tool=function_name,
                        tool_input=tool_input,
                        log=log,
                        message_log=[message],
                        tool_call_id=tool_call_id,
                    )
                )

        return function_action_result_stack

    except Exception as e:
        logger.error(f"Error parsing function_chunk: {e}", exc_info=True)
        raise OutputParserException(
            f"Error parsing function_chunk: {e} "
        )