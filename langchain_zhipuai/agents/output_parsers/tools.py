import json
import logging
from json import JSONDecodeError
from typing import List, Union, Optional, Dict, Any
from zhipuai.core import BaseModel

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

from langchain_zhipuai.chat_models.all_tools_message import ALLToolsMessageChunk

logger = logging.getLogger(__name__)


class CodeInterpreterAgentAction(ToolAgentAction):
    outputs: List[Union[str, dict]] = None
    """Output of the tool call."""
    platform_params: dict = None


class AllToolsMessageToolCall(BaseModel):
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    id: Optional[str]


class AllToolsMessageToolCallChunk(BaseModel):
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    id: Optional[str]
    index: Optional[int]


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
            elif "code_interpreter" == tool_call["type"]:
                code_interpreter = tool_call["code_interpreter"]

                tool_calls.append(
                    ToolCall(
                        name="code_interpreter",
                        args=code_interpreter,
                        id=tool_call["id"] if tool_call["id"] else "abc",
                    )
                )

    code_interpreter_chunk: List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            code_interpreter_chunk = _best_effort_parse_code_interpreter_tool_calls(message.tool_call_chunks)
    else:
        code_interpreter_chunk = _best_effort_parse_code_interpreter_tool_calls(tool_calls)

    if code_interpreter_chunk and len(code_interpreter_chunk) > 1:
        actions.append(
            _paser_code_interpreter_chunk_input(message, code_interpreter_chunk)
        )

    # TODO: parse platform tools built-in @langchain_zhipuai
    # delete code_interpreter_chunk
    tool_calls = [
        tool_call for tool_call in tool_calls if "code_interpreter" != tool_call["name"]
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


def _best_effort_parse_code_interpreter_tool_calls(
        tool_call_chunks: List[dict]
) -> List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]]:
    code_interpreter_chunk: List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]] = []
    # Best-effort parsing allready parsed tool calls
    for code_interpreter in tool_call_chunks:
        if "code_interpreter" == code_interpreter["name"]:

            if isinstance(code_interpreter["args"], str):
                args_ = parse_partial_json(code_interpreter["args"])
            else:
                args_ = code_interpreter["args"]
            if not isinstance(args_, dict):
                raise ValueError("Malformed args.")

            if "outputs" in args_:
                code_interpreter_chunk.append(
                    AllToolsMessageToolCall(
                        name=code_interpreter["name"],
                        args=args_,
                        id=code_interpreter["id"],
                    )
                )
            else:
                code_interpreter_chunk.append(AllToolsMessageToolCallChunk(
                    name=code_interpreter["name"],
                    args=args_,
                    id=code_interpreter["id"],
                    index=code_interpreter.get("index"),
                ))

    return code_interpreter_chunk


def _paser_code_interpreter_chunk_input(
        message: BaseMessage, code_interpreter_chunk: List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]]
) -> CodeInterpreterAgentAction:
    try:
        input_log_chunk = []

        outputs = []
        for interpreter_chunk in code_interpreter_chunk:
            interpreter_chunk_args = interpreter_chunk.args

            if "input" in interpreter_chunk_args:
                input_log_chunk.append(interpreter_chunk_args["input"])
            if "outputs" in interpreter_chunk_args:
                outputs.extend(interpreter_chunk_args["outputs"])

        out_logs = [logs["logs"] for logs in outputs if "logs" in logs]
        log = f"{''.join(input_log_chunk)}\n{''.join(out_logs)}\n"
        tool_call_id = (
            code_interpreter_chunk[0].id
            if code_interpreter_chunk[0].id
            else "abc"
        )
        code_interpreter_action = CodeInterpreterAgentAction(
            tool="code_interpreter",
            tool_input="".join(input_log_chunk),
            outputs=outputs,
            log=log,
            message_log=[message],
            tool_call_id=tool_call_id,
        )

        return code_interpreter_action
    except Exception as e:
        logger.error(f"Error parsing code_interpreter_chunk: {e}", exc_info=True)
        raise OutputParserException(
            f"Could not parse tool input: code_interpreter because "
            f"the `arguments` is not valid JSON."
        )
