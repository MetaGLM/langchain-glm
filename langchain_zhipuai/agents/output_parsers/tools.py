import json
from json import JSONDecodeError
from typing import List, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, Generation

from langchain.agents.agent import MultiActionAgentOutputParser


class ToolAgentAction(AgentActionMessageLog):
    tool_call_id: str
    """Tool call that this message is responding to."""


class CodeInterpreterALLToolAgentAction(ToolAgentAction):
    outputs: List[Union[str, dict]]
    """Output of the tool call."""
    input: str


def parse_ai_message_to_tool_action(
        message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    actions: List = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get("tool_calls"):
            return AgentFinish(
                return_values={"output": message.content}, log=str(message.content)
            )
        # Best-effort parsing allready parsed tool calls
        # TODO: parse platform tools built-in @langchain_zhipuai.agents.zhipuai_all_tools.base._get_assistants_tool
        # type in the future "function" or "code_interpreter"
        tool_calls = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            if 'function' in tool_call['type']:
                function = tool_call["function"]
                function_name = function["name"]
                try:
                    args = json.loads(function["arguments"] or "{}")
                    tool_calls.append(
                        ToolCall(name=function_name,
                                 args=args,
                                 id=tool_call["id"] if tool_call["id"] else "abc")
                    )
                except JSONDecodeError:
                    raise OutputParserException(
                        f"Could not parse tool input: {function} because "
                        f"the `arguments` is not valid JSON."
                    )
            elif 'code_interpreter' in tool_call['type']:
                code_interpreter = tool_call["code_interpreter"]

                tool_calls.append(
                    ToolCall(name='code_interpreter',
                             args=code_interpreter,
                             id=tool_call["id"] if tool_call["id"] else "abc"
                             )
                )

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
                tool_call_id=tool_call["id"],
            )
        )
    return actions


class ToolsAgentOutputParser(MultiActionAgentOutputParser):
    """Parses a message into agent actions/finish.

    If a tool_calls parameter is passed, then that is used to get
    the tool names and tool inputs.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "tools-agent-output-parser"

    def parse_result(
            self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return parse_ai_message_to_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")
