from typing import Dict, Type

from langchain_zhipuai.agent_toolkits import AdapterAllTool
from langchain_zhipuai.agent_toolkits.all_tools.code_interpreter_tool import (
    CodeInterpreterAdapterAllTool,
)
from langchain_zhipuai.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)

TOOL_STRUCT_TYPE_TO_TOOL_CLASS: Dict[AdapterAllToolStructType, Type[AdapterAllTool]] = {
    AdapterAllToolStructType.CODE_INTERPRETER: CodeInterpreterAdapterAllTool,
}
