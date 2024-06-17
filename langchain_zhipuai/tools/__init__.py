from typing import Dict, Union

from langchain.tools import BaseTool


def get_tool() -> Union[Dict[str, BaseTool]]:
    from .tools_factory import tools_registry

    return tools_registry._TOOLS_REGISTRY
