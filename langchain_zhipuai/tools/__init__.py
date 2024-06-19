from typing import Dict, Union

from langchain.tools import BaseTool
import langchain_zhipuai.tools.tools_factory


def get_tool() -> Union[Dict[str, BaseTool]]:
    from langchain_zhipuai.tools import tools_registry

    return tools_registry._TOOLS_REGISTRY
