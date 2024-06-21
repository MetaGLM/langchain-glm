import logging
import logging.config

import pytest

from langchain_zhipuai.agents.zhipuai_all_tools import (
    ZhipuAIAllToolsRunnable,
)
from langchain_zhipuai.agents.zhipuai_all_tools.base import (
    AllToolsAction,
    AllToolsActionToolEnd,
    AllToolsActionToolStart,
    AllToolsFinish,
    AllToolsLLMStatus,
)
from langchain_zhipuai.callbacks.agent_callback_handler import (
    AgentStatus,
)


@pytest.mark.asyncio
async def test_all_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="tob-alltools-api-dev"
        # tools=[
        #     {
        #         "type": "code_interpreter"
        #     }
        # ]
    )
    chat_iterator = agent_executor.invoke(chat_input="看下本地文件有哪些，告诉我你用的是什么文件,查看当前目录")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

    chat_iterator = agent_executor.invoke(chat_input="打印下test_alltools.py")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

