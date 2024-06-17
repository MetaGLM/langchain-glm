import logging
import logging.config
from asyncio import sleep

import pytest

from langchain_zhipuai.agents.zhipuai_all_tools import (
    AllToolsChatInput,
    ZhipuAIAllToolsRunnable,
)
from langchain_zhipuai.agents.zhipuai_all_tools.base import (
    AllToolsAction,
    AllToolsActionToolEnd,
    AllToolsActionToolStart,
    AllToolsFinish,
    AllToolsLLMStatus,
)
from langchain_zhipuai.callbacks.callback_handler.agent_callback_handler import (
    AgentExecutorAsyncIteratorCallbackHandler,
    AgentStatus,
)


@pytest.mark.asyncio
async def test_all_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore
    callback = AgentExecutorAsyncIteratorCallbackHandler()

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="chatglm3-qingyan-alltools-130b", callback=callback,
        tools=[
            {
                "type": "code_interpreter"
            }
        ]
    )
    chat_input = AllToolsChatInput(
        query="""你好，帮我算下100*1，然后给我说下你用什么计算的"""
    )
    chat_iterator = agent_executor.invoke(chat_input=chat_input)
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionEnd:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)
