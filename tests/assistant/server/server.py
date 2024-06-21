import threading
from typing import List, Tuple

from langchain_zhipuai.agents.zhipuai_all_tools import ZhipuAIAllToolsRunnable
from langchain_zhipuai.agents.zhipuai_all_tools.base import OutputType
from tests.assistant.server.tools.tools_registry import BaseToolOutput, calculate
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, status
from sse_starlette.sse import EventSourceResponse

from zhipuai.core.logs import (
    get_config_dict,
    get_log_file,
    get_timestamp_ms,
)
from uvicorn import Config, Server
import logging.config
from langchain_core.agents import AgentAction

intermediate_steps: List[Tuple[AgentAction, BaseToolOutput]] = []


async def chat(query: str = Body(..., description="用户输入", examples=["帮我计算100+1"]),
               message_id: str = Body(None, description="数据库消息ID"),
               history: List = Body(
                   [],
                   description="历史对话，设为一个整数可以从数据库中读取历史消息",
                   examples=[
                       [
                           {"role": "user",
                            "content": "你好"},
                           {"role": "assistant", "content": "有什么需要帮助的"}
                       ]
                   ]
               )
               ):
    '''Agent 对话'''
    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="chatglm3-qingyan-alltools-130b",
        history=history,
        intermediate_steps=intermediate_steps,
        tools=[
            {
                "type": "code_interpreter"
            },
            calculate
        ]
    )
    chat_iterator = agent_executor.invoke(chat_input=query)

    async def chat_generator():
        async for chat_output in chat_iterator:
            yield chat_output.to_json()

            # if agent_executor.callback.out:
            # intermediate_steps.extend(agent_executor.callback.intermediate_steps)

    return EventSourceResponse(chat_generator())


if __name__ == "__main__":
    logging_conf = get_config_dict(
        "debug",
        get_log_file(log_path="logs", sub_dir=f"local_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    chat_router = APIRouter()

    chat_router.add_api_route(
        "/chat",
        chat,
        response_model=OutputType,
        status_code=status.HTTP_200_OK,
        methods=["POST"],
        description="与llm模型对话(通过LLMChain)"
    )
    app.include_router(chat_router)

    config = Config(
        app=app,
        host="127.0.0.1",
        port=10000,
        log_config=logging_conf,
    )
    _server = Server(config)


    def run_server():
        _server.shutdown_timeout = 2  # 设置为2秒

        _server.run()


    _server_thread = threading.Thread(target=run_server)
    _server_thread.start()
    _server_thread.join()
