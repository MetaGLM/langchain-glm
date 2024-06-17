import asyncio
import json
import logging
import queue
import time
import uuid
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import zhipuai
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic.v1 import BaseModel, Field

from langchain_zhipuai.agents.all_tools_bind.base import create_zhipuai_tools_agent
from langchain_zhipuai.callbacks.callback_handler.agent_callback_handler import (
    AgentExecutorAsyncIteratorCallbackHandler,
    AgentStatus,
)
from langchain_zhipuai.chat_models import ChatZhipuAI
from langchain_zhipuai.tools import get_tool
from langchain_zhipuai.utils import History

logger = logging.getLogger()


def _is_assistants_builtin_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> bool:
    """platform tools built-in"""
    assistants_builtin_tools = (
        "code_interpreter",
    )
    return (
        isinstance(tool, dict)
        and ("type" in tool)
        and (tool["type"] in assistants_builtin_tools)
    )


def _get_assistants_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an ZhipuAI tool.

    such as "code_interpreter"
    """
    if _is_assistants_builtin_tool(tool):
        return tool  # type: ignore
    else:
        # in case of a custom tool, convert it to an function of type
        return convert_to_openai_tool(tool)


def _agents_registry(
    llm: BaseLanguageModel,
    llm_with_all_tools: RunnableBindingBase = None,
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = [],
    callbacks: List[BaseCallbackHandler] = [],
    verbose: bool = False,
):
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_zhipuai_tools_agent(
        prompt=prompt,
        llm_with_all_tools=llm_with_all_tools
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=verbose, callbacks=callbacks
    )

    return agent_executor


class MsgType:
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        msg = f"Caught exception: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)
    finally:
        # Signal the aiter to stop.
        event.set()


class AllToolsAction(AgentAction):
    """AgentFinish with run and thread metadata."""

    run_id: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True


class AllToolsFinish(AgentFinish):
    """AgentFinish with run and thread metadata."""

    run_id: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True


class AllToolsActionToolStart(AgentAction):
    """AllToolsAction with run and thread metadata."""

    run_id: Optional[str] = None

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True


class AllToolsActionToolEnd(AgentAction):
    """AllToolsActionEnd with run and thread metadata."""

    run_id: str
    is_error: bool
    tool_output: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True


class AllToolsLLMStatus(BaseModel):
    run_id: str
    status: int  # AgentStatus
    text: str
    message_type: int = MsgType.TEXT

    class Config:
        arbitrary_types_allowed = True

    def model_dump(self) -> dict:
        result = {
            "run_id": self.run_id,
            "status": self.status,
            "message_type": self.message_type,
        }

        return result

    def model_dump_json(self):
        return json.dumps(self.model_dump(), ensure_ascii=False)


class AllToolsChatInput(BaseModel):
    id: Optional[str] = None
    query: Optional[str] = None
    history: Optional[List[History]] = None

    class Config:
        arbitrary_types_allowed = True


OutputType = Union[
    AllToolsAction,
    AllToolsActionToolStart,
    AllToolsActionToolEnd,
    AllToolsFinish,
    AllToolsLLMStatus,
]


class ZhipuAIAllToolsRunnable(RunnableSerializable[Dict, OutputType]):
    agent_executor: AgentExecutor
    """ZhipuAI AgentExecutor."""

    model_name: str = Field(default="chatglm3-qingyan-alltools-130b")
    """工具模型"""
    callback: AsyncIteratorCallbackHandler
    """ZhipuAI AgentExecutor callback."""
    check_every_ms: float = 1_000.0
    """Frequency with which to check run progress in ms."""
    _call_data: Dict[str, Any] = {}
    """_call_data to store the data to be processed."""
    _message_data: Dict[str, Any] = {}
    """_message_data to store the data to be processed."""

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create_agent_executor(
        cls,
        model_name: str,
        callback: AsyncIteratorCallbackHandler,
        *,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> "ZhipuAIAllToolsRunnable":
        """Create an ZhipuAI Assistant and instantiate the Runnable."""
        callbacks = [callback]
        params = dict(
            streaming=True,
            verbose=True,
            callbacks=callbacks,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

        llm = ChatZhipuAI(**params)
        all_tools = get_tool().values()

        llm_with_all_tools = None
        if tools:
            temp_tools = []
            temp_tools.extend(tools)
            temp_tools.extend(all_tools)
            llm_with_all_tools = llm.bind(
                tools=[_get_assistants_tool(tool) for tool in temp_tools]
            )

        tools = [t.copy(update={"callbacks": callbacks}) for t in all_tools]
        agent_executor = _agents_registry(
            llm=llm, callbacks=callbacks,
            tools=tools,
            llm_with_all_tools=llm_with_all_tools,
            verbose=True
        )
        return cls(
            model_name=model_name,
            agent_executor=agent_executor,
            callback=callback,
            **kwargs,
        )

    def invoke(
        self, chat_input: AllToolsChatInput, config: Optional[RunnableConfig] = None
    ) -> AsyncIterable[OutputType]:
        async def chat_iterator() -> AsyncIterable[OutputType]:
            history_message = []
            if chat_input.history:
                _history = [History.from_data(h) for h in chat_input.history]
                chat_history = [h.to_msg_tuple() for h in _history]

                history_message = convert_to_messages(chat_history)

            task = asyncio.create_task(
                wrap_done(
                    self.agent_executor.ainvoke(
                        {
                            "input": chat_input.query,
                            "chat_history": history_message,
                        }
                    ),
                    self.callback.done,
                )
            )

            async for chunk in self.callback.aiter():
                data = json.loads(chunk)
                class_status = None
                if data["status"] == AgentStatus.llm_start:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["text"],
                    )
                    self._message_data[data["run_id"]] = class_status

                elif data["status"] == AgentStatus.llm_new_token:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["text"],
                    )
                elif data["status"] == AgentStatus.llm_end:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["text"],
                    )
                elif data["status"] == AgentStatus.agent_action:
                    class_status = AllToolsAction(
                        run_id=data["run_id"], **data["action"]
                    )
                    self._call_data[data["run_id"]] = class_status

                elif data["status"] == AgentStatus.tool_start:
                    class_status = AllToolsActionToolStart(
                        run_id=data["run_id"],
                        tool_input=data["tool_input"],
                        tool=data["tool"],
                        log=data["tool"],
                    )
                    self._call_data[data["run_id"]] = class_status

                elif data["status"] in [AgentStatus.tool_end]:
                    last_status: AllToolsAction = self._call_data[data["run_id"]]
                    class_status = AllToolsActionToolEnd(
                        run_id=data["run_id"],
                        tool_input=last_status.tool_input,
                        tool=last_status.tool,
                        log=last_status.log,
                        tool_output=data["tool_output"],
                        is_error=data.get("is_error", False),
                    )
                elif data["status"] == AgentStatus.agent_finish:
                    class_status = AllToolsFinish(
                        run_id=data["run_id"],
                        **data["finish"],
                    )

                yield class_status

            await task

        return chat_iterator()
