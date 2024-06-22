from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
)

from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.tools import InvalidTool
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.tools import BaseTool

from langchain_zhipuai.agent_toolkits import AdapterAllTool
from langchain_zhipuai.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)

logger = logging.getLogger(__name__)


class ZhipuAiAllToolsAgentExecutor(AgentExecutor):
    def _perform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            # TODO: platform adapter tool for all  tools,
            #       view tools binding langchain_zhipuai/agents/zhipuai_all_tools/base.py:188
            if "code_interpreter" in agent_action.tool:
                observation = tool.run(
                    {
                        "agent_action": agent_action,
                    },
                    verbose=self.verbose,
                    color="red",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _aperform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            # TODO: platform adapter tool for all  tools,
            #       view tools binding
            #       langchain_zhipuai.agents.zhipuai_all_tools.base.ZhipuAIAllToolsRunnable.paser_all_tools
            if agent_action.tool in AdapterAllToolStructType.__members__.values():
                observation = await tool.arun(
                    {
                        "agent_action": agent_action,
                    },
                    verbose=self.verbose,
                    color="red",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)
