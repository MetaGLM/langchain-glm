from __future__ import annotations
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#
# # fix platform adapter tool for all tools.  with langchain.agents.agent.AgentExecutor
# def _perform_agent_action(
#         self,
#         name_to_tool_map: Dict[str, BaseTool],
#         color_mapping: Dict[str, str],
#         agent_action: AgentAction,
#         run_manager: Optional[CallbackManagerForChainRun] = None,
# ) -> AgentStep:
#     """Perform an agent action. platform adapter tool for all tools."""
#     if run_manager:
#         run_manager.on_agent_action(agent_action, color="green")
#     # Otherwise we lookup the tool
#     if agent_action.tool in name_to_tool_map:
#         tool = name_to_tool_map[agent_action.tool]
#         return_direct = tool.return_direct
#         color = color_mapping[agent_action.tool]
#         tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#         if return_direct:
#             tool_run_kwargs["llm_prefix"] = ""
#         # We then call the tool on the tool input to get an observation
#         observation = tool.run(
#             agent_action.tool_input,
#             verbose=self.verbose,
#             color=color,
#             callbacks=run_manager.get_child() if run_manager else None,
#             **tool_run_kwargs,
#         )
#     else:
#         tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#
#         adapter_all_tools_ = AdapterAllTool(name=agent_action.tool).copy(update={"callbacks": self.callbacks})
#
#         observation = adapter_all_tools_.run(
#             {
#                 "tool": agent_action.tool,
#                 "tool_input": agent_action.tool_input,
#                 "log": agent_action.log,
#             },
#             verbose=self.verbose,
#             color=None,
#             callbacks=run_manager.get_child() if run_manager else None,
#             **tool_run_kwargs,
#         )
#     return AgentStep(action=agent_action, observation=observation)
#
#
# # fix platform adapter tool for all tools.  with langchain.agents.agent.AgentExecutor
# async def _aperform_agent_action(
#         self,
#         name_to_tool_map: Dict[str, BaseTool],
#         color_mapping: Dict[str, str],
#         agent_action: AgentAction,
#         run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
# ) -> AgentStep:
#     """Perform an agent action. platform adapter tool for all tools."""
#     if run_manager:
#         await run_manager.on_agent_action(
#             agent_action, verbose=self.verbose, color="green"
#         )
#     # Otherwise we lookup the tool
#     if agent_action.tool in name_to_tool_map:
#         tool = name_to_tool_map[agent_action.tool]
#         return_direct = tool.return_direct
#         color = color_mapping[agent_action.tool]
#         tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#         if return_direct:
#             tool_run_kwargs["llm_prefix"] = ""
#         # We then call the tool on the tool input to get an observation
#         observation = await tool.arun(
#             agent_action.tool_input,
#             verbose=self.verbose,
#             color=color,
#             callbacks=run_manager.get_child() if run_manager else None,
#             **tool_run_kwargs,
#         )
#     else:
#         tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#
#         adapter_all_tools_ = AdapterAllTool(name=agent_action.tool).copy(update={"callbacks": self.callbacks})
#
#         observation = await adapter_all_tools_.arun(
#             {
#                 "tool": agent_action.tool,
#                 "tool_input": agent_action.tool_input,
#                 "log": agent_action.log,
#             },
#             verbose=self.verbose,
#             color="red",
#             callbacks=run_manager.get_child() if run_manager else None,
#             **tool_run_kwargs,
#         )
#     return AgentStep(action=agent_action, observation=observation)
