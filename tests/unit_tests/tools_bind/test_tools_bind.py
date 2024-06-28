from langchain_core.runnables import RunnableBinding

from langchain_zhipuai.agents.zhipuai_all_tools.base import _get_assistants_tool
from langchain_zhipuai.chat_models import ChatZhipuAI

from langchain.agents import tool as register_tool
from langchain.tools.shell import ShellTool
from pydantic.v1 import BaseModel, Extra, Field
from langchain_zhipuai.agent_toolkits import BaseToolOutput


class TestToolsBind():


    def test_tools_bind(self):


        @register_tool
        def shell(query: str = Field(description="The command to execute")):
            """Use Shell to execute system shell commands"""
            tool = ShellTool()
            return BaseToolOutput(tool.run(tool_input=query))

        llm = ChatZhipuAI(api_key="abc") # Create a new instance of the ChatZhipuAI class

        tools = [
            shell,
            {"type": "code_interpreter", "code_interpreter": {"sandbox": "none"}},
            {"type": "web_browser"},
            {"type": "drawing_tool"},
        ]
        dict_tools = [_get_assistants_tool(tool) for tool in tools]
        assert isinstance(dict_tools, list)
        assert isinstance(dict_tools[0], dict)
        assert isinstance(dict_tools[1], dict)
        assert isinstance(dict_tools[2], dict)
        assert isinstance(dict_tools[3], dict)
        llm_with_all_tools = llm.bind(
            tools=dict_tools
        )

        assert llm_with_all_tools is not None
        assert isinstance(llm_with_all_tools, RunnableBinding)
        self.llm_with_all_tools = llm_with_all_tools

    def test_create_zhipuai_tools_agent(self):
        
        create_zhipuai_tools_agent(

        )
