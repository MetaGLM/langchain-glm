import asyncio
import threading
from datetime import datetime
import os
import uuid
import json
from typing import List, Dict, AsyncIterable, AsyncIterator, Optional
from langchain_core.runnables.utils import (
    AddableDict,
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldSpec,
    Input,
    Output,
    accepts_config,
    accepts_context,
    accepts_run_manager,
    create_model,
    gather_with_concurrency,
    get_function_first_arg_dict_keys,
    get_function_nonlocals,
    get_lambda_source,
    get_unique_config_specs,
    indent_lines_after_first,
)

from concurrent.futures import FIRST_COMPLETED, wait
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    get_executor_for_config,
    merge_configs,
    patch_config,
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
import base64
from io import BytesIO

import nest_asyncio
import streamlit as st
import streamlit_antd_components as sac
from streamlit_chatbox import *
from streamlit_extras.bottom_container import bottom

from langchain_zhipuai.agents.zhipuai_all_tools.base import AllToolsAction, AllToolsActionToolStart, AllToolsFinish, \
    AllToolsActionToolEnd, AllToolsLLMStatus, ZhipuAIAllToolsRunnable
from langchain_zhipuai.callbacks.callback_handler.agent_callback_handler import AgentStatus
from tests.assistant.call_collector import IteratorCallbackHandler, collect_results
from tests.assistant.utils import run_sync


def get_img_base64(file_name: str) -> str:
    '''
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    '''
    # 读取图片
    with open(file_name, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"


st.set_page_config(
    "assistant",
    get_img_base64("chatchat_icon_blue_square_v2.png"),
    initial_sidebar_state="expanded",
    menu_items={
    },
    layout="wide"
)

# use the following code to set the app to wide mode and the html markdown to increase the sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebarUserContent"] {
        padding-top: 20px;
    }
    .block-container {
        padding-top: 25px;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 20px;
    }
    """,
    unsafe_allow_html=True,
)

chat_box = ChatBox(
    assistant_avatar=get_img_base64("chatchat_icon_blue_square_v2.png")
)


def save_session(conv_name: str = None):
    """save session state to chat context"""
    chat_box.context_from_session(conv_name, exclude=["selected_page", "prompt", "cur_conv_name"])


def restore_session(conv_name: str = None):
    """restore sesstion state from chat context"""
    chat_box.context_to_session(conv_name, exclude=["selected_page", "prompt", "cur_conv_name"])


def rerun():
    """
    save chat context before rerun
    """
    save_session()
    st.rerun()


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    """
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    """

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    messages = chat_box.filter_history(history_len=history_len, filter=filter)
    if sys_msg := st.session_state.get("system_message"):
        messages = [{"role": "system", "content": sys_msg}] + messages
    return messages


def add_conv(name: str = ""):
    conv_names = chat_box.get_chat_names()
    if not name:
        i = len(conv_names) + 1
        while True:
            name = f"会话{i}"
            if name not in conv_names:
                break
            i += 1
    if name in conv_names:
        sac.alert("创建新会话出错", f"该会话名称 “{name}” 已存在", color="error", closable=True)
    else:
        chat_box.use_chat_name(name)
        st.session_state["cur_conv_name"] = name


def del_conv(name: str = None):
    conv_names = chat_box.get_chat_names()
    name = name or chat_box.cur_chat_name
    if len(conv_names) == 1:
        sac.alert("删除会话出错", f"这是最后一个会话，无法删除", color="error", closable=True)
    elif not name or name not in conv_names:
        sac.alert("删除会话出错", f"无效的会话名称：“{name}”", color="error", closable=True)
    else:
        chat_box.del_chat_name(name)
        restore_session()
        st.session_state["cur_conv_name"] = chat_box.cur_chat_name


def clear_conv(name: str = None):
    chat_box.reset_history(name=name or None)


def list_tools():
    return {}


DEFULT_LLM = "chatglm3-qingyan-alltools-130b"
HISTORY_LEN = 10


async def dialogue_page(
        is_lite: bool = False,
):
    ctx = chat_box.context
    ctx.setdefault("uid", uuid.uuid4().hex)
    ctx.setdefault("file_chat_id", None)
    ctx.setdefault("llm_model", DEFULT_LLM)
    ctx.setdefault("temperature", "0.7")
    st.session_state.setdefault("cur_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("last_conv_name", chat_box.cur_chat_name)

    # sac on_change callbacks not working since st>=1.34
    if st.session_state.cur_conv_name != st.session_state.last_conv_name:
        save_session(st.session_state.last_conv_name)
        restore_session(st.session_state.cur_conv_name)
        st.session_state.last_conv_name = st.session_state.cur_conv_name

    @st.experimental_dialog("重命名会话")
    def rename_conversation():
        name = st.text_input("会话名称")
        if st.button("OK"):
            chat_box.change_chat_name(name)
            restore_session()
            st.session_state["cur_conv_name"] = name
            rerun()

    with st.sidebar:
        tab1, _ = st.tabs(["会话设置", "test"])

        with tab1:
            # 会话
            cols = st.columns(3)
            conv_names = chat_box.get_chat_names()

            def on_conv_change():
                print(conversation_name, st.session_state.cur_conv_name)
                save_session(conversation_name)
                restore_session(st.session_state.cur_conv_name)

            conversation_name = sac.buttons(conv_names, label="当前会话：", key="cur_conv_name",
                                            on_change=on_conv_change, )
            chat_box.use_chat_name(conversation_name)
            conversation_id = chat_box.context["uid"]
            if cols[0].button("新建", on_click=add_conv):
                ...
            if cols[1].button("重命名"):
                rename_conversation()
            if cols[2].button("删除", on_click=del_conv):
                ...

    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "请输入对话内容 "

    # chat input
    with bottom():
        cols = st.columns([1, 1, 15])
        prompt = cols[2].chat_input(chat_input_placeholder, key="prompt")
    if prompt:
        history = get_messages_history(
            HISTORY_LEN
        )
        chat_box.user_say(prompt)

        chat_box.ai_say("正在思考...")
        agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
            model_name="chatglm3-qingyan-alltools-130b",
            history=history,
            tools=[
                {
                    "type": "code_interpreter"
                }
            ]
        )
        text = ""
        message_id = uuid.uuid4().hex

        metadata = {
            "message_id": message_id,
        }
        chat_iterator = agent_executor.invoke(chat_input=prompt)
        async for item in chat_iterator:
            chat_box.update_msg("", streaming=False)
            if isinstance(item, AllToolsAction):
                chat_box.insert_msg(f"正在解读{item.tool}工具输出结果...")

            elif isinstance(item, AllToolsFinish):
                chat_box.insert_msg(item.log.replace("\n", "\n\n"))

            elif isinstance(item, AllToolsActionToolStart):
                formatted_data = {
                    "Function": item.tool,
                    "function_input": item.tool_input
                }
                formatted_json = json.dumps(formatted_data, indent=2, ensure_ascii=False)
                text = """\n```{}\n```\n""".format(formatted_json)
                chat_box.insert_msg(  # TODO: insert text directly not shown
                    Markdown(text, title="Function call", in_expander=True, expanded=True, state="running"))

            elif isinstance(item, AllToolsActionToolEnd):

                text = """\n```\nObservation:\n{}\n```\n""".format(item.tool_output)
                chat_box.update_msg(text, streaming=False, expanded=False, state="complete")

            elif isinstance(item, AllToolsLLMStatus):

                if item.status == AgentStatus.error:
                    st.error(item.text)
                elif item.status == AgentStatus.chain_start:
                    chat_box.insert_msg(f"")
                elif item.status == AgentStatus.llm_start:
                    text = item.text or ""
                    chat_box.insert_msg(text.replace("\n", "\n\n"))

                elif item.status == AgentStatus.llm_new_token:
                    text += item.text
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True, metadata=metadata)
                elif item.status == AgentStatus.llm_end:
                    text += item.text
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=False, state="complete")
                elif item.status == AgentStatus.chain_end:
                    text += item.text
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=False, state="complete")
                else:
                    st.write(item.text)

        chat_box.update_msg(text, streaming=False, metadata=metadata)

    now = datetime.now()
    with tab1:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )

if __name__ == "__main__":
    run_sync(dialogue_page)
