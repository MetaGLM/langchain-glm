from langchain_zhipuai.agent_toolkits.all_tools.code_interpreter_tool import CodeInterpreterAllToolExecutor


def test_python_ast_interpreter():
    out = CodeInterpreterAllToolExecutor._python_ast_interpreter(
        code_input="print('Hello, World!')"
    )
    print(out.data)
    assert out.data != """Access：code_interpreter,python_repl_ast, Message: print('Hello, World!')
Hello, World!
"""
