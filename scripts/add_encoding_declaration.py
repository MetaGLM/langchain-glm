# -*- coding: utf-8 -*-
import os


def add_encoding_declaration(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r+", encoding="utf-8") as f:
                    content = f.read()
                    if not content.startswith("# -*- coding: utf-8 -*-"):
                        f.seek(0, 0)
                        f.write("# -*- coding: utf-8 -*-\n" + content)


if __name__ == "__main__":
    # 使用你的项目路径
    project_directory = "/media/gpt4-pdf-chatbot-langchain/langchain-zhipuai"
    add_encoding_declaration(project_directory)
