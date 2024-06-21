import importlib
import sys
import types

# 动态导入 模块
chatchat = importlib.import_module('langchain_zhipuai')

# 创建新的模块对象
module = types.ModuleType('langchain_zhipu')
sys.modules['langchain_zhipu'] = module

# 把 a_chatchat 的所有属性复制到 langchain_chatchat
for attr in dir(chatchat):
    if not attr.startswith('_'):
        setattr(module, attr, getattr(chatchat, attr))


# 动态导入子模块
def import_submodule(name):
    full_name = f'langchain_zhipuai.{name}'
    submodule = importlib.import_module(full_name)
    sys.modules[f'langchain_zhipu.{name}'] = submodule
    for attr in dir(submodule):
        if not attr.startswith('_'):
            setattr(module, attr, getattr(submodule, attr))


# 需要的子模块列表，自己添加
submodules = ['configs', 'server',
              'startup', 'webui_pages'
              ]

# 导入所有子模块
for submodule in submodules:
    import_submodule(submodule)
