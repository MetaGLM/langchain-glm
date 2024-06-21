# ruff: noqa: E402
"""Main entrypoint into package."""
from langchain_zhipuai.agents import ZhipuAIAllToolsRunnable
from importlib import metadata
try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)



__all__ = [
    "ZhipuAIAllToolsRunnable",
]

