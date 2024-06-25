from zhipuai.core import BaseModel
from typing import Optional, Dict, Any


class AllToolsMessageToolCall(BaseModel):
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    id: Optional[str]


class AllToolsMessageToolCallChunk(BaseModel):
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    id: Optional[str]
    index: Optional[int]
