"""Test openai embeddings."""
import numpy as np
import pytest

from langchain_zhipuai.embeddings.base import ZhipuAIAIEmbeddings


@pytest.mark.scheduled
def test_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = ZhipuAIAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


@pytest.mark.scheduled
def test_openai_embedding_documents_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = ZhipuAIAIEmbeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


@pytest.mark.scheduled
async def test_openai_embedding_documents_async_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = ZhipuAIAIEmbeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


@pytest.mark.scheduled
def test_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = ZhipuAIAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024


@pytest.mark.scheduled
async def test_openai_embedding_async_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = ZhipuAIAIEmbeddings()
    output = await embedding.aembed_query(document)
    assert len(output) == 1024


@pytest.mark.scheduled
def test_embed_documents_normalized() -> None:
    output = ZhipuAIAIEmbeddings().embed_documents(["foo walked to the market"])
    assert np.isclose(np.linalg.norm(output[0]), 1.0)


@pytest.mark.scheduled
def test_embed_query_normalized() -> None:
    output = ZhipuAIAIEmbeddings().embed_query("foo walked to the market")
    assert np.isclose(np.linalg.norm(output), 1.0)