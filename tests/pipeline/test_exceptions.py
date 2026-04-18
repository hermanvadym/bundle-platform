from bundle_platform.pipeline.exceptions import RagUnavailable


def test_rag_unavailable_is_exception():
    exc = RagUnavailable("embedder failed")
    assert isinstance(exc, Exception)
    assert "embedder failed" in str(exc)
