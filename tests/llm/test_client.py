from dataclasses import dataclass

from bundle_platform.llm.client import LLMClient, LLMResponse, LLMUsage


def test_llm_response_dataclass_fields():
    usage = LLMUsage(input_tokens=10, output_tokens=5)
    resp = LLMResponse(content=[], stop_reason="end_turn", usage=usage)
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    assert resp.usage.cache_creation_tokens == 0


def test_llm_client_is_runtime_checkable_protocol():
    @dataclass
    class Fake:
        model_id: str = "fake"
        def complete(self, system, messages, tools, max_tokens=4096):
            return LLMResponse(content=[], stop_reason="end_turn",
                               usage=LLMUsage(input_tokens=0, output_tokens=0))
    assert isinstance(Fake(), LLMClient)
