from dataclasses import replace
from types import SimpleNamespace

from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.cot_guidance import (
    GuidanceConfig, finish_answer,
)


class Fake:
    model_id = 'fake'
    tokenizer = SimpleNamespace(eos_token_id=99)
    def __init__(self): self.requests = []
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)
    def sample_batch(self, requests):
        self.requests.extend(requests)
        return [SequenceSample(tuple(r.prefix),(65,66,90),(-1.,-2.,-3.),r.sampling.policy_id,self.model_id,r.request_id, 'stop') for r in requests]


def test_answer_continues_after_length_stop_without_closing_it_again():
    class Answer(Fake):
        def sample_batch(self,requests):
            r,=requests
            self.requests.append(r)
            ids=(65,66) if len(self.requests)==1 else (67,99)
            return [SequenceSample(r.prefix,ids,(-1.,)*2,r.sampling.policy_id,self.model_id,r.request_id,'length' if len(self.requests)==1 else 'eos')]
    raw=Answer()
    result=finish_answer(raw,(1,),(2,90),(90,),replace(GuidanceConfig(),answer_chunk=2,answer_tokens=8),1)
    assert not result['truncated'] and not result['forced_thinking_end']
    assert raw.requests[1].prefix==(1,2,90,65,66)


def test_length_limit_is_reported_as_failure_not_natural_eos():
    raw=Fake()
    # Fake reports stop; an answer EOS in thinking must never be repaired into a normal answer.
    result=finish_answer(raw,(1,),(65,99),(90,),GuidanceConfig(),1)
    assert result['eos_inside_thinking'] and result['empty_answer'] and not raw.requests
