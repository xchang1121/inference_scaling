"""Eight short native requests: two old TRAIN prompts, collection off/on twice."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time

from benchmark_is_gpu import engine_settings, reset_cache
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import load_backend, close_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.retrieval.cot_guidance import DualStatistics
from inference_scaling.retrieval.internal_signals import InternalSignals, select_signal_worker


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--settings', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hidden', action='store_true')
    p.add_argument('--attention', action='store_true')
    a = p.parse_args()
    from test_internal_signals import check_tensor_truth
    check_tensor_truth()
    settings = load_settings(a.settings)
    cases = json.loads(a.prepared.read_text())['cases'][:2]
    assert len(cases) == 2 and all(c['id'].startswith('TRAIN_') for c in cases)
    select_signal_worker(hidden=a.hidden,attention=a.attention)
    raw = tap = signals = None
    with a.output.open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            started = time.perf_counter()
            if a.hidden or a.attention:
                settings['ar']['engine']['vllm']['engine_kwargs']['max_logprobs'] = 160 if a.hidden else 64
            raw = load_backend(settings['ar']['model'], settings['ar']['engine'], seed=20261115, logprobs=12)
            signals = InternalSignals(raw, hidden_dimensions=128 if a.hidden else 0,
                                      attention_topk=16 if a.attention else 0)
            tap = DualStatistics(raw)
            close = raw.tokenizer.convert_tokens_to_ids('</think>')
            write({'kind': 'engine', 'model_id': raw.model_id, 'settings': engine_settings(raw),
                   'load_seconds': time.perf_counter()-started,
                   'distribution': 'processed sampling distribution, full support, temperature=1',
                   'hidden_features': {'dimensions': 128, 'projection_seed': 2026111502,
                                       'layer': 'final normalized hidden state before LM head'} if a.hidden else None,
                   'attention_features': {'layer':63,'topk':16,'head_pooling':'mean of all heads',
                       'source':'native post-RoPE Q, full paged K/V; P@V checked against actual attention output',
                       'capture_copies':'present in both OFF and ON controls'} if a.attention else None,
                   'source': {str(f.relative_to(Path(__file__).resolve().parents[1])):
                              hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
            baseline = {}
            layout_effects = []
            for repeat in range(2):
                # AB/BA order limits a systematic cold-start advantage.
                # Hidden permutation checks use a matched off/on control in EACH
                # layout; native BF16 packing itself may change across layouts.
                for enabled in ((False, True) if repeat == 0 or a.hidden or a.attention else (True, False)):
                    signals.enabled = enabled
                    reset_cache(raw)
                    policy = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                    ordered_cases = cases[::-1] if (a.hidden or a.attention) and repeat == 1 else cases
                    jobs = [GenerationRequest(tuple(c['prompt_ids']), 8 if a.attention else 64, policy, 20261115+cases.index(c),
                             f'signal-check:{repeat}:{enabled}:{c["id"]}',
                             statistic=TokenStatistic(policy, top_k=5), stop_sequences=((close,),),
                             intermediate_events=False) for c in ordered_cases]
                    b0 = asdict(raw.snapshot())
                    started = time.perf_counter()
                    samples = raw.sample_batch(jobs)
                    elapsed = time.perf_counter()-started
                    b1 = asdict(raw.snapshot())
                    completed = []
                    for case, job, sample in zip(ordered_cases, jobs, samples, strict=True):
                        sc, cn = tap.take(job, sample)
                        values = signals.take(job, sample) if enabled else ()
                        row = {'kind': 'measurement', 'case': case['id'], 'enabled': enabled,
                               'repeat': repeat, 'batch_seconds': elapsed, 'request': asdict(job),
                               'sample': asdict(sample), 'sc': sc, 'cn': cn, 'signals': values,
                               'batch_cost': {k: b1[k]-b0[k] for k in b0
                                              if type(b0[k]) in (int, float)}}
                        write(row)
                        completed.append((case, sample, sc, cn, values))
                    # Persist every output from an executed batch before validation;
                    # even a failed assertion must retain its sibling's native cost.
                    for case, sample, sc, cn, values in completed:
                        identity = (tuple(sample.token_ids), tuple(sample.token_logprobs), sc, cn)
                        control_key = (case['id'], repeat if a.hidden or a.attention else 0)
                        if control_key in baseline:
                            expected = baseline[control_key]
                            assert identity[0] == expected[0], 'diagnostics altered native sampled tokens'
                            for left, right in zip(identity[1:], expected[1:], strict=True):
                                assert len(left) == len(right)
                                assert all(math.isclose(x, y, abs_tol=1e-6, rel_tol=1e-6)
                                           for x, y in zip(left, right, strict=True)), 'LP/SC/CN changed'
                        else:
                            baseline[control_key] = identity
                        if a.hidden and enabled:
                            feature_key = case['id']+':features'
                            features = [v['hidden_projection'] for v in values]
                            assert all(len(f) == 128 for f in features)
                            if feature_key in baseline:
                                expected = baseline[feature_key]
                                squared = sum((x-y)**2 for f,g in zip(features,expected,strict=True)
                                              for x,y in zip(f,g,strict=True))
                                magnitude = sum(y*y for g in expected for y in g)
                                first = baseline[case['id'],0]
                                layout_effects.append({'case':case['id'],
                                    'native_tokens_equal':identity[0] == first[0],
                                    'native_lp_max_difference':max(abs(x-y) for x,y in zip(identity[1],first[1],strict=True)),
                                    'native_sc_max_difference':max(abs(x-y) for x,y in zip(identity[2],first[2],strict=True)),
                                    'hidden_relative_l2':math.sqrt(squared/max(1e-12,magnitude))})
                            else:
                                baseline[feature_key] = features
            assert not tap.rows and not signals.rows and not signals.pending
            write({'kind': 'complete', 'requests': 8, 'pass': True,
                   'validation': 'instrumentation off/on matched within each batch layout',
                   'cross_layout_effects':layout_effects})
        except BaseException as error:
            write({'kind': 'failed', 'error': repr(error)})
            raise
        finally:
            if tap is not None:
                tap.close()
            if signals is not None:
                signals.close()
            if raw is not None:
                close_backend(raw)


if __name__ == '__main__':
    main()
