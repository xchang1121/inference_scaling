"""Eight short TRAIN requests validate real multi-layer graph capture."""
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
from inference_scaling.retrieval.internal_signals import InternalSignals


PAYLOAD_DIMENSIONS = 513
SCHEMA = {'version': 1, 'block_dimensions': 128, 'blocks': [
    {'source': 'input_layernorm_output', 'layer': layer} for layer in (16, 32, 48)] +
    [{'source': 'final_normalized_pre_lm_head'}], 'representation_dimensions': 512,
    'projection_seed': 2026111502, 'same_projection_for_each_real_block': True,
    'alignment_diagnostic_index': 512, 'state_offset': -1,
    'capture_copies': 'compiled graph operations present in OFF and ON'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('settings', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    cases = json.loads(args.prepared.read_text())['cases'][:2]
    assert len(cases) == 2 and all(c['id'].startswith('TRAIN_') for c in cases)
    from inference_scaling.arllm.backends import vllm_backend
    vllm_backend._ASCEND_WORKER = 'inference_scaling.retrieval.multilayer_signals_worker.MultilayerSignalWorker'
    settings['ar']['engine']['vllm']['engine_kwargs']['max_logprobs'] = 600
    raw = tap = signals = None
    controls, native_checks, feature_checks = {}, [], []
    with args.output.open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            started = time.perf_counter()
            raw = load_backend(settings['ar']['model'], settings['ar']['engine'], seed=20261115, logprobs=12)
            signals = InternalSignals(raw, hidden_dimensions=PAYLOAD_DIMENSIONS)
            tap = DualStatistics(raw)
            close = raw.tokenizer.convert_tokens_to_ids('</think>')
            write({'kind': 'engine', 'model_id': raw.model_id, 'settings': engine_settings(raw),
                'load_seconds': time.perf_counter() - started, 'hidden_features': SCHEMA,
                'distribution': 'processed sampling distribution, full support, temperature=1',
                'source': {str(f.relative_to(Path(__file__).resolve().parents[1])):
                    hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
            for repeat in range(2):
                ordered = cases if repeat == 0 else cases[::-1]
                for enabled in (False, True):
                    signals.enabled = enabled
                    reset_cache(raw)
                    policy = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                    requests = [GenerationRequest(tuple(c['prompt_ids']), 64, policy,
                        20261115 + cases.index(c), f'multilayer:{repeat}:{enabled}:{c["id"]}',
                        statistic=TokenStatistic(policy, top_k=5), stop_sequences=((close,),),
                        intermediate_events=False) for c in ordered]
                    before = asdict(raw.snapshot())
                    started = time.perf_counter()
                    samples = raw.sample_batch(requests)
                    seconds = time.perf_counter() - started
                    after = asdict(raw.snapshot())
                    completed = []
                    for case, request, sample in zip(ordered, requests, samples, strict=True):
                        sc, cn = tap.take(request, sample)
                        values = signals.take(request, sample) if enabled else ()
                        row = {'kind': 'measurement', 'case': case['id'], 'repeat': repeat,
                            'enabled': enabled, 'batch_seconds': seconds, 'request': asdict(request),
                            'sample': asdict(sample), 'sc': sc, 'cn': cn, 'signals': values,
                            'batch_cost': {k: after[k] - before[k] for k in before
                                if type(before[k]) in (int, float)}}
                        write(row)
                        completed.append((case, sample, sc, cn, values))
                    # Persist both siblings before checking any native or layer invariant.
                    for case, sample, sc, cn, values in completed:
                        key = (case['id'], repeat)
                        identity = (tuple(sample.token_ids), tuple(sample.token_logprobs), tuple(sc), tuple(cn))
                        if not enabled:
                            controls[key] = identity
                            continue
                        expected = controls[key]
                        native = {'case': case['id'], 'repeat': repeat,
                            'tokens_exact': identity[0] == expected[0],
                            'lp_sc_cn_exact': identity[1:] == expected[1:],
                            'max_lp_sc_cn_differences': [max((abs(x-y) for x,y in zip(left,right,strict=True)), default=0.)
                                for left,right in zip(identity[1:], expected[1:], strict=True)]}
                        native_checks.append(native)
                        assert native['tokens_exact'], 'capture altered sampled tokens'
                        assert all(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-6)
                            for left,right in zip(identity[1:],expected[1:],strict=True)
                            for x,y in zip(left,right,strict=True)), 'capture altered LP/SC/CN'
                        assert values and len(values) == len(sample.token_ids)
                        features = [v['hidden_projection'] for v in values]
                        assert all(len(f) == PAYLOAD_DIMENSIONS for f in features)
                        errors = [f[-1] for f in features]
                        check = {'case': case['id'], 'repeat': repeat, 'rows': len(features),
                            'max_final_buffer_relative_l2': max(errors),
                            'layer_vs_final_relative_l2': []}
                        for block in range(3):
                            numerator = math.fsum((f[block*128+i]-f[384+i])**2 for f in features for i in range(128))
                            scale = math.fsum(f[384+i]**2 for f in features for i in range(128))
                            check['layer_vs_final_relative_l2'].append(math.sqrt(numerator/max(scale,1e-20)))
                        feature_checks.append(check)
                        write({'kind': 'validation', 'native': native, 'features': check})
                        assert all(error == 0. for error in errors), 'stale or misindexed final hidden buffer'
                        assert all(v > 1e-6 for v in check['layer_vs_final_relative_l2']), 'middle block duplicates final representation'
                        assert all(any(abs(f[block*128+i]) > 0. for f in features for i in range(128)) for block in range(4))
            assert not tap.rows and not signals.rows and not signals.pending
            write({'kind': 'complete', 'requests': 8, 'pass': True,
                'native_checks': native_checks, 'feature_checks': feature_checks,
                'limitation': 'short requests only; fullIS capture and learned layer selection not yet validated'})
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
