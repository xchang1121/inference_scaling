"""Eight short native requests check genuine layer capture in paid auxiliary pools."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from benchmark_is_gpu import engine_settings, reset_cache
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.retrieval.auxiliary_signals import AuxiliarySampler, hidden_disagreement
from inference_scaling.retrieval.cot_guidance import DualStatistics, GuidanceConfig
from inference_scaling.retrieval.internal_signals import InternalSignals
from inference_scaling.retrieval.multilayer_capture import SCHEMA
from inference_scaling.retrieval.native_journal import NativeJournal
from inference_scaling.retrieval.tuned_online_is import TuningTap
from inference_scaling.shared.rng import SeedStream


def validate_pool(calls):
    """Diagnostic-only views; never pass concatenated layers as a final-state score."""
    assert len(calls) == 2
    assert len({c['prefix_sha256'] for c in calls}) == 1
    assert len({c['request']['seed'] for c in calls}) == 2
    for c in calls:
        assert 0 < len(c['token_ids']) <= 32
        assert len(c['internal_signals']) == len(c['token_ids'])
        for signal in c['internal_signals']:
            v = signal['hidden_projection']
            assert len(v) == 513 and v[512] == 0 and all(math.isfinite(x) for x in v)
            assert all(any(x != 0 for x in v[start:start+128]) for start in (0, 128, 256, 384))
        vectors = [s['hidden_projection'] for s in c['internal_signals']]
        assert all(any(v[start:start+128] != v[384:512] for v in vectors) for start in (0, 128, 256))
        if len(vectors) > 1:
            assert all(len({tuple(v[start:start+128]) for v in vectors}) > 1 for start in (0, 128, 256, 384))
    result = {}
    for block, layer in enumerate((16, 32, 48, 'final')):
        view = copy.deepcopy(calls)
        for c in view:
            for signal in c['internal_signals']:
                signal['hidden_projection'] = signal['hidden_projection'][128*block:128*(block+1)]
        result[str(layer)] = {str(tail): hidden_disagreement(view, tail=tail) for tail in (1, 4, 16)}
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('settings', 'prepared', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    prepared = a.prepared.read_bytes()
    cases = json.loads(prepared)['cases'][:2]
    assert len(cases) == 2 and len({c['id'] for c in cases}) == 2
    assert all(c['id'].startswith('TRAIN_') for c in cases)
    settings = load_settings(a.settings)
    from inference_scaling.arllm.backends import vllm_backend
    vllm_backend._ASCEND_WORKER = 'inference_scaling.retrieval.multilayer_signals_worker.MultilayerSignalWorker'
    settings['ar']['engine']['vllm']['engine_kwargs']['max_logprobs'] = 600
    raw = dual = signals = journal = None
    with a.output.open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            started = time.perf_counter()
            raw = load_backend(settings['ar']['model'], settings['ar']['engine'], seed=20261115, logprobs=6)
            signals = InternalSignals(raw, hidden_dimensions=513)
            dual = DualStatistics(raw)
            tap = TuningTap(dual, signals)
            journal = NativeJournal(raw, dual, a.output.with_name(a.output.stem+'-native.jsonl'))
            close = (raw.tokenizer.convert_tokens_to_ids('</think>'),)
            write({'kind': 'engine', 'model_id': raw.model_id, 'settings': engine_settings(raw),
                'load_seconds': time.perf_counter()-started, 'schema': SCHEMA,
                'prepared_sha256': hashlib.sha256(prepared).hexdigest(), 'requests_planned': 8,
                'sampling': 'Existing AuxiliarySampler defaults; two independent seeds per same prefix; no IS or quality policy.',
                'source': {str(f.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
            proofs = []
            for case in cases:
                control = None
                seed = SeedStream(SeedStream(20261115).derive('question', case['id'])).derive('search')
                for enabled in (False, True):
                    signals.enabled = enabled
                    tap.history.clear()
                    reset_cache(raw)
                    journal.unit = f'{case["id"]}:auxiliary-multilayer:{enabled}'
                    sampler = AuxiliarySampler(raw, tap, seed, close,
                        context_tokens=GuidanceConfig().context_tokens, work_tokens=64)
                    calls = sampler.sample(tuple(case['prompt_ids']), (), purpose='seakr', count=2, tokens=32)
                    # Persist both independent siblings before validating any numerical equality.
                    write({'kind': 'measurement', 'case': case['id'], 'enabled': enabled,
                           'auxiliary_calls': calls, 'auxiliary_work': sampler.work})
                    assert calls is not None and len(calls) == 2
                    assert sum(len(c['token_ids']) for c in calls) == sampler.work
                    identity = [{k: c[k] for k in ('request', 'prefix_ids', 'token_ids', 'lp', 'sc', 'cn', 'finish_reason')}
                                for c in calls]
                    if not enabled:
                        assert all(c['internal_signals'] is None for c in calls)
                        control = identity
                    else:
                        layer_metrics = validate_pool(calls)
                        proof = {'case': case['id'], 'native_exact': identity == control,
                                 'layer_metrics': layer_metrics, 'actual_tokens': sampler.work}
                        proofs.append(proof)
                        write({'kind': 'validation', **proof})
            assert not dual.rows and not signals.rows and not signals.pending
            write({'kind': 'complete', 'requests': 8, 'passed': all(p['native_exact'] for p in proofs),
                   'proofs': proofs, 'scope': 'Two-sample short geometry check only; no four-sample calibration or full-IS efficacy claim.'})
            assert all(p['native_exact'] for p in proofs), 'preserved auxiliary capture native difference'
        except BaseException as error:
            write({'kind': 'failed', 'error': repr(error)})
            raise
        finally:
            if journal is not None:
                journal.close()
            if dual is not None:
                dual.close()
            if signals is not None:
                signals.close()
            if raw is not None:
                close_backend(raw)


if __name__ == '__main__':
    main()
