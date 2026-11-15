"""Eight one-token TRAIN recaptures; never regenerate or replace outcome labels."""
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
from inference_scaling.retrieval.multilayer_capture import SCHEMA
from inference_scaling.retrieval.stepwise import token_hash


def main():
    p = argparse.ArgumentParser()
    for name in ('settings', 'prepared', 'output', 'capture-plan'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.capture_plan.read_text())
    assert plan['kind'] == 'multilayer_prefix_recapture_pilot' and plan['repeats'] == 2 and len(plan['items']) == 4
    assert all(i['question_id'].startswith('TRAIN_') and token_hash(i['prefix_ids']) == i['prefix_sha256'] for i in plan['items'])
    settings = load_settings(a.settings)
    from inference_scaling.arllm.backends import vllm_backend
    vllm_backend._ASCEND_WORKER = 'inference_scaling.retrieval.multilayer_signals_worker.MultilayerSignalWorker'
    settings['ar']['engine']['vllm']['engine_kwargs']['max_logprobs'] = 600
    raw = tap = signals = None
    with a.output.open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        write({'kind':'metadata', 'plan':plan, 'plan_sha256':hashlib.sha256(a.capture_plan.read_bytes()).hexdigest(),
            'schema':SCHEMA, 'source':{str(f.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
        try:
            started = time.perf_counter()
            raw = load_backend(settings['ar']['model'], settings['ar']['engine'], seed=20261115, logprobs=6)
            signals = InternalSignals(raw, hidden_dimensions=513)
            signals.enabled = True
            tap = DualStatistics(raw)
            write({'kind':'engine', 'model_id':raw.model_id, 'settings':engine_settings(raw), 'load_seconds':time.perf_counter()-started})
            for repeat in range(2):
                ordered = plan['items'] if repeat == 0 else plan['items'][::-1]
                reset_cache(raw)
                sampling = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                requests = [GenerationRequest(tuple(item['prefix_ids']), 1, sampling, item['seed'],
                    f'recapture:{repeat}:{item["prefix_sha256"]}', uniform_offset=item['uniform_offset'],
                    statistic=TokenStatistic(sampling, top_k=5), intermediate_events=False) for item in ordered]
                before = asdict(raw.snapshot())
                started = time.perf_counter()
                samples = raw.sample_batch(requests)
                seconds = time.perf_counter()-started
                after = asdict(raw.snapshot())
                captured = []
                for item, request, sample in zip(ordered, requests, samples, strict=True):
                    sc, cn = tap.take(request, sample)
                    values = signals.take(request, sample)
                    write({'kind':'measurement', 'question_id':item['question_id'], 'prefix_sha256':item['prefix_sha256'],
                        'repeat':repeat, 'enabled':True, 'request':asdict(request), 'sample':asdict(sample),
                        'sc':sc, 'cn':cn, 'signals':values, 'batch_seconds':seconds,
                        'batch_cost':{k:after[k]-before[k] for k in before if type(before[k]) in (int,float)}})
                    captured.append((item, sample, values))
                # Preserve all raw sibling rows before checking feature diagnostics.
                for item, sample, values in captured:
                    assert len(sample.token_ids) == len(values) == 1
                    vector = values[0]['hidden_projection']
                    assert len(vector) == 513 and all(math.isfinite(v) for v in vector) and vector[-1] == 0
                    reference = item['original_hidden']
                    difference = math.sqrt(math.fsum((v-w)**2 for v,w in zip(vector[384:512],reference,strict=True)))
                    scale = math.sqrt(math.fsum(v*v for v in reference))
                    write({'kind':'validation', 'prefix_sha256':item['prefix_sha256'], 'repeat':repeat,
                        'final_relative_l2_from_original':difference/max(scale,1e-20),
                        'original_token_repeated':sample.token_ids[0] == item['original_token_id'],
                        'note':'Different prefill/batch layout; differences are measured, not treated as new labels.'})
            assert not signals.rows and not signals.pending and not tap.rows
            write({'kind':'complete', 'pass':True, 'requests':8, 'labels_added':0, 'independent_groups_added':0})
        except BaseException as error:
            write({'kind':'failed', 'error':repr(error)})
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
