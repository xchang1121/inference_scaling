"""From-start IS with candidate-local online evidence; no controller prefix."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from benchmark_flare_gap import clean, prepare
from benchmark_is_gpu import GraphMeasurements, engine_settings, reset_cache
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Retriever
from inference_scaling.retrieval.cot_guidance import DualStatistics, GuidanceConfig, finish_answer
from inference_scaling.retrieval.literature import FactBank
from inference_scaling.retrieval.online_is import OnlineRetrievalConfig, run_online_is
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream

ARMS = ('direct', 'closed_is', 'online_is', 'online_is_noise')


def run_arm(raw, tap, facts, case, run_seed, arm, cfg, close, noise,
            retrieval_cfg=OnlineRetrievalConfig()):
    if arm not in ARMS:
        raise ValueError(arm)
    base = tuple(case['prompt_ids'])
    seed = SeedStream(run_seed).derive('question', case['id'])
    if arm != 'direct':
        result = run_online_is(raw, tap, base, close, seed, cfg,
            question=case['question'], facts=facts, noise=noise,
            mode={'closed_is': 'closed', 'online_is': 'relevant', 'online_is_noise': 'noise'}[arm],
            retrieval_cfg=retrieval_cfg)
    else:
        started = time.perf_counter()
        request = GenerationRequest(base, cfg.thinking_tokens,
            SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id),
            SeedStream(seed).derive('direct'), f'{seed}:direct',
            stop_sequences=(close,), intermediate_events=False)
        sample, = raw.sample_batch([request])
        thinking_seconds = time.perf_counter()-started
        started = time.perf_counter()
        answer = finish_answer(raw, base, sample.token_ids, close, cfg,
            SeedStream(seed).derive('answer_suffix', token_hash(sample.token_ids)))
        result = {'root_ids': list(base), 'carried': {'sc': [], 'cn': [], 'lp': []},
            'candidates': [sample.token_ids], 'calls': [{
                'prefix_ids': list(base), 'prefix_sha256': token_hash(base),
                'seed': request.seed, 'max_new_tokens': request.max_new_tokens,
                'token_ids': list(sample.token_ids), 'finish_reason': sample.finish_reason}],
            'native_calls': [], 'events': [], 'selected_event_ids': [],
            'selected_physical_ids': list(base+sample.token_ids),
            'presearch_work': 0, 'search_work': len(sample.token_ids),
            'total_generation_tokens': len(sample.token_ids)+len(answer['answer_ids']),
            'answer': answer, 'audit': {'method': 'direct', 'selection': {'direct': 0}},
            'stage_seconds': {'search_seconds': thinking_seconds,
                'materialization_seconds': 0., 'answer_seconds': time.perf_counter()-started}}
    return {'arm': arm, 'question_id': case['id'], 'seed': run_seed,
            'base_ids': list(base), **result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('settings', 'prepared', 'corpus', 'questions', 'noise', 'output'):
        p.add_argument('--'+name, type=Path, required=name in {'settings', 'prepared', 'corpus'})
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--case-ids', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[20261112, 20261113])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--worker', default='unassigned')
    a = p.parse_args()
    settings, cfg, retrieval_cfg = load_settings(a.settings), GuidanceConfig(), OnlineRetrievalConfig()
    if a.prepare_only:
        if a.prepared.exists():
            raise ValueError('prepared data exists')
        prepare(a, settings)
        return
    if a.output is None or a.output.exists():
        raise ValueError('fresh output required')
    prepared = json.loads(a.prepared.read_text(encoding='utf-8'))
    cases = [c for c in prepared['cases'] if not a.case_ids or c['id'] in a.case_ids]
    if not cases or a.case_ids and {c['id'] for c in cases} != set(a.case_ids):
        raise ValueError('invalid case selection')
    import fcntl
    with a.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with a.output.open('x', encoding='utf-8') as stream:
            def write(row):
                stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            source = Path(__file__).resolve().parents[1]
            hashes = {f.relative_to(source).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [*source.joinpath('src').rglob('*.py'), Path(__file__)]}
            write({'kind': 'metadata', 'schema': 1, 'config': asdict(cfg),
                'retrieval_config': asdict(retrieval_cfg), 'worker': a.worker,
                'source': hashes, 'cases': [c['id'] for c in cases], 'seeds': a.seeds,
                'arms': a.arms, 'prepared_sha256': hashlib.sha256(a.prepared.read_bytes()).hexdigest(),
                'settings_sha256': hashlib.sha256(a.settings.read_bytes()).hexdigest()})
            raw = graph = tap = None
            try:
                raw = load_backend(settings['ar']['model'], settings['ar']['engine'], seed=a.seeds[0], logprobs=6)
                graph, tap = GraphMeasurements(raw), DualStatistics(raw)
                fmt, = thinking_format_from_backend(raw, output_settings_from_config(settings['ar'])).formats
                close = tuple(fmt.end_token_ids)
                if len(close) != 1:
                    raise ValueError('this frozen configuration requires a single thinking-close token')
                started = time.perf_counter()
                index = Retriever.from_jsonl(a.corpus, kind='bm25')
                assert index.fingerprint == prepared['corpus_sha256']
                facts = FactBank(raw, index)
                write({'kind': 'engine', 'model_id': raw.model_id, 'config': engine_settings(raw),
                    'fact_build_seconds': time.perf_counter()-started, 'close_ids': close,
                    'eos_id': raw.tokenizer.eos_token_id})
                warm = GenerationRequest(tuple(cases[0]['prompt_ids']), 64,
                    SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id), 20261112,
                    'online-is:warmup', stop_sequences=(close,), intermediate_events=False)
                raw.sample_batch([warm])
                write({'kind': 'warmup', 'complete': True})
                completed = 0
                for qi, case in enumerate(cases):
                    for si, run_seed in enumerate(a.seeds):
                        offset = (qi+si) % len(a.arms)
                        for arm in list(a.arms[offset:])+list(a.arms[:offset]):
                            uid = f"{case['id']}:{run_seed}:{arm}"
                            print(json.dumps({'unit': uid, 'completed': completed}), flush=True)
                            reset_cache(raw)
                            b0, g0 = _snapshot(raw), graph.snapshot()
                            started = time.perf_counter()
                            result = run_arm(raw, tap, facts, case, run_seed, arm, cfg, close,
                                prepared['noise_ids'], retrieval_cfg)
                            elapsed = time.perf_counter()-started
                            b1 = _snapshot(raw)
                            write({'kind': 'measurement', 'unit_id': uid, 'worker': a.worker,
                                'seconds': elapsed, 'cost': {k: b1[k]-b0[k] for k in b0.keys() & b1.keys()
                                    if type(b0[k]) in (int, float) and type(b1[k]) in (int, float)},
                                **graph.difference(g0), **result})
                            if result['answer']['truncated']:
                                raise RuntimeError('final answer truncated: preserve the run')
                            if tap.rows:
                                raise RuntimeError('unconsumed statistics')
                            completed += 1
                write({'kind': 'completed', 'units': completed})
            except BaseException as error:
                write({'kind': 'failed', 'error': repr(error)})
                raise
            finally:
                if tap is not None:
                    tap.close()
                if graph is not None:
                    graph.close()
                if raw is not None:
                    close_backend(raw)


if __name__ == '__main__':
    main()
