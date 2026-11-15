"""One small, immutable TRAIN tuning batch; no answers in runtime inputs."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

from benchmark_flare_gap import clean, render
from benchmark_is_gpu import GraphMeasurements, engine_settings, reset_cache
from benchmark_online_is import run_arm
from benchmark_smart_online_is import run_arm as run_legacy_arm
from inference_scaling.retrieval.smart_online_is import VARIANTS as LEGACY_VARIANTS
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import load_backend, close_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.retrieval.ablation import SYSTEM
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Retriever
from inference_scaling.retrieval.cot_guidance import DualStatistics, GuidanceConfig
from inference_scaling.retrieval.literature import FactBank, ExampleBank, LiteratureConfig
from inference_scaling.retrieval.online_is import OnlineRetrievalConfig
from inference_scaling.retrieval.internal_signals import InternalSignals, select_signal_worker
from inference_scaling.retrieval.native_journal import NativeJournal
from inference_scaling.retrieval.auxiliary_online_is import AuxiliaryPolicy
from inference_scaling.retrieval.gap_online_is import GapPolicy
from inference_scaling.retrieval.example_online_is import ExamplePolicy
from inference_scaling.retrieval.ranked_facts import FactRanking,RankedFactBank
from inference_scaling.retrieval.probe_online_is import ProbePolicy
from inference_scaling.retrieval.aligned_online_is import AlignmentPolicy
from inference_scaling.retrieval.hidden_alignment import DocumentVectors
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, TuningTap, run_tuned_is
from inference_scaling.shared.rng import SeedStream


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--settings', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--examples', type=Path)
    p.add_argument('--encoder', type=Path)
    p.add_argument('--document-vectors',type=Path)
    a = p.parse_args()
    batch = json.loads(a.batch.read_text())
    expected = len(batch['cases'])*len(batch['configs'])*len(batch['seeds'])
    assert 1 <= expected <= 12, 'small batches only'
    for case in batch['cases']:
        assert not set(case)-{'id','question','dataset','category','cohort'}
        assert case['id'].startswith('TRAIN_'), 'held-out evaluation needs a separate frozen runner'
        assert case['cohort'] in {'tuning', 'probe_train'}, 'independent selection requires a frozen selection runner'
    settings, cfg = load_settings(a.settings), GuidanceConfig()
    raw = tap = graph = signals = journal = None
    def signal_mode(config):
        mode = config.get('collect_signals', 'none')
        if mode not in {'none', 'distribution', 'hidden', 'attention', 'multilayer'}:
            raise ValueError('unknown signal collection mode')
        if config.get('auxiliary_policy', {}).get('method') == 'seakr':
            return 'hidden'
        if 'probe_policy' in config or 'alignment_policy' in config:return 'hidden'
        if config.get('policy', {}).get('trigger') == 'dragin' or config.get('policy', {}).get('query') == 'attention':
            return 'attention'
        return mode if mode != 'none' else ('distribution' if config.get('policy', {}).get('trigger')
                                          in {'entropy','margin','risk'} else 'none')
    modes = [signal_mode(c) for c in batch['configs']]
    worker_mode = batch.get('diagnostic_worker', 'none')
    if worker_mode not in {'none', 'distribution', 'hidden', 'attention', 'multilayer'}:
        raise ValueError('unknown diagnostic worker')
    all_modes = set(modes) | {worker_mode}
    multilayer = 'multilayer' in all_modes
    if multilayer:
        from inference_scaling.retrieval.multilayer_capture import SCHEMA, validate_diagnostic_batch
        validate_diagnostic_batch(batch)
    if {'hidden', 'attention'} <= all_modes:
        raise ValueError('hidden and attention require separately validated workers')
    need_signals = all_modes != {'none'}
    hidden_dimensions = 513 if multilayer else 128 if 'hidden' in all_modes else 0
    attention_topk = 16 if 'attention' in all_modes else 0
    if need_signals:
        if multilayer:
            from inference_scaling.arllm.backends import vllm_backend
            vllm_backend._ASCEND_WORKER = 'inference_scaling.retrieval.multilayer_signals_worker.MultilayerSignalWorker'
        else:
            select_signal_worker(hidden=bool(hidden_dimensions), attention=bool(attention_topk))
    with a.output.open('x', encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        write({'kind': 'metadata', 'batch': batch, 'config': asdict(cfg), 'native_journal_expected': True,
               'batch_sha256': hashlib.sha256(a.batch.read_bytes()).hexdigest(),
               'source': {str(f.relative_to(Path(__file__).resolve().parents[1])):
                          hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
        try:
            if need_signals:
                settings['ar']['engine']['vllm']['engine_kwargs']['max_logprobs'] = (
                    600 if multilayer else 160 if hidden_dimensions else 45 if attention_topk else 10)
            raw = load_backend(settings['ar']['model'], settings['ar']['engine'], seed=batch['seeds'][0], logprobs=6)
            if need_signals:
                signals = InternalSignals(raw, hidden_dimensions=hidden_dimensions, attention_topk=attention_topk)
            graph, tap = GraphMeasurements(raw), DualStatistics(raw)
            journal = NativeJournal(raw, tap, a.output.with_name(a.output.stem+'-native.jsonl'))
            controller_tap = TuningTap(tap, signals) if need_signals else tap
            close = (raw.tokenizer.convert_tokens_to_ids('</think>'),)
            prepared = json.loads(a.prepared.read_text())
            indices, banks, example_banks = {}, {}, {}
            document_vectors=DocumentVectors(a.document_vectors) if a.document_vectors else None
            for config in batch['configs']:
                if 'example_policy' in config:
                    if not a.examples or not a.encoder:
                        raise ValueError('example corpus and encoder required')
                    example_cfg=LiteratureConfig(**config.get('fact_config',{}))
                    key=json.dumps(asdict(example_cfg),sort_keys=True)
                    reused_from=example_banks[key][0] if key in example_banks else None
                    if reused_from is None:
                        example_banks[key]=(config['id'],ExampleBank(raw,a.examples,a.encoder,example_cfg))
                    banks[config['id']] = example_banks[key][1]
                    write({'kind':'example_bank','config_id':config['id'],'manifest':banks[config['id']].manifest,
                           'reused_from_config_id':reused_from,
                           'examples_sha256':hashlib.sha256(a.examples.read_bytes()).hexdigest()})
                    continue
                kind = config.get('retriever', 'bm25')
                if kind not in indices:
                    indices[kind] = Retriever.from_jsonl(a.corpus, kind=kind)
                bank_cls=RankedFactBank if 'fact_ranking' in config else FactBank
                extra={'ranking':FactRanking(**config['fact_ranking'])} if 'fact_ranking' in config else {}
                banks[config['id']] = bank_cls(raw, indices[kind], LiteratureConfig(**config.get('fact_config', {})),**extra)
            write({'kind': 'engine', 'model_id': raw.model_id, 'config': engine_settings(raw),
                   'corpus_sha256': next(iter(indices.values())).fingerprint if indices else None})
            warm_prefix = render(raw.tokenizer, settings['ar'], SYSTEM, batch['cases'][0]['question'], True)
            started = time.perf_counter()
            warm, = raw.sample_batch([GenerationRequest(warm_prefix, 64,
                SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id), 20261115,
                'tuning:warmup', stop_sequences=(close,), intermediate_events=False)])
            write({'kind': 'warmup', 'seconds': time.perf_counter()-started,
                   'generated_tokens': len(warm.token_ids)})
            completed = 0
            for qi, case in enumerate(batch['cases']):
                case = {**case, 'prompt_ids': render(raw.tokenizer, settings['ar'], SYSTEM, case['question'], True)}
                for seed in batch['seeds']:
                    offset = qi % len(batch['configs'])
                    configs = batch['configs'][offset:]+batch['configs'][:offset]
                    for config in configs:
                        journal.unit = ':'.join(map(str, (case['id'], seed, config['id'])))
                        reset_cache(raw)
                        if need_signals:
                            signals.enabled = signal_mode(config) != 'none'
                            controller_tap.history.clear()
                        before, g0, journal_io0 = _snapshot(raw), graph.snapshot(), journal.io_seconds
                        started = time.perf_counter()
                        ret = dict(config.get('retrieval_config', {}))
                        if 'positions' in ret:
                            ret['positions'] = tuple(ret['positions'])
                        if config['arm'] == 'tuned_online':
                            policy = TuningPolicy(**config.get('policy', {}))
                            result = run_tuned_is(raw, controller_tap, tuple(case['prompt_ids']), close,
                                SeedStream(seed).derive('question', case['id']), cfg,
                                question=case['question'], facts=banks[config['id']], policy=policy,
                                auxiliary_policy=AuxiliaryPolicy(**config['auxiliary_policy'])
                                    if 'auxiliary_policy' in config else None,
                                gap_policy=GapPolicy(**config['gap_policy']) if 'gap_policy' in config else None,
                                example_policy=ExamplePolicy(**config['example_policy']) if 'example_policy' in config else None,
                                probe_policy=ProbePolicy(**config['probe_policy']) if 'probe_policy' in config else None,
                                alignment_policy=AlignmentPolicy(**config['alignment_policy']) if 'alignment_policy' in config else None,
                                document_vectors=document_vectors)
                            result.update(arm=config['arm'], question_id=case['id'], seed=seed,
                                          base_ids=list(case['prompt_ids']), static_packets=[],
                                          static_injected_ids=[], static_retrievals=[])
                        else:
                            runner = run_legacy_arm if config['arm'] in LEGACY_VARIANTS else run_arm
                            result = runner(raw, controller_tap, banks[config['id']], case, seed, config['arm'], cfg,
                                             close, prepared['noise_ids'], OnlineRetrievalConfig(**ret))
                        if need_signals:
                            for call in result['native_calls']:
                                if call['request_id'] in controller_tap.history:
                                    call['internal_signals'] = controller_tap.history[call['request_id']]
                            assert not signals.rows and not signals.pending
                        result['signal_collection'] = {'requested': signal_mode(config),
                            'hidden_dimensions': hidden_dimensions if need_signals and signals.enabled else 0,
                            'attention_topk': attention_topk if need_signals and signals.enabled else 0,
                            'hidden_source': 'multilayer_diagnostic' if multilayer else 'final_normalized_pre_lm_head', 'projection_seed': 2026111502,
                            'state_offset': -1}
                        if multilayer:
                            result['signal_collection']['schema'] = SCHEMA
                        elapsed, after = time.perf_counter()-started, _snapshot(raw)
                        write({'kind': 'measurement', 'config_id': config['id'], 'seconds': elapsed,
                               'native_journal_io_seconds': journal.io_seconds-journal_io0,
                               'cost': {k: after[k]-before[k] for k in before.keys() & after.keys()
                                        if type(before[k]) in (int, float)},
                               **graph.difference(g0), **result})
                        completed += 1
                        print(json.dumps({'completed': completed, 'expected': expected,
                                          'case': case['id'], 'config': config['id']}), flush=True)
                        assert not tap.rows
            assert completed == expected
            write({'kind': 'completed', 'units': completed})
        except BaseException as error:
            write({'kind': 'failed', 'error': repr(error),
                   'unit': journal.unit if journal is not None else 'setup',
                   'native_snapshot': _snapshot(raw) if raw is not None else {}})
            raise
        finally:
            if journal is not None:
                journal.close()
            if tap is not None:
                tap.close()
            if signals is not None:
                signals.close()
            if graph is not None:
                graph.close()
            if raw is not None:
                close_backend(raw)


if __name__ == '__main__':
    main()
