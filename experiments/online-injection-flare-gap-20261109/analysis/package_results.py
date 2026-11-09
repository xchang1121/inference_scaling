"""Package only a complete, audited formal run; no network/model work."""
from pathlib import Path
import gzip,hashlib,io,json,shutil,tarfile
BASE=Path(__file__).resolve().parent;PILOT=BASE.parent/'online_injection_20261109';REPO=PILOT/'source';REPORT=BASE.parents[1]/'outputs/online-injection-flare-gap-20261109';DEST=REPO/'experiments/online-injection-flare-gap-20261109'
def detail(name,data):return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def main():
 summary=json.loads((REPORT/'summary.json').read_text(encoding='utf-8'));assert summary['conditions']==summary['expected_conditions']==176 and summary['completed_marker'] and not summary['audit_failures'] and not summary['failed_executions']
 state=json.loads((BASE/'snapshot/state-formal.json').read_text(encoding='utf-8'));assert state['exit']=='0' and not state['changed_files'] and not state['changed_inputs'] and state['prepared_matches_frozen'] and state['runtime_runner_matches_frozen'] and not state['gold_deployment_files']
 finish=json.loads((BASE/'provenance/formal-container-finish.json').read_text(encoding='utf-8'));assert not finish['running'] and finish['generation_exit']==0 and not finish['state']['OOMKilled']
 assert not DEST.exists(),'preserve an existing canonical package'
 entries={}
 def add(name,path):
  assert name not in entries;entries[name]=path.read_bytes()
 for folder,prefix in [('snapshot','snapshot'),('provenance','protocol'),('initial-smoke','initial-smoke'),('final-smoke','final-smoke')]:
  for p in (BASE/folder).rglob('*'):
   if p.is_file():add(prefix+'/'+p.relative_to(BASE/folder).as_posix(),p)
 for p in BASE.glob('source-frozen*.tgz'):add('protocol/'+p.name,p)
 for name in ['CPU-TESTS.txt','LINT.txt','checker-contract-audit.json','launch_stage.py','bootstrap-result.txt','launch-stage-result.txt','bootstrap-result.txt','collect.py','fetch_trace_vocabulary.py','record_resources.py','verify_handoff.py','manual-gate-review.jsonl']:
  if (BASE/name).exists() and 'tools/'+name not in entries:add('tools/'+name,BASE/name)
 for name in ['questions.jsonl','gold.jsonl','smoke-questions.jsonl','manifest.json']:add('data/'+name,BASE/'data'/name)
 add('data/unrelated-prose.txt',BASE/'unrelated-prose.txt')
 add('protocol/unrelated-prose.txt',BASE/'unrelated-prose.txt')
 add('data/DATASET_CARD.md',PILOT/'techqa_raw/README.md');add('data/LICENSE-Apache-2.0.txt',PILOT/'NVIDIA-LICENSE-Apache-2.0.txt')
 prepared=json.loads((BASE/'snapshot/prepared.json').read_text(encoding='utf-8'));needed=set()
 for c in prepared['cases']:needed.update(c['static_packet']['parent_ids'])
 for line in (BASE/'data/gold.jsonl').read_text(encoding='utf-8').splitlines():needed.update(json.loads(line)['document_ids'])
 rows=[json.loads(l) for l in (BASE/'snapshot/formal.jsonl').read_text(encoding='utf-8').splitlines()]
 for row in rows:
  if row['kind']=='execution_unit':
   if row.get('packet'):needed.update(row['packet']['parent_ids'])
   for event in row['result']['events']:
    if event.get('packet'):needed.update(event['packet']['parent_ids'])
 documents=[]
 for line in (PILOT/'data/techqa/corpus.jsonl').read_text(encoding='utf-8').splitlines():
  if json.loads(line)['id'] in needed:documents.append(line)
 assert len(documents)==len(needed)
 entries['data/retrieved-and-gold-documents.jsonl']=('\n'.join(documents)+'\n').encode()
 with tarfile.open(BASE/'source-frozen.tgz') as tar:
  for member in tar:
   if member.isfile():
    assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    entries['generation-source/'+member.name]=tar.extractfile(member).read()
 frozen=json.loads((BASE/'provenance/FROZEN.json').read_text(encoding='utf-8'))
 for name,h in frozen['source_files'].items():assert hashlib.sha256(entries['generation-source/'+name.removeprefix('source/')]).hexdigest()==h,name
 (DEST/'full').mkdir(parents=True);members=[];archive=DEST/'full/analysis-workspace.tar.gz'
 with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
  for name,data in sorted(entries.items()):
   members.append(detail(name,data));m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o644;tar.addfile(m,io.BytesIO(data))
 for p in REPORT.rglob('*'):
  if p.is_file():
   target=DEST/'reports'/p.relative_to(REPORT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 (DEST/'analysis').mkdir()
 for name in ['summarize_flare_gap.py','plot_flare_gap.py','export_flare_gap_traces.py']:shutil.copyfile(REPO/'scripts'/name,DEST/'analysis'/name)
 shutil.copyfile(__file__,DEST/'analysis/package_results.py');shutil.copyfile(REPO/'experiments/qwen38-20261106/unpack.py',DEST/'unpack.py')
 readme='''# FLARE + evidence-gap retrieval at complete reasoning steps

16 questions (8 newly held out, 8 regression), 11 conditions each, **176 formal condition outcomes**. Qwen3.8-27B on 198 physical NPU3 only, TP1. The prior 336-result archive remains unchanged.

- [Audited results and limits](reports/REPORT.md), [interpretation](reports/OBSERVATIONS.md)
- [Exact protocol](../../docs/reports/FLARE_GAP_PROTOCOL.md)
- [Per-condition grades/costs](reports/graded.jsonl), [summary](reports/summary.json)
- [All structural audits](reports/audit.jsonl), [paired controls](reports/pair-audit.jsonl), [gate decisions](reports/gate-events.jsonl)
- [Figure](reports/figures/comparison.png), [preselected full saved traces](reports/traces/manifest.json), [every accepted intervention](reports/accepted-traces/manifest.json)

Every condition has a 1024-token generated work cap, including all drafts, discarded previews, checker/planner output and sampled end/EOS markers. Final answer cap 512; evidence cap 256 including wrappers. Evidence input tokens consume context/prefill, rather than generated work. Equal token caps are not equal native FLOPs or equal actual use before natural termination. Main thinking is enabled; independent checkers use no-thinking greedy JSON. Rendered main prompts contain no effort instruction.

Primary FLARE inserts a short quoted packet directly after a complete assistant reasoning step. Its paired rebuild variant uses the exact same accepted checkpoint and packet in the original user reference region. Static RAG, legacy SC, and step-boundary SC are included. The legacy SC criterion is retained; all methods now share the short window retriever, so it is not a bitwise replay of the previous long-document SC arm. Unrelated bodies are substituted after the real-evidence support gate, with labels/wrappers and exact payload length unchanged.

Exact pre-intervention units are shared, not independently rerun. Each standalone condition is charged the full common stem plus its own cold branch. Actual study totals sum unique units. If no gate accepts, all corresponding content/format conditions share one completed result. All EOS, empty answers, rejected/invalid checks and capped answers remain in the denominator. One seed per question; F1 is normalized generative token overlap, not official extractive TechQA accuracy or semantic correctness.

## Verify and unpack

Python 3.11+:

```bash
python experiments/online-injection-flare-gap-20261109/unpack.py --verify-only
python experiments/online-injection-flare-gap-20261109/unpack.py --output analysis-workspace/flare-gap
```

The target must be new or empty. Extraction verifies every direct file and archive member, uses no network and runs no model. Generation source and final analysis scripts are separately hashed: analysis received output determinism/reporting fixes after generation was frozen.

## Rebuild results offline

No model weights, accelerator or full tokenizer are required. A verified ByteLevel decoder subset reconstructs every saved execution-unit thinking/answer and the two preselected heldout trace examples. NumPy and Matplotlib are needed only for the figure.

```bash
PYTHONPATH=analysis-workspace/flare-gap/generation-source/src python experiments/online-injection-flare-gap-20261109/analysis/summarize_flare_gap.py --generation analysis-workspace/flare-gap/snapshot/formal.jsonl --prepared analysis-workspace/flare-gap/snapshot/prepared.json --gold analysis-workspace/flare-gap/data/gold.jsonl --output analysis-workspace/flare-gap/rebuilt
PYTHONPATH=analysis-workspace/flare-gap/generation-source/src python experiments/online-injection-flare-gap-20261109/analysis/plot_flare_gap.py --summary analysis-workspace/flare-gap/rebuilt/summary.json --output analysis-workspace/flare-gap/rebuilt/figures
PYTHONPATH=analysis-workspace/flare-gap/generation-source/src python experiments/online-injection-flare-gap-20261109/analysis/export_flare_gap_traces.py --snapshot analysis-workspace/flare-gap/snapshot --output analysis-workspace/flare-gap/rebuilt/traces
```

An additional accepted-traces directory covers every accepted FLARE case with all 11 conditions, selected by gate decision rather than answer quality.

The archive includes formal raw requests, exact prefixes/IDs/logprobs/SC, discarded drafts, evidence/gates, both TRAIN smoke versions, a separately labeled synthetic positive checker contract, failed initial CPU preparation, runtime/container/source/input receipts and local grading gold. Gold was not deployed during generation. Corpus parent/gold document subsets, data card and license are included. Re-running BM25 parent selection needs the full 28,481-document corpus at the pinned NVIDIA revision and SHA256; subset replay is not full-index retrieval reproduction. Model weights/container images and full corpus are not duplicated.

Unit native graph figures estimate dense padded FLOPs; attention/host work and initialization profiling are not fully instrumented. Stage wall-time receipts include initialization/teardown. Smoke, warmup and synthetic contract costs are reported separately from formal condition costs. The synthetic fixture's provided reasoning is input, never presented as sampled prehistory or a benchmark answer.
'''
 (DEST/'README.md').write_text(readme,encoding='utf-8',newline='\n')
 direct=[detail(p.relative_to(DEST).as_posix(),p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file() and 'full' not in p.relative_to(DEST).parts]
 manifest={'schema':1,'source_commit':frozen['source_commit'],'formal_conditions':176,'cohorts':{'heldout':8,'regression':8},'devices':[3],'retrieved_and_gold_documents':len(needed),'direct_files':direct,'archives':[{**detail('full/analysis-workspace.tar.gz',archive.read_bytes()),'destination':'.','files':members}]}
 (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'package':str(DEST),'archive_bytes':archive.stat().st_size,'members':len(members),'direct_files':len(direct),'documents':len(needed)}))
if __name__=='__main__':main()
