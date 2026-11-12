"""Collect, audit, package, and byte-replay the finished paired experiment."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--skip-collect',action='store_true');a=p.parse_args()
base=Path(__file__).resolve().parent;repo=base.parent/'online_injection_20261109/source'
dest=repo/'experiments/online-injection-literature-20261112'
assert not dest.exists(),'preserve any existing release package'
if not a.skip_collect:subprocess.run([sys.executable,'-X','utf8',str(base/'collect.py'),'formal'],check=True)
for stage in ['smoke','formal']:
    subprocess.run([sys.executable,'-X','utf8',str(base/'analyze.py'),stage],check=True)
summary=json.loads((base/'reports-formal/summary.json').read_text(encoding='utf-8'))
assert summary['questions']==16 and summary['measurements']==352 and summary['audit']['passed']
assert summary['audit']['final_truncated']==0
subprocess.run([sys.executable,'-X','utf8',str(base/'interpret.py')],check=True)
subprocess.run([sys.executable,'-X','utf8',str(repo/'scripts/diagnose_literature_empty_answers.py'),
    '--snapshot',str(base/'snapshot/formal'),'--summary',str(base/'reports-formal/summary.json'),
    '--answers',str(base/'reports-formal/answers.jsonl'),'--output',str(base/'diagnostics')],check=True)
plot_python=base/'plot-venv/Scripts/python.exe'
assert plot_python.exists(),'prepare the pinned plotting environment before finalizing'
plot_env=dict(os.environ);plot_env.pop('PYTHONPATH',None)
subprocess.run([str(plot_python),str(repo/'scripts/plot_literature_retrieval.py'),
    '--summary',str(base/'reports-formal/summary.json'),
    '--answers',str(base/'reports-formal/answers.jsonl'),
    '--output',str(base/'visuals')],env=plot_env,check=True)
manifest=json.loads((base/'data/manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():assert hashlib.sha256((base/'data'/name).read_bytes()).hexdigest()==digest,name
for folder in (base/'snapshot/formal').iterdir():
    if not folder.is_dir():continue
    finish=json.loads((folder/'formal-finish.json').read_text())
    assert finish['exit']==0 and not finish['state']['Running']
dest.mkdir(parents=True)
def copy(src,relative):
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
for name in ['questions.jsonl','smoke-questions.jsonl','gold.jsonl','examples.jsonl','manifest.json']:
    copy(base/'data'/name,'inputs/'+name)
for name in ['settings.json','FROZEN_IS.json','formal-approval.json']:
    copy(base/name,'inputs/'+name)
copy(base/'source-frozen.tgz','generation-source.tar.gz')
copy(base/'source-frozen-v1.tgz','retired/template-v1/generation-source.tar.gz')
copy(repo/'docs/reports/LITERATURE_RETRIEVAL_PROTOCOL.md','PROTOCOL.md')
copy(repo/'docs/third_party/THINKRETRIEVE-LICENSE.txt','licenses/THINKRETRIEVE-LICENSE.txt')
copy(base/'offline-smoke-v2/VERIFICATION.json','SMOKE-OFFLINE-VERIFICATION.json')
copy(base.parent/'online_injection_20261109/NVIDIA-LICENSE-Apache-2.0.txt','licenses/TECHQA-LICENSE-Apache-2.0.txt')
copy(base.parent/'online_injection_scaling_20261109/unrelated-prose.txt','inputs/unrelated-prose.txt')
input_digests={}
for src,name in [(base.parent/'online_injection_20261109/data/techqa/corpus.jsonl','corpus.jsonl'),
                 (base.parent/'online_injection_20261109/techqa_raw/train.json','example-bank-source.json')]:
    payload=src.read_bytes();digest=hashlib.sha256(payload).hexdigest()
    if name=='corpus.jsonl':
        start=json.loads((base/'snapshot/formal/a3/formal-start.json').read_text())
        assert digest==start['source']['hashes']['data/corpus.jsonl']
    else:assert digest==manifest['bank']['raw_sha256']
    (dest/'inputs'/(name+'.gz')).write_bytes(gzip.compress(payload,mtime=0))
    input_digests[name]=digest
(dest/'inputs/compressed-input-hashes.json').write_text(json.dumps(input_digests,indent=2)+'\n',encoding='utf-8')
for stage in ['smoke-template-v1','diagnostic','diagnostic-spawn-v1']:
    for f in sorted((base/'snapshot'/stage).glob('*.tar.gz')):
        copy(f,'retired/'+stage+'/'+f.name)
    receipt=base/'snapshot'/stage/'manifest.json'
    if receipt.exists():copy(receipt,'retired/'+stage+'/manifest.json')
for f in sorted((base/'reports-smoke-template-v1').iterdir()):
    if f.is_file():copy(f,'retired/template-v1/reports/'+f.name)
engine=None
for stage in ['smoke','formal']:
    for f in sorted((base/'snapshot'/stage).glob('*.tar.gz')):copy(f,'raw/'+stage+'/'+f.name)
    for f in sorted((base/('reports-'+stage)).iterdir()):
        if f.is_file():copy(f,'reports/'+stage+'/'+f.name)
    for folder in sorted((base/'snapshot'/stage).iterdir()):
        if not folder.is_dir():continue
        for line in (folder/(stage+'.jsonl')).open(encoding='utf-8'):
            row=json.loads(line)
            if row['kind']=='engine':engine=row;break
assert engine is not None
(dest/'inputs/encoder-manifest.json').write_text(json.dumps(engine['retrieval']['examples']['encoder'],indent=2)+'\n',encoding='utf-8')
(dest/'inputs/index-metadata.json').write_text(json.dumps(engine['retrieval'],indent=2)+'\n',encoding='utf-8')
for name in ['summarize_literature_retrieval.py','summarize_cot_guidance.py','summarize_flare_gap.py',
             'plot_literature_retrieval.py','diagnose_literature_empty_answers.py']:
    copy(repo/'scripts'/name,'analysis/'+name)
for name in ['test_literature_retrieval.py','test_literature_report.py','test_cot_guidance.py',
             'test_literature_plots.py','test_literature_empty_diagnostics.py']:
    copy(repo/'tests'/name,'analysis/tests/'+name)
copy(base/'plot-requirements.txt','analysis/plot-requirements.txt')
for f in sorted((base/'visuals').iterdir()):
    if f.is_file():copy(f,'visuals/'+f.name)
for f in sorted((base/'diagnostics').iterdir()):
    if f.is_file():copy(f,'diagnostics/'+f.name)
for name in ['prepare.py','deploy.py','launch.py','collect.py','approve.py','monitor_remote.py','watch.py',
             'analyze.py','replay.py','finalize.py','download_encoder_remote.py','native_diagnostic.py',
             'launch_diagnostic.py','preserve_v1.py','interpret.py','smoke_roundtrip.py','peek_audit.py']:
    copy(base/name,'tools/'+name)
readme='''# IRCoT and ThinkRetrieve with frozen Block-Metropolized SC

See [Chinese interpretation](reports/formal/RESULTS_ZH.md), [formal results](reports/formal/REPORT.md), [all answers and costs](reports/formal/answers.jsonl),
[the complete audit](reports/formal/audit.json), and [the frozen protocol](PROTOCOL.md).
The [overview](visuals/overview.png) compares F1, generation cost, latency and empty
answers. The [paired question view](visuals/question_differences.png) shows F1
differences relative to no-injection IS, with empty-answer markers. SVG originals
and the per-question comparison CSV are included in `visuals/`.
The [empty-answer diagnostics](diagnostics/EMPTY_ANSWERS.md) locate the observed
termination phase, distinguishing controller EOS, checkpoint EOS, selected IS
trajectory EOS and empty final readout. These locations do not establish why
the model selected EOS.

16 identical TechQA DEV questions × 2 seeds × 11 arms = 352 native measurements.
The independent TRAIN smoke contributes 22 measurements. The same IS algorithm,
SC reward, candidate/rollout counts, sampling, and generation limits are used in
all IS arms. Runtime retrieval receives no DEV gold. Every pipeline's controller,
interim answers, discarded candidates, and final answer are included in costs.
Cold-prefix-cache end-to-end latency is compared against a separately executed
ordinary generation on the same question, seed, and worker. Loading and initial
index construction are excluded and recorded separately.

These are explicit TechQA/native-token adaptations, not paper benchmark replicas.
IRCoT uses a separate retrieval agent and fresh IS reader. ThinkRetrieve uses E5
joint-QA retrieval over complete source-supported TRAIN troubleshooting examples,
intercepts thought-closing markers, and appends examples to retained thinking.
Actual EOS is never suppressed. The protocol explains the controls and differences.

Before formal evaluation, a TRAIN-only cold-cache diagnostic compared our initial
paraphrased insertion template against the official ThinkRetrieve template.
The official template improved continuation on one of two TRAIN fixtures, so V2
restored its exact text and example format. No DEV output or TRAIN F1 was used
to make this correction. The retired V1 source, all 22 V1 smoke measurements,
six diagnostic calls, and the failed diagnostic startup receipt are retained
under `retired/`. Formal results use V2 exclusively, after a fresh 22-run smoke.
The upstream MIT license is included under `licenses/`.

Raw archives include generated token IDs, SC values, exact prompt prefixes, all
IS pools and selection records, final answers, query and evidence records, native
cost counters, elapsed times, and a tokenizer-vocabulary subset. Reader source
hashes are frozen in `inputs/FROZEN_IS.json`. `generation-source.tar.gz` contains
the exact deployed source. E5 weights are not duplicated here; their pinned
revision and all file hashes are in `inputs/encoder-manifest.json`.
The exact retrieval corpus, original example-bank source, and unrelated text
are included under `inputs/`; large text inputs use gzip with uncompressed
SHA-256 values in `compressed-input-hashes.json`.

Offline replay needs Python and NumPy, without a model or network:

```text
python tools/replay.py --package . --workdir /absolute/path/to/new-replay-directory
```

Replay verifies package hashes and reconstructs both reports byte-for-byte from
raw records. The receipt is `OFFLINE-VERIFICATION.json`. Task-owned native
containers stopped after completion; unrelated workloads were left untouched.

To also regenerate and byte-check all visual artifacts, install the pinned
`analysis/plot-requirements.txt` dependencies and add `--figures` to the replay
command. `--plot-python /path/to/python` optionally selects a separate plotting
environment. The complete delivery was checked with this figure replay enabled.
Synthetic plotting fixtures are tests only and are never included in results.
'''
(dest/'README.md').write_text(readme,encoding='utf-8')
def write_manifest():
    files={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'
        and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    (dest/'MANIFEST.json').write_text(json.dumps({'schema':1,'files':files},indent=2)+'\n',encoding='utf-8')
write_manifest()
env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=str(base.parent/'online_injection_20261109/devdeps')
replay=base/'offline-replay-final'
subprocess.run([sys.executable,'-X','utf8',str(dest/'tools/replay.py'),'--package',str(dest),
    '--workdir',str(replay),'--figures','--plot-python',str(plot_python)],env=env,check=True)
copy(replay/'OFFLINE-VERIFICATION.json','OFFLINE-VERIFICATION.json');write_manifest()
print(json.dumps({'package':str(dest),'audit':summary['audit']}),flush=True)
