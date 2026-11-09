"""Task-local archive round trip and independent offline analysis rebuild."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
PILOT = BASE.parent/'online_injection_20261109'
REPO = PILOT/'source'
PACKAGE = REPO/'experiments/online-injection-ablation-20261109'
REPORT = BASE.parents[1]/'outputs/online-injection-ablation-20261109'
WORKSPACE = BASE/'handoff-roundtrip'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    subprocess.run([sys.executable,'-X','utf8',*map(str,args)],check=True)


def main():
    assert not WORKSPACE.exists(), 'Preserve any prior verification workspace.'
    run(PACKAGE/'unpack.py','--verify-only')
    run(PACKAGE/'unpack.py','--output',WORKSPACE)
    frozen = json.loads((WORKSPACE/'protocol/FROZEN.json').read_text(encoding='utf-8'))
    count = 0
    for name,expected in frozen['source_files'].items():
        p = WORKSPACE/'generation-source'/name.removeprefix('source/')
        assert digest(p) == expected,name
        count += 1
    for name,expected in frozen['hashes'].items():
        p = WORKSPACE/name if name.startswith('data/') else WORKSPACE/'protocol'/name
        if p.is_file():assert digest(p) == expected,name
        else:assert name == 'data/techqa/corpus.jsonl',name
    assert digest(WORKSPACE/'snapshot/prepared.json') == frozen['prepared_sha256']
    assert digest(WORKSPACE/'protocol/runtime-model_runner_v1.py') == frozen['runtime_model_runner_sha256']
    rebuilt = WORKSPACE/'rebuilt';rebuilt.mkdir()
    shutil.copyfile(REPORT/'OBSERVATIONS.md',rebuilt/'OBSERVATIONS.md')
    shutil.copytree(REPORT/'figures',rebuilt/'figures')
    frozen_source = WORKSPACE/'generation-source'
    os.environ['PYTHONPATH'] = os.pathsep.join([str(frozen_source/'src'),str(frozen_source/'scripts'),str(PILOT/'devdeps')])
    run(PACKAGE/'analysis/summarize_injection_ablation.py','--snapshot',WORKSPACE/'snapshot',
        '--gold',WORKSPACE/'data/techqa/gold.jsonl','--output',rebuilt)
    compared = {}
    for name in ('summary.json','audit-checks.json','grades.jsonl','REPORT.md'):
        assert digest(rebuilt/name) == digest(REPORT/name),name
        compared[name] = digest(rebuilt/name)
    summary = json.loads((rebuilt/'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'complete' and summary['generated'] == 336
    assert summary['all_saved_measurements_pass_audit']
    run(PACKAGE/'analysis/plot_injection_ablation.py','--summary',rebuilt/'summary.json',
        '--grades',rebuilt/'grades.jsonl','--output',rebuilt/'rebuilt-figures')
    figures = {}
    for stem in ('immediate-end','stop-outcomes','thinking-length','quality'):
        for ext in ('png','svg'):
            name = stem+'.'+ext
            assert digest(rebuilt/'rebuilt-figures'/name) == digest(REPORT/'figures'/name),name
            figures[name] = digest(rebuilt/'rebuilt-figures'/name)
    run(PACKAGE/'analysis/export_injection_ablation_traces.py','--snapshot',WORKSPACE/'snapshot',
        '--token-map',WORKSPACE/'snapshot/trace-tokenizer-subset.json','--output',rebuilt/'traces')
    traces = {}
    for p in sorted((REPORT/'traces').iterdir()):
        assert digest(rebuilt/'traces'/p.name) == digest(p),p.name
        traces[p.name] = digest(p)
    receipt = {'verified_at_utc':datetime.now(UTC).isoformat(),
        'archive_sha256':digest(PACKAGE/'full/analysis-workspace.tar.gz'),'verified_frozen_source_files':count,
        'prepared_input_matches_frozen':True,'runtime_runner_matches_frozen':True,'recomputed_measurements':336,
        'analysis_uses_packaged_scripts_and_extracted_frozen_source':True,
        'scores_audits_and_report_byte_identical':compared,'regenerated_png_and_svg_byte_identical':figures,
        'regenerated_trace_files_byte_identical':traces,
        'network_requests':0,'model_generation_requests':0,'verification_script_sha256':digest(Path(__file__))}
    data = json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
    (REPORT/'handoff-verification.json').write_text(data,encoding='utf-8')
    target = PACKAGE/'reports/handoff-verification.json';target.write_text(data,encoding='utf-8')
    verifier = PACKAGE/'analysis/verify_handoff.py';shutil.copyfile(__file__,verifier)
    path = PACKAGE/'manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
    for p in (target,verifier):
        name = p.relative_to(PACKAGE).as_posix()
        assert not any(e['path']==name for e in manifest['direct_files'])
        manifest['direct_files'].append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)})
    manifest['direct_files'].sort(key=lambda e:e['path'])
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    run(PACKAGE/'unpack.py','--verify-only')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()
