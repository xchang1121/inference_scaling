"""Package the completed matched-content study with immutable raw provenance."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
PILOT = BASE.parent/'online_injection_20261109'
REPO = PILOT/'source'
REPORT = BASE.parents[1]/'outputs/online-injection-ablation-20261109'
DEST = REPO/'experiments/online-injection-ablation-20261109'


def details(name,data):
    return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    summary = json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'complete' and summary['generated'] == 336
    assert summary['generation_exit'] == '0' and summary['all_saved_measurements_pass_audit']
    figures = json.loads((REPORT/'figures/figures-manifest.json').read_text(encoding='utf-8'))
    assert figures['summary_sha256'] == hashlib.sha256((REPORT/'summary.json').read_bytes()).hexdigest()
    assert figures['grades_sha256'] == hashlib.sha256((REPORT/'grades.jsonl').read_bytes()).hexdigest()
    assert (REPORT/'OBSERVATIONS.md').is_file()
    assert not DEST.exists(), 'Preserve an existing package.'
    assert (BASE/'formal-provenance/final-runtime-receipt.json').is_file()
    assert not json.loads((BASE/'formal-provenance/final-runtime-receipt.json').read_text())['state']['Running']
    prepared = json.loads((BASE/'snapshot/prepared.json').read_text(encoding='utf-8'))
    entries = {}

    def add(name,path):
        assert name not in entries
        entries[name] = path.read_bytes()

    for p in (BASE/'snapshot').iterdir():
        if p.is_file() and p.suffix in {'.json','.jsonl','.exit'}:add('snapshot/'+p.name,p)
    for p in (BASE/'formal-provenance').rglob('*'):
        if p.is_file():add('protocol/'+p.relative_to(BASE/'formal-provenance').as_posix(),p)
    add('protocol/source-frozen.tgz',BASE/'source-frozen.tgz')
    for name in ('questions.jsonl','gold.jsonl','manifest.json'):
        add('data/techqa/'+name,PILOT/'data/techqa'/name)
    needed = {h['document']['id'] for c in prepared['cases'] for h in c['hits']}
    for line in (PILOT/'data/techqa/gold.jsonl').read_text(encoding='utf-8').splitlines():
        needed.update(json.loads(line)['document_ids'])
    selected = []
    for line in (PILOT/'data/techqa/corpus.jsonl').read_text(encoding='utf-8').splitlines():
        if json.loads(line)['id'] in needed:selected.append(line)
    assert len(selected) == len(needed)
    entries['data/techqa/retrieved-and-gold-documents.jsonl'] = ('\n'.join(selected)+'\n').encode()
    add('data/techqa/DATASET_CARD.md',PILOT/'techqa_raw/README.md')
    add('data/techqa/LICENSE-Apache-2.0.txt',PILOT/'NVIDIA-LICENSE-Apache-2.0.txt')
    add('data/unrelated-prose.txt',BASE/'unrelated-prose.txt')
    with tarfile.open(BASE/'source-frozen.tgz') as tar:
        for m in tar:
            if m.isfile():
                assert '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
                entries['generation-source/'+m.name] = tar.extractfile(m).read()
    (DEST/'full').mkdir(parents=True)
    archive = DEST/'full/analysis-workspace.tar.gz';members=[]
    with (archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped,
          tarfile.open(fileobj=zipped,mode='w') as tar):
        for name,data in sorted(entries.items()):
            members.append(details(name,data));m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o644
            tar.addfile(m,io.BytesIO(data))
    for p in REPORT.rglob('*'):
        if p.is_file():
            target=DEST/'reports'/p.relative_to(REPORT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    (DEST/'analysis').mkdir()
    for name in ('summarize_injection_ablation.py','plot_injection_ablation.py','export_injection_ablation_traces.py','benchmark_injection.py'):
        shutil.copyfile(REPO/'scripts'/name,DEST/'analysis'/name)
    assert hashlib.sha256((DEST/'analysis/benchmark_injection.py').read_bytes()).hexdigest() == summary['analysis_runtime']['grading_script_sha256']
    shutil.copyfile(__file__,DEST/'analysis/package_results.py')
    shutil.copyfile(REPO/'experiments/qwen38-20261106/unpack.py',DEST/'unpack.py')
    readme = '''# 无 effort 提示的思考预算与注入格式对照

本轮使用原 8 道 TechQA、3 档 thinking 预算、14 个方法，共 336 份完整轨迹。所有正式模型计算统一使用 198 物理 3 号单卡，TP=1。单轨迹采样，不包含 IS/MH 或候选选择。

- [完整结果、机制对照与图表](reports/REPORT.md)
- [逐条判分和停止行为](reports/grades.jsonl)、[完整统计](reports/summary.json)、[完整性审计](reports/audit-checks.json)
- [冻结方案](../../docs/reports/ONLINE_INJECTION_FOLLOWUP_PROTOCOL.md)
- [更智能的检索触发与插入：论文和接入取舍](../../docs/reports/ADAPTIVE_RETRIEVAL_OPTIONS.md)
- [首轮 pilot](../online-injection-20261109/README.md)

实际 prompt 没有任何 effort 指令。当前模型模板省略参数会默认插入 xhigh，因此通过其不生成 effort 文本的 medium 模板分支渲染，并逐题检查实际 token。thinking 预算 256 / 1024 / 4096；最终回答独立上限 512 token；外部资料不占 thinking 预算，但占上下文和 prefill。

无关对照只替换正文，保留来源标题与标签；外层去包装和仅包装使用等 token 数换行。五种资料变体逐题 token 数完全相同。relevant 标签指 BM25 检索资料，不是保证正确的 gold 证据。这些条件隔离当前正文和外层文字，不能涵盖所有格式效应。

## 校验与展开

在仓库根目录使用 Python 3.11 或更新版本：

```bash
python experiments/online-injection-ablation-20261109/unpack.py --verify-only
python experiments/online-injection-ablation-20261109/unpack.py --output analysis-workspace/injection-ablation
```

目标必须是新建或空目录。展开只恢复文件，不联网、不运行模型。manifest 保存直接文件、归档及每个成员的 SHA256。原始数据包括 snapshot/generation.jsonl、冻结 prepared.json、token IDs、生成概率、事件、成本、源码和运行时证据；初次纯 CPU 配置校验失败也保留，没有正式模型测量。

## 离线重建

安装 NumPy 和 Matplotlib 后，使用归档中的分析脚本与展开的冻结源码，不需要运行模型：

```bash
PYTHONPATH=analysis-workspace/injection-ablation/generation-source/src python experiments/online-injection-ablation-20261109/analysis/summarize_injection_ablation.py --snapshot analysis-workspace/injection-ablation/snapshot --gold analysis-workspace/injection-ablation/data/techqa/gold.jsonl --output analysis-workspace/injection-ablation/rebuilt
PYTHONPATH=analysis-workspace/injection-ablation/generation-source/src python experiments/online-injection-ablation-20261109/analysis/plot_injection_ablation.py --summary analysis-workspace/injection-ablation/rebuilt/summary.json --grades analysis-workspace/injection-ablation/rebuilt/grades.jsonl --output analysis-workspace/injection-ablation/rebuilt/figures
PYTHONPATH=analysis-workspace/injection-ablation/generation-source/src python experiments/online-injection-ablation-20261109/analysis/export_injection_ablation_traces.py --snapshot analysis-workspace/injection-ablation/snapshot --token-map analysis-workspace/injection-ablation/snapshot/trace-tokenizer-subset.json --output analysis-workspace/injection-ablation/rebuilt/traces
```

不重复包含模型权重、运行时镜像或完整 28,481 文档语料；prepared 输入包含实际检索资料，另保存全部检索及 gold 文档子集。重新检索需按原数据 manifest 的 NVIDIA 固定 revision 下载完整语料并核对 SHA256。TechQA 数据卡与 Apache-2.0 声明包含在归档中。

本轮是同八题的机制复核，三档预算不构成 24 道独立新题；F1 不能代替语义正确率。按完整原始轨迹判断自然结束、强制预算结束、无注入和空答案，失败及不符合预期的结果不丢弃。
'''
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    direct = [details(p.relative_to(DEST).as_posix(),p.read_bytes()) for p in sorted(DEST.rglob('*'))
              if p.is_file() and 'full' not in p.relative_to(DEST).parts]
    manifest = {'schema':1,'source_commit':summary['frozen']['source_commit'],'formal_measurements':336,
        'thinking_budgets':summary['frozen']['budgets'],'devices':[3],'prepared_sha256':summary['frozen']['prepared_sha256'],
        'retrieved_and_gold_documents':len(needed),'direct_files':direct,
        'archives':[{**details('full/analysis-workspace.tar.gz',archive.read_bytes()),'destination':'.','files':members}]}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(DEST),'archive_bytes':archive.stat().st_size,'members':len(members),'direct_files':len(direct)}))


if __name__ == '__main__':
    main()
