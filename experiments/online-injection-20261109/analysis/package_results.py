"""Package completed formal results with byte-preserved provenance."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE/'source'
REPORT = BASE.parents[1]/'outputs/online-injection-a3-20261109'
DEST = REPO/'experiments/online-injection-20261109'


def details(name, data):
    return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    summary = json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'complete' and summary['all_saved_measurements_pass_audit']
    assert summary['validation']['status'] in {'passed','failed'}, 'Validation must have a recorded terminal outcome.'
    if summary['validation']['status'] == 'failed':
        assert summary['validation']['conditional_cache']['status'] in {'passed','failed'}
        assert '目标模型的同种子续写复现要求没有通过' in (REPORT/'REPORT.md').read_text(encoding='utf-8')
    assert {k:s['generated'] for k,s in summary['datasets'].items()} == {'api':208,'techqa':104}
    assert not DEST.exists(), 'Use a fresh package directory; existing handoff is preserved.'
    assert (REPORT/'CASE_REVIEW.md').is_file()
    for name in ('FROZEN.json','settings-a3.json','launch.py','start.sh','runtime-verification.json','generation.log'):
        assert (BASE/'formal-provenance'/name).is_file(), name
    review = [json.loads(s) for s in (BASE/'manual-case-review.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(review) == len({(r['question_id'],r['arm']) for r in review}) == 104
    entries = {}

    def add(name, path):
        assert name not in entries
        entries[name] = path.read_bytes()

    for p in (BASE/'snapshot-a3').iterdir():
        if p.is_file() and p.suffix in {'.json','.jsonl','.exit'}:
            add('snapshot/'+p.name,p)
    for p in (BASE/'formal-provenance').rglob('*'):
        if p.is_file():add('protocol/'+p.relative_to(BASE/'formal-provenance').as_posix(),p)
    for p in (BASE/'data/api').rglob('*'):
        if p.is_file() and 'source_snapshots' not in p.relative_to(BASE/'data/api').parts:
            add('data/api/'+p.relative_to(BASE/'data/api').as_posix(),p)
    for name in ('questions.jsonl','gold.jsonl','manifest.json'):
        add('data/techqa/'+name,BASE/'data/techqa'/name)
    # Preserve every cited and gold document without duplicating the whole
    # 115 MB public corpus. The original full corpus hash and revision remain.
    needed = set()
    for line in (BASE/'data/techqa/gold.jsonl').read_text(encoding='utf-8').splitlines():
        needed.update(json.loads(line)['document_ids'])
    for line in (BASE/'snapshot-a3/techqa.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        for event in row.get('events',[]):needed.update(h['id'] for h in event.get('hits',[]))
    selected = []
    for line in (BASE/'data/techqa/corpus.jsonl').read_text(encoding='utf-8').splitlines():
        if json.loads(line)['id'] in needed:selected.append(line)
    assert len(selected) == len(needed)
    entries['data/techqa/retrieved-and-gold-documents.jsonl'] = ('\n'.join(selected)+'\n').encode()
    add('data/techqa/DATASET_CARD.md',BASE/'techqa_raw/README.md')
    add('data/techqa/LICENSE-Apache-2.0.txt',BASE/'NVIDIA-LICENSE-Apache-2.0.txt')
    for name in ('manual-case-review.jsonl','techqa-span-audit.json','chunking-diagnostic.json',
                 'chunking-diagnostic-protocol.json','chunking-retrieved-documents.jsonl'):
        add('annotations/'+name,BASE/name)
    with tarfile.open(BASE/'source-frozen.tgz') as source:
        for member in source:
            if member.isfile():
                assert '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
                entries['generation-source/'+member.name] = source.extractfile(member).read()
    for name in ('api.jsonl','techqa.jsonl','state.json'):
        add('excluded-preflight/'+name,BASE/'snapshot'/name)
    for p in (BASE/'data/synthetic-api').iterdir():
        if p.is_file():add('excluded-preflight/synthetic-api/'+p.name,p)
    entries['excluded-preflight/README.txt'] = b'Historical A/B preflight with other GPU configurations and fictional API contracts. Excluded from every formal score and cost table.\n'

    (DEST/'full').mkdir(parents=True)
    target = DEST/'full/analysis-workspace.tar.gz'
    members = []
    with (target.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped,
          tarfile.open(fileobj=zipped,mode='w') as bundle):
        for name,data in sorted(entries.items()):
            members.append(details(name,data))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            bundle.addfile(info,io.BytesIO(data))
    archive = {**details('full/analysis-workspace.tar.gz',target.read_bytes()),'destination':'.','files':members}
    for p in REPORT.rglob('*'):
        if p.is_file():
            dst = DEST/'reports'/p.relative_to(REPORT)
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dst)
    for name in ('summarize_injection.py','plot_injection_results.py','inspect_injected_evidence.py','validate_injection_cache.py','diagnose_injection_cache.py'):
        dst = DEST/'analysis'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/'scripts'/name,dst)
    shutil.copyfile(__file__,DEST/'analysis/package_results.py')
    shutil.copyfile(BASE/'chunking_diagnostic.py',DEST/'analysis/chunking_diagnostic.py')
    shutil.copyfile(BASE/'render_case_review.py',DEST/'analysis/render_case_review.py')
    shutil.copyfile(REPO/'experiments/qwen38-20261106/unpack.py',DEST/'unpack.py')
    readme = '''# 在线知识注入：首轮正式结果与复核材料

本实验所有正式测量统一使用 27.106.124.198 的物理 3 号卡、TP=1。控制器自动检索本地文档并追加资料，模型不生成工具调用。

- [结果报告](reports/REPORT.md)：8 道 TechQA、16 道官方文档规则诊断，各 13 种方法，共 312 份最终回答。
- [逐题复核](reports/CASE_REVIEW.md)：104 份 TechQA 回答的来源对照和探索性语义判断。没有独立盲审，不替代预定的 F1/EM。
- [详细方案](../../docs/online_injection.md)：触发参数、检索与注入流程、预算、概率与 KV 约束、后续 IS/ETS 计划。
- [统计](reports/summary.json) / [审计](reports/audit-checks.json) / [逐条判分](reports/grades.jsonl)。

正式生成完成不代表缓存复现测试通过：本次同种子多 token 续写有一个分支未通过，CPU 参考测试通过。后续固定历史的条件状态诊断单独记录，原失败保留；具体状态和适用范围见报告。规则题的严格得分包含输出格式因素，并有一个秒数字段歧义，见 api-diagnostic-review.json。

冻结生成源码为 `cbf7998dfcf997436137a0e7385de2da63f5f031`。原始 token、片段来源、检索事件、概率、成本、配置、初始化记录与网页来源清单使用普通 Git 压缩包保存。旧 A/B 输出和虚构合同数据位于 excluded-preflight，排除在正式结果之外。官方网页的完整下载快照保留在实验服务器与本地任务目录；本包保存短改述、URL、下载时间和快照 SHA256。

## 校验与展开

需要 Python 3.11 或更新版本。在仓库根目录运行：

```bash
python experiments/online-injection-20261109/unpack.py --verify-only
python experiments/online-injection-20261109/unpack.py --output analysis-workspace/online-injection
```

目标目录必须为空。展开只读取和恢复文件，不运行模型、不执行生成内容、不连接网络。manifest.json 保存归档、成员及直接展示文件的 SHA256。

展开目录中的 `snapshot/{api,techqa}.jsonl` 包含全部正式测量；`data` 中的问题、gold 与公开文档字段分离。`protocol` 保存实际配置、启动器、运行时版本与对齐证据；`generation-source` 是本次运行的冻结代码。没有模型权重和完整运行时镜像，不能据此保证重新生成相同随机输出。

TechQA 的全量 28,481 文档语料没有重复装入归档；保存了全部实际引用与 gold 文档的子集，文件名为 retrieved-and-gold-documents.jsonl。该子集不能冒充完整检索语料。重新生成检索排名需按 manifest 中的固定 NVIDIA revision 下载并校验原始 corpus.zip，再用 prepare_techqa_injection.py 准备全语料。

## 重建统计与图片

在安装了本项目和 NumPy / Matplotlib 的环境中运行：

```bash
PYTHONPATH=src python scripts/summarize_injection.py --snapshot analysis-workspace/online-injection/snapshot --data analysis-workspace/online-injection/data --output analysis-workspace/online-injection/rebuilt
PYTHONPATH=src python scripts/plot_injection_results.py --summary analysis-workspace/online-injection/rebuilt/summary.json --snapshot analysis-workspace/online-injection/snapshot --output analysis-workspace/online-injection/rebuilt/figures
```

语义复核标签是保存的分析，不由关键词重新自动判分。自动规则题评分要求 JSON 决策标签和秒数都正确；TechQA F1 为词项重叠，不能直接解释为语义正确率。原始数据集的 Apache-2.0 声明和来源见归档数据卡；官方规则页的 URL、下载时间与快照哈希见各自 manifest。
'''
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    direct = [details(p.relative_to(DEST).as_posix(),p.read_bytes()) for p in sorted(DEST.rglob('*'))
              if p.is_file() and 'full' not in p.relative_to(DEST).parts]
    manifest = {'schema':1,'source_commit':summary['frozen']['source_commit'],'formal_measurements':312,
                'primary_status':summary['status'],'cache_validation_status':summary['validation']['status'],
                'conditional_cache_status':summary['validation']['conditional_cache']['status'],
                'devices':summary['frozen']['devices'],'retrieved_and_gold_documents':len(needed),
                'direct_files':direct,'archives':[archive]}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(DEST),'archive_bytes':target.stat().st_size,'members':len(members),'direct_files':len(direct)}))


if __name__ == '__main__':
    main()
