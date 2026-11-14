from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

base = Path(__file__).resolve().parent
repo = base.parent/'online_injection_20261109/source'
package = repo/'experiments/online-injection-full-is-metrics-20261113'
summary = json.loads((base/'reports-formal/summary.json').read_text())
assert summary['measurements'] == 192 and summary['questions'] == 16 and summary['audit']['passed']
plot_python = base.parent/'online_injection_literature_20261112/plot-venv/Scripts/python.exe'
subprocess.run([str(plot_python), str(repo/'scripts/plot_online_is.py'), '--summary', str(base/'reports-formal/summary.json'),
    '--output', str(base/'figures')], check=True)
package.mkdir(exist_ok=False)
for folder in ('inputs', 'analysis', 'raw/smoke', 'raw/formal', 'tools'):
    (package/folder).mkdir(parents=True, exist_ok=True)
for name in ('source-frozen.tgz', 'audit-smoke.json', 'local-validation.json', 'replay.py'):
    shutil.copyfile(base/name, package/name)
for name in ('settings.json', 'FROZEN_IS.json', 'PROTOCOL.json', 'formal-approval.json'):
    shutil.copyfile(base/name, package/'inputs'/name)
for name in ('questions.jsonl', 'smoke-questions.jsonl', 'gold.jsonl', 'manifest.json'):
    shutil.copyfile(base/'data'/name, package/'inputs'/name)
for stage in ('smoke', 'formal'):
    for file in (base/'snapshot'/stage).iterdir():
        if file.is_file():
            shutil.copyfile(file, package/'raw'/stage/file.name)
for name in ('audit_online_is.py', 'summarize_online_is.py', 'summarize_cot_guidance.py', 'summarize_flare_gap.py', 'plot_online_is.py'):
    shutil.copyfile(repo/'scripts'/name, package/'analysis'/name)
for name in ('test_online_is.py', 'test_online_is_metrics.py'):
    shutil.copyfile(repo/'tests'/name, package/'analysis'/name)
for name in ('deploy.py', 'launch.py', 'collect.py', 'monitor_remote.py', 'approve.py', 'analyze.py', 'finalize.py'):
    shutil.copyfile(base/name, package/'tools'/name)
shutil.copytree(base/'reports-formal', package/'reports')
shutil.copytree(base/'figures', package/'figures')
shutil.copyfile(base.parent/'online_injection_literature_20261112/plot-requirements.txt', package/'analysis/plot-requirements.txt')
old = repo/'experiments/online-injection-literature-20261112/inputs/corpus.jsonl.gz'
(package/'inputs/corpus-reference.json').write_text(json.dumps({'repository_path': old.relative_to(repo).as_posix(),
    'sha256': hashlib.sha256(old.read_bytes()).hexdigest()}, indent=2)+'\n')
readme = '''# 从起点 IS 在线注入：指标评测

[中文结果与配对分析](reports/RESULTS_ZH.md) · [结构化汇总](reports/summary.json) · [逐题逐种子分数](reports/scores.csv)

![质量、生成成本与耗时](figures/overview.png)

16 道固定 DEV 题、2 个种子、6 组，共 192 次全新原生运行；另有 12 次 TRAIN 完整性验证。
原 IS 算法、参数与在线注入实现未改动；本轮增加静态／等长噪声对照。
原生 EOS、空答案和截断保留。运行数据不包含 gold；参考答案仅在离线分析时读取。

包内保留冻结生成源码、配置、题目、原生轨迹、成本、源文件哈希与图表。
语料复用同仓库之前冻结的副本，位置和校验值见 inputs/corpus-reference.json。
tools 保存原工作区的部署脚本用于追溯，路径依赖原环境；raw 另有实际启动脚本和设备记录。

## 离线重放

安装项目基础依赖的 Python 3.12+ 环境中，在本目录执行（目标目录必须不存在）：

```sh
python replay.py --destination /path/to/new/replay
```

脚本先验证逐文件 SHA-256，再展开原生记录和冻结源码，重新执行完整审计与全部指标计算，要求输出逐字节一致。
可加 `--plot-python /path/to/plot/python` 重放并逐字节校验图表；对应依赖版本见 analysis/plot-requirements.txt。

这批题此前用于开发对照，不是新的独立测试集。F1 是词级重叠指标，不能替代人工语义正确率。
'''
(package/'README.md').write_text(readme, encoding='utf-8', newline='\n')
manifest = {'files': {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(package.rglob('*')) if p.is_file()}}
(package/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
env = dict(os.environ)
env['PYTHONPATH'] = str(base.parent/'online_injection_20261109/devdeps')
subprocess.run([sys.executable, str(package/'replay.py'), '--destination', str(base/'offline-replay-final'),
    '--plot-python', str(plot_python)], env=env, check=True)
print(json.dumps({'package': str(package), 'manifest_files': len(manifest['files']), 'audit_checks': summary['audit']['checks']}))
