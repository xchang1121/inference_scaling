from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

base=Path(__file__).resolve().parent
repo=base.parent/'online_injection_20261109/source'
package=repo/'experiments/online-injection-smart-is-20261114'
baseline=repo/'experiments/online-injection-full-is-metrics-20261113'
summary=json.loads((base/'reports-formal/summary.json').read_text())
assert summary['measurements']==256 and summary['questions']==16 and summary['audit']['passed']
plot_python=base.parent/'online_injection_literature_20261112/plot-venv/Scripts/python.exe'
subprocess.run([str(plot_python),str(repo/'scripts/plot_smart_online_is.py'),
               '--summary',str(base/'reports-formal/summary.json'),'--output',str(base/'figures')],check=True)
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join([str(repo/'src'),str(base.parent/'online_injection_20261109/devdeps')])
subprocess.run([sys.executable,str(repo/'scripts/diagnose_online_is_failures.py'),
    '--package',str(baseline),'--snapshot',str(base.parent/'online_injection_full_is_metrics_20261113/snapshot/formal/a3'),
    '--output',str(base/'baseline-analysis')],env=env,check=True)
subprocess.run([sys.executable,str(repo/'scripts/diagnose_smart_pilot.py'),
    '--snapshot',str(base/'snapshot/smoke-pilot-v1'),'--output',str(base/'pilot-analysis')],env=env,check=True)
package.mkdir(exist_ok=False)
for folder in ('inputs','analysis','tools','raw/smoke','raw/formal','raw/smoke-failed-v1','raw/smoke-pilot-v1'):
    (package/folder).mkdir(parents=True,exist_ok=True)
for name in ('source-frozen.tgz','audit-smoke.json','audit-smoke-a0.json','audit-smoke-a1.json','audit-smoke-a3.json','local-validation.json','replay.py'):
    shutil.copyfile(base/name,package/name)
for name in ('settings.json','FROZEN_IS.json','ORIGINAL_ONLINE.json','PROTOCOL.json','formal-approval.json','pilot-v1-protocol.json'):
    shutil.copyfile(base/name,package/'inputs'/name)
for name in ('questions.jsonl','smoke-questions.jsonl','gold.jsonl','manifest.json'):
    shutil.copyfile(base/'data'/name,package/'inputs'/name)
for stage in ('smoke','formal','smoke-failed-v1','smoke-pilot-v1'):
    for f in (base/'snapshot'/stage).iterdir():
        if f.is_file():shutil.copyfile(f,package/'raw'/stage/f.name)
for name in ('audit_online_is.py','audit_smart_online_is.py','summarize_cot_guidance.py','summarize_flare_gap.py',
             'summarize_smart_online_is.py','plot_smart_online_is.py','diagnose_online_is_failures.py','diagnose_smart_pilot.py'):
    shutil.copyfile(repo/'scripts'/name,package/'analysis'/name)
for name in ('test_online_is.py','test_online_is_metrics.py','test_smart_online_is.py'):
    shutil.copyfile(repo/'tests'/name,package/'analysis'/name)
for name in ('deploy.py','launch.py','collect.py','monitor_remote.py','approve.py','analyze.py','finalize.py'):
    shutil.copyfile(base/name,package/'tools'/name)
shutil.copytree(base/'reports-formal',package/'reports')
shutil.copytree(base/'figures',package/'figures')
shutil.copytree(base/'baseline-analysis',package/'baseline-analysis')
shutil.copytree(base/'pilot-analysis',package/'pilot-analysis')
shutil.copyfile(base.parent/'online_injection_literature_20261112/plot-requirements.txt',package/'analysis/plot-requirements.txt')
corpus=repo/'experiments/online-injection-literature-20261112/inputs/corpus.jsonl.gz'
(package/'inputs/corpus-reference.json').write_text(json.dumps({'repository_path':corpus.relative_to(repo).as_posix(),
    'sha256':hashlib.sha256(corpus.read_bytes()).hexdigest()},indent=2)+'\n')
(package/'inputs/baseline-reference.json').write_text(json.dumps({'repository_path':baseline.relative_to(repo).as_posix(),
    'manifest_sha256':hashlib.sha256((baseline/'MANIFEST.json').read_bytes()).hexdigest()},indent=2)+'\n')
readme='''# 更智能的在线检索与注入：探索实验

[本轮指标与逐题结果](reports/RESULTS_ZH.md) · [上一轮失败轨迹诊断](baseline-analysis/BASELINE_DIAGNOSIS_ZH.md) · [TRAIN格式对照](pilot-analysis/PILOT_FORMAT_ZH.md)

![质量与实测成本](figures/overview.png)

固定16道DEV题、2个种子、8组，共256次全新运行。所有条件按问题在同一张卡上配对；本轮使用A0/A1/A3。
六种在线策略包含：继续推理提示、原问题占优的双查询融合与窗口去重、句末/换行优先注入、不确定性门控、未完成的第一人称续写桥接、提前到32/256的桥接。
静态事实与原在线注入均重新运行作为基线。详见 inputs/PROTOCOL.json。

原IS内核、SC评分、模型设置及预算不变；没有工具调用、额外检索代理生成或固定思考前缀。
保留自然EOS、空答案和截断，不重试或补答。输入证据及继续推理提示不计入思考token或SC评分，但记录输入成本。
正式生成前完成48次TRAIN原生完整性验证；此前另有36次六组TRAIN试验。本地测试与逐文件冻结信息见 local-validation.json。

首次TRAIN启动因runner参数签名不匹配，在第一条测量之前退出；失败日志与当时源码保存在 raw/smoke-failed-v1。
修正入口后，六组TRAIN试验发现命令式提示仍可能直接EOS，因此保留全部六组并增加第一人称桥接、提前桥接两组；此前试验完整保存在 raw/smoke-pilot-v1。最终八组再次完成48次完整性验证；没有丢弃或重跑任何正式测量。

## 复现

安装项目基础依赖后，在本目录运行（目标目录需不存在）：

```sh
python replay.py --destination /path/to/new/replay
```

脚本验证逐文件SHA-256，再用冻结源码重放候选请求、分支注入、概率门控、IS评分选择和答案读出，重新计算指标并要求逐字节一致。
加 `--plot-python /path/to/plot/python` 可重放图表，依赖版本见 analysis/plot-requirements.txt。
加 `--baseline-package ../online-injection-full-is-metrics-20261113` 可重放上一轮诊断的全部轨迹导出和报告。
原生重跑还需要既有模型、Ascend/vLLM环境和 inputs/corpus-reference.json 所指冻结语料。
tools 是原工作区部署脚本的来源记录；实际启动脚本与设备状态包含在原始归档中。

本轮方案基于已查看的DEV结果设计，因此属于开发集探索，不是独立留出测试。
F1为词级重叠，不能替代人工语义正确率。区间未对多重比较校正；边界策略还改变原生分段和派生随机流。
'''
(package/'README.md').write_text(readme,encoding='utf-8',newline='\n')
manifest={'files':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(package.rglob('*')) if p.is_file()}}
(package/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
env['PYTHONPATH']=str(base.parent/'online_injection_20261109/devdeps')
subprocess.run([sys.executable,str(package/'replay.py'),'--destination',str(base/'offline-replay-final'),
               '--plot-python',str(plot_python),'--baseline-package',str(baseline)],env=env,check=True)
print(json.dumps({'package':str(package),'manifest_files':len(manifest['files']),'audit_checks':summary['audit']['checks']}))
