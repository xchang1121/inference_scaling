"""Write the Chinese interpretation only after the complete formal audit passes."""
from pathlib import Path
import argparse
import json

base=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--summary',type=Path,default=base/'reports-formal/summary.json')
p.add_argument('--output',type=Path,default=base/'reports-formal/RESULTS_ZH.md')
a=p.parse_args()
result=json.loads(a.summary.read_text(encoding='utf-8'))
assert result['measurements']==352 and result['questions']==16 and result['audit']['passed']
rows={r['arm']:r for r in result['summary']}
contrasts={(c['left'],c['right']):c for c in result['contrasts']}
names={'direct':'普通直接生成','closed_is':'无注入 IS','static_facts':'静态事实 + IS',
    'static_facts_noise':'静态事实等长噪声','ircot':'IRCoT + IS','ircot_noise':'IRCoT 等长噪声',
    'static_examples':'静态完整样例 + IS','static_examples_noise':'静态样例等长噪声',
    'thinkretrieve':'ThinkRetrieve + IS','thinkretrieve_noise':'ThinkRetrieve 等长噪声','rethink':'仅重新思考 + IS'}

def contrast(a,b):
    c=contrasts[a,b];lo,hi=c['bootstrap_95']
    return f"{names[a]} − {names[b]}：{c['difference']:+.3f} F1 点，95% 区间 [{lo:+.3f}, {hi:+.3f}]"

text=['# IRCoT / ThinkRetrieve 正式实验结果','',
    '同一组 16 道 TechQA DEV 题 × 2 个随机种子 × 11 个实验组，共 352 项完整原生测量。'
    'IS 算法、Self-Certainty 奖励及参数保持不变；本次只改变检索和注入控制器。', '',
    '本次结果支持检索到的事实内容有用：IRCoT 和静态事实注入均优于各自等长噪声。'
    'IRCoT 的平均 F1 最高，但相对静态事实注入的增益区间跨零，尚不能据此确认动态检索稳定更好。'
    '静态事实注入在本次测量中耗时更少，可作为后续扩样的成本基线。', '',
    'ThinkRetrieve 在当前模型与 TechQA 适配下没有带来收益，平均 F1 低于无注入 IS 和静态样例注入，且耗时更长。'
    '真实样例组有 13/32 个空答案，噪声组有 20/32 个；终止位置分别为控制器思考阶段 9/11 个、'
    'IS 选中思考轨迹 4/9 个。这定位了失败阶段，但不能单独证明模板、检索内容或奖励造成了 EOS。'
    '逐例信息见 [空答案诊断](../../diagnostics/EMPTY_ANSWERS.md)。', '',
    '| 方法 | F1（0–100） | 平均总生成 token | token / 直接生成 | 平均耗时（秒） | 耗时 / 直接生成 | 空答案 / 32 | 最终截断 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for arm,r in rows.items():
    text.append(f"| {names[arm]} | {r['f1']:.3f} | {r['mean_generation_tokens']:,.1f} | {r['mean_generation_tokens']/rows['direct']['mean_generation_tokens']:.3f}× | {r['mean_seconds']:.2f} | {r['latency_ratio_of_means']:.3f}× | {r['empty']} | {r['truncated']} |")
text+=['','总生成 token 包括检索代理、临时答案、探测、所有被丢弃的 IS 候选及最终答案。'
    '注入文本属于输入，单独记录。所有 IS 组共享 65,536 token 的搜索与控制器总预算，'
    '保留思考上限为 8,192 token；普通直接生成是单条思考轨迹，上限同为 8,192 token，不执行候选搜索。'
    '每组最终答案上限均为 16,384 token。实际消耗并不强行相等，直接生成用于提供成本与延迟基准。'
    '耗时包含完整检索与生成流水线；每组开始前清空 prefix cache，模型加载、一次性索引构建和预热不计入该均值。',
    '', '## 主要配对比较','']
for pair in [('closed_is','direct'),('ircot','closed_is'),('ircot','static_facts'),
             ('thinkretrieve','closed_is'),('thinkretrieve','static_examples'),('thinkretrieve','rethink')]:
    text.append('- '+contrast(*pair)+'。')
text+=['','## 内容是否比等长无关文本有用','']
for pair in [('static_facts','static_facts_noise'),('ircot','ircot_noise'),
             ('static_examples','static_examples_noise'),('thinkretrieve','thinkretrieve_noise')]:
    text.append('- '+contrast(*pair)+'。')
matched=result['audit']['noise_pair_count']-len(result['audit']['noise_length_mismatches'])
text+=['',f"实际注入长度完全匹配 {matched}/{result['audit']['noise_pair_count']} 对。"
    'ThinkRetrieve 噪声组使用自己的思考轨迹和临时查询，以参考组相同注入次序的长度替换整个样例内容；'
    '如果真实 EOS 提前终止，缺失的后续注入保留为不匹配记录，不补造生成。'
    'IRCoT 噪声组复跑同一相关检索代理，只替换独立 reader 接收的事实内容。'
    '11 对不匹配全部来自 ThinkRetrieve 噪声组在第二次注入前 EOS，已发生的注入包仍逐个精确等长。'
    '因此 ThinkRetrieve 的 32 对中只有 21 对总注入长度相同；其总体对比不能视为严格等长的内容消融，'
    '也不应删除早停样本后再解释为无偏的内容效果。其余三类对照各 32/32 对精确匹配。',
    '', '## EOS 与注入后的继续生成','',
    '| 方法 | 注入事件 | 注入后立即结束思考 | 注入后立即 EOS | 最后一次注入后的平均思考 token |',
    '|---|---:|---:|---:|---:|']
for arm,r in rows.items():
    if not r['injection_episodes']:continue
    n=r['mean_thinking_after_last_injection'];shown='—' if n is None else f'{n:.1f}'
    text.append(f"| {names[arm]} | {r['injection_episodes']} | {r['post_injection_immediate_close_events']} | {r['post_injection_immediate_eos_events']} | {shown} |")
text+=['','思考结束标记 `</think>` 与 EOS 分开统计。表中事件以实际插入位置为起点，'
    '不把之后 IS 后缀的立即结束混称为注入后立即结束。所有空答案按零分进入均值；'
    '若某组因 EOS 少生成而更快，不能将较低耗时单独解释为有效推理效率提升。',
    '', '## 协议适配与边界','',
    'IRCoT 使用 BM25 的问题检索及最多两轮“生成下一句—检索”，随后由全新 IS reader 读取累计事实。'
    'ThinkRetrieve 保留思考前缀，在自然思考结束或每 2,048 token 处生成临时答案，'
    '以 E5-base-v2 的问题与答案联合向量检索完整 TRAIN 操作样例，最多插入两次，再交给相同 IS。'
    '它们的比较覆盖整套适配流水线，不能解释为只改变插入位置的纯因果效应。',
    '', '正式版使用 ThinkRetrieve 官方模板及样例格式。初版改写模板在两道 TRAIN 冒烟题上出现早停；'
    '独立冷缓存诊断复现该现象，官方模板在其中一道题上恢复正常继续生成，因此在正式运行前恢复原模板。'
    '旧版 22 项结果、六次诊断以及全部源码已保留，随后完整重跑 22 项 TRAIN 冒烟。'
    '没有按当前 DEV 输出或 TRAIN F1 调整模板，也没有屏蔽 EOS。',
    '', '这是一组已在前序实验使用过的小规模 DEV 题，结论属于该设置下的探索性机制结果，'
    '并非论文基准复现，也不代表未见测试集上的普遍收益。置信区间按题聚类：先平均两个种子，'
    '再做 5,000 次题级 bootstrap；没有做多重比较校正。区间跨零时不据此宣称稳定提升。',
    '', '## 审计与复核','',
    f"全部 {result['audit']['checks']:,} 项离线审计通过；最终截断 {result['audit']['final_truncated']} 项。"
    '审计重算生成成本、前缀拼接、Self-Certainty 奖励、IS 候选权重与抽样选择，'
    '并核对冻结文件哈希、题目与种子集合、噪声内容及长度。',
    '', f"正式测量累计生成 {result['total_generation_tokens']:,} token，流水线累计 {result['total_measured_pipeline_seconds']/3600:.3f} 小时；"
    f"含启动和退出的原生任务占用 {sum(j['seconds'] for j in result['jobs'])/3600:.3f} 设备小时。", '',
    '逐项答案与成本见 [answers.jsonl](answers.jsonl)，完整配对区间和额外诊断见 [REPORT.md](REPORT.md)，'
    '审计明细见 [audit.json](audit.json)。发布包提供原始 token 轨迹、冻结源码和无需模型的离线复算脚本。']
a.output.write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
print('Chinese interpretation written after full formal audit.')
