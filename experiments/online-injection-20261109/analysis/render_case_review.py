"""Render the complete exploratory source review, checking trajectory identities."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE/'source/src'),str(BASE/'source/scripts')]
from benchmark_injection import arms, grade


def main():
    out = BASE.parents[1]/'outputs/online-injection-a3-20261109'
    raw = [r for s in (BASE/'snapshot-a3/techqa.jsonl').read_text(encoding='utf-8').splitlines()
           if (r:=json.loads(s)).get('kind') == 'measurement']
    rows = {(r['question_id'],r['arm']):r for r in raw}
    path = BASE/'manual-case-review.jsonl'
    notes = {(r['question_id'],r['arm']):r for s in path.read_text(encoding='utf-8').splitlines() if (r:=json.loads(s))}
    assert len(rows) == len(notes) == 104 and rows.keys() == notes.keys()
    assert all(rows[k]['trajectory_sha256'] == notes[k]['trajectory_sha256'] for k in rows)
    questions = [json.loads(s) for s in (BASE/'data/techqa/questions.jsonl').read_text(encoding='utf-8').splitlines()]
    gold = {r['id']:r for s in (BASE/'data/techqa/gold.jsonl').read_text(encoding='utf-8').splitlines() if (r:=json.loads(s))}
    expected = {
        'DEV_Q209':'ACCE 的 Class Definition → Properties → Property Definitions 下拉选择属性，在单独标签页修改，并回到类定义保存。SC 下降方法检索到替代文档 swg21680952，包含关键路径，但没有命中唯一 gold 文档 ID；标注 ID 不是全部可用证据的清单。',
        'DEV_Q195':'同一 HotFix 支持 TLSv1.2。若 appliance 禁用 TLSv1，需手动设置 ITCAM 的 KBN_SOMA_PROTOCOL=TLSv1.2；只调整 appliance Crypto Profile 未回答客户端配置问题。',
        'DEV_Q239':'清理用户 .eclipse/ibm.software.delivery.platform_7.5.0_???? 旧配置目录并重新启动；来源没有把该问题诊断为 JDK7 要求或目录权限。',
        'DEV_Q160':'位置与优先级从低到高为 server → Enterprise Application → Web Application → web.xml → runtime code；控制台子级需选择 Override。原 gold 只摘录了位置，未摘录完整优先级，因此 F1 不能直接判定回答是否覆盖问题。',
        'DEV_Q011':'增大 jextract 自身的 JVM 堆，明确支持的命令参数是 -J-Xmx。来源同时说明工具需与生成 core 的系统/安装匹配。实际 token 解码核对显示，本题全部注入片段均未包含 -J-Xmx；虽命中文档，前部截取没有暴露该关键参数。解码记录和 tokenizer SHA256 见归档 snapshot/evidence-exposure*。',
        'DEV_Q262':'OS 与数据库均 UTF-8 时，CLP 不进行码页转换；若实际输入不是 Unicode/UTF-8，会保存错误码点并在读取时出错。题目的弯引号不能直接当作终止 SQL 字符串的 ASCII 单引号。',
        'DEV_Q184':'同一对 mediation primitives 之间存在视觉重叠的重复连线；移动节点检查，删除冗余线，保存并重新部署。',
        'DEV_Q249':'IBM 公告 swg21998663。来源说明已认证且有 queue manager 权限的用户可能触发段错误和拒绝服务，CVSS 5.3；对应维护版本为 MQ 9.0.0.1 / 8.0.0.6。问句要求信息出处，只有准确摘要而无出处的回答记为部分匹配。',
    }
    labels = {'correct_core':'关键结论符合','partial':'部分匹配','wrong':'与参考不符','empty':'空回答'}
    lines = ['# TechQA：104 份最终回答的来源复核','',
        '这是正式自动评分之外的探索性分析。判断针对给定历史文档的关键规则，不验证回答中的全部额外生产环境建议，也不把 F1 当作语义正确率。它不是独立盲审：第一题未隐藏方法名；之后按答案去重、用散列 ID 隐藏方法名展示，但复核者已见过部分早期输出。方法暴露与主观归类可能产生偏差。未使用这些标签调整模型、检索器或触发阈值。','',
        '“关键结论符合”只表示所问的核心修复/规则得到来源支持；“部分匹配”包括遗漏关键位置、替代命令未被来源验证、截断或混有不可靠额外内容。明确与核心规则相反或没有有效技术内容的回答记为“与参考不符”。空白单独列出。所有标签与原始轨迹 SHA256 关联。','',
        '完整问题、gold 和全部实际引用/gold 文档见压缩包内 data/techqa；完整回答、推理、token 与注入事件见 snapshot/techqa.jsonl。本文件的文档 ID 命中仅按标注计算，不是对所有替代证据的语义评判。']
    for q in questions:
        qid = q['id']
        lines += ['',f"## {qid} — {q['question'].splitlines()[0]}",'',expected[qid],'',
                  '标注文档：'+', '.join('`'+d+'`' for d in gold[qid]['document_ids'])+'。','',
                  '| 方法 | 回答 F1 | 注入偏移 / 标注 ID 命中 | 来源复核 | 说明 |','|---|---:|---|---|---|']
        for mode in arms():
            r,n = rows[qid,mode],notes[qid,mode]
            events = [e for e in r['events'] if e.get('injected')]
            hit = any(h['id'] in gold[qid]['document_ids'] for e in events for h in e['hits'])
            injected = f"{events[0]['actual_generated_offset']} / {'是' if hit else '否'}" if events else '未注入'
            score = grade(r['answer_text'],gold[qid],r['dataset'])['f1']
            lines.append(f"| {mode} | {score*100:.1f}% | {injected} | {labels[n['assessment']]} | {n['note'].replace('|','／')} |")
    (out/'CASE_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'manual-case-review.jsonl').write_bytes(path.read_bytes())
    metadata = {'reviewed':len(notes),'source':'frozen TechQA documents','independent_blinded_review':False,
        'used_to_tune_frozen_methods':False,'label_counts':dict(Counter(n['assessment'] for n in notes.values())),
        'labels_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'all_trajectory_hashes_match':True,'scope':'Source-based exploratory assessment of the central rule, not exhaustive validation of every added claim.'}
    (out/'manual-review-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
