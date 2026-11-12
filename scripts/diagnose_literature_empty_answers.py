"""Locate empty-answer termination stages in a complete, audited native run."""
from pathlib import Path
import argparse
import csv
import json
from collections import Counter

EOS = 248046
PHASES = {
    'controller_thought_eos': 'Retrieval/continuation controller thinking',
    'checkpoint_thought_eos': 'Initial thinking checkpoint',
    'direct_thought_eos': 'Direct-generation thinking',
    'is_selected_thought_eos': 'Selected IS thinking trajectory',
    'final_answer_empty_stop': 'Final answer generation',
}


def classify(native, audited):
    """Report the observed terminal segment, not an inferred cause of failure."""
    answer, handoff = native['answer'], native['handoff']
    if not answer['empty_answer']:
        return None
    assert not answer['answer_text'].strip() and audited['empty'] and audited['f1'] == 0
    if handoff['terminal_eos']:
        phase = 'controller_thought_eos'
        terminal = handoff['controller_calls'][-1]['token_ids']
        assert EOS in terminal
    elif native['audit'].get('terminal_checkpoint'):
        phase = 'checkpoint_thought_eos'
        terminal = native['probe']['reason_ids']
        assert EOS in terminal
    elif answer['eos_inside_thinking']:
        phase = 'direct_thought_eos' if native['arm'] == 'direct' else 'is_selected_thought_eos'
        terminal = native['candidates'][0]
        assert EOS in terminal
    else:
        phase = 'final_answer_empty_stop'
        terminal = answer['answer_ids']
        assert answer['calls']
    return {'unit_id': native['unit_id'], 'question_id': native['question_id'],
        'seed': native['seed'], 'arm': native['arm'], 'terminal_phase': phase,
        'terminal_segment_tokens': len(terminal),
        'terminal_eos_position_zero_based': terminal.index(EOS) if EOS in terminal else None,
        'injections': len(handoff['injected_ids']),
        'retained_thinking_before_is': len(native['carried']['sc']),
        'selected_suffix_generated_tokens': 0 if handoff['terminal_eos'] else len(native['candidates'][0]),
        'interim_answer_calls': len(handoff['interim_calls']),
        'interim_answer_tokens': sum(len(c['token_ids']) for c in handoff['interim_calls']),
        'generation_tokens': audited['generation_tokens'], 'seconds': audited['seconds'],
        'final_answer_tokens': len(answer['answer_ids']), 'truncated': answer['truncated']}


def run(snapshot, summary_path, answers_path, output):
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['stage'] == 'formal' and summary['audit']['passed']
    assert summary['questions'] == 16 and summary['measurements'] == 352
    answers = [json.loads(line) for line in answers_path.read_text(encoding='utf-8').splitlines()]
    audited = {r['unit_id']: r for r in answers}
    assert len(audited) == len(answers) == 352
    native = {}
    for path in sorted(snapshot.glob('*/formal.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['kind'] == 'measurement':
                assert row['unit_id'] not in native
                native[row['unit_id']] = row
    assert set(native) == set(audited), 'require the complete audited native cohort'
    events = []
    for uid in sorted(native):
        event = classify(native[uid], audited[uid])
        if event is not None:
            events.append(event)
    assert len(events) == sum(r['empty'] for r in summary['summary'])
    counts = Counter((event['arm'], event['terminal_phase']) for event in events)
    result = {'measurements': 352, 'empty_answers': len(events), 'counts':
        [{'arm': arm, 'terminal_phase': phase, 'count': count}
         for (arm, phase), count in sorted(counts.items())], 'events': events}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'empty_answers.json').write_text(json.dumps(result, indent=2) + '\n',
                                               encoding='utf-8', newline='\n')
    fields = list(events[0]) if events else ['unit_id', 'arm', 'terminal_phase']
    with (output / 'empty_answers.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    text = ['# Empty-answer termination diagnostics', '',
        f'{len(events)} of 352 native measurements have an empty final answer.', '',
        'Empty means the final answer is an empty string after removing special tokens and '
        'surrounding whitespace. It does not mean that no thinking, retrieval or interim '
        'answer was generated. All empty answers receive zero F1 and retain their full costs.', '',
        '| Arm | Observed terminal phase | Empty answers |', '|---|---|---:|']
    for row in result['counts']:
        text.append(f"| {row['arm']} | {PHASES[row['terminal_phase']]} | {row['count']} |")
    if not events:
        text.append('| All arms | No empty final answers | 0 |')
    text += ['', '## Interpretation', '',
        '- This records where generation ended, not why the model selected EOS. It cannot '
        'by itself establish that a template, retrieved content or reward caused the stop.',
        '- A controller EOS is handled before IS starts. The runner materializes an EOS '
        'sentinel for the common final-answer interface; that sentinel is not an IS-generated token. '
        'These cases therefore report zero selected-suffix generated tokens.',
        '- A selected IS trajectory ending in EOS may follow substantial search, including '
        'discarded candidates. Its short selected suffix does not imply low total compute.',
        '- Interim answers drive retrieval and are not substituted for a missing final answer.',
        '- Terminal positions are zero-based within the observed terminal segment. '
        'The CSV and JSON retain per-run costs and actual insertion counts.', '']
    (output / 'EMPTY_ANSWERS.md').write_text('\n'.join(text), encoding='utf-8', newline='\n')
    print(json.dumps({'measurements': 352, 'empty_answers': len(events), 'counts': result['counts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('snapshot', 'summary', 'answers', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.summary, args.answers, args.output)
