"""Analysis-only length metadata. Never modifies model inputs or generation locks."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from extract_evidence import TEX
from sentence_units import count_units
from protocol import atomic_json, digest, lock_json, read_json

VERSION = 'explanation-length-report-v1'
BINS = ['1', '2', '3', '4', '5']
PLAN = {
    'version': VERSION,
    'role': 'Exploratory descriptive supplement; not a new primary outcome or success test.',
    'count': 'Deterministic punctuation-based prose units in the normalized explanation actually supplied. Protect TeX, decimals and common abbreviations; split paragraphs/list entries; count nonempty trailing prose fragments once. This is an automatic sentence-length proxy, not linguistically certified sentence annotation.',
    'math': 'Standalone TeX displays do not count as prose sentences; count display and inline TeX spans separately. Short headings and list fragments are counted and flagged as unpunctuated units.',
    'bins': BINS,
    'bin_rule': '1, 2, 3, 4, 5 automatic prose units. Fixed before examining new R2 outcomes; do not choose cutoffs from favorable effects.',
    'pairing': 'Assign each question by its R2 explanation length, then compare R1/S0, R1/S1, R2/S0, R2/S1 on the identical IDs in each bin. R1 has zero supplied explanation but inherits the matching question bin.',
    'populations': 'Initial wrong: correction; initial correct: retention. Initial invalid: length inventory and format checks only.',
    'unit': 'Question. Three phrasings per question are averaged before paired contrasts and subject-stratified resampling.',
    'intervals': 'Descriptive pointwise 95% intervals only; no multiplicity-adjusted significance claims or between-bin hypothesis tests. Flag bins with fewer than 20 questions. Original conservative interval fallback applies to tiny/constant samples.',
    'contrasts': ['R1 S1-S0', 'R2 S1-S0', '(R2 S1-S0)-(R1 S1-S0)'],
    'limits': 'Different lengths belong to different questions, subjects, difficulty levels and explanation structures. Differences across bins do not identify a causal effect of length. All max5 inputs and the primary analysis remain unchanged.',
}




def length_bin(n):
    if not 1 <= n <= 5:
        raise ValueError('Explanation has no prose units')
    return str(n)


def item_lengths(manifest):
    rows = []
    for r in manifest['excerpts']:
        units = count_units(r['text'])
        rows.append({'question_id': r['question_id'], 'subject': r['subject'],
                     'initial_cohort': r['initial_cohort'], **units,
                     'sentence_bin': length_bin(units['sentence_count_auto']),
                     'word_count_whitespace': len(r['text'].split()),
                     'character_count': len(r['text']), 'max_R2_input_tokens': r['max_R2_input_tokens'],
                     'explanation_sha256': hashlib.sha256(r['text'].encode()).hexdigest()})
    frame = pd.DataFrame(rows).sort_values('question_id').reset_index(drop=True)
    if frame.question_id.duplicated().any():
        raise ValueError('Duplicate question IDs')
    return frame


def inventory_tables(lengths):
    cols = ['sentence_count_auto', 'word_count_whitespace', 'display_math_blocks', 'max_R2_input_tokens']
    summary = lengths[cols].agg(['min', 'median', 'max']).T.reset_index().rename(columns={'index': 'measure'})
    counts = pd.crosstab(lengths.sentence_bin, lengths.initial_cohort).reindex(index=BINS, columns=['wrong', 'correct', 'invalid'], fill_value=0)
    counts['total'] = counts.sum(axis=1)
    return summary, counts.reset_index()


def prepare_length_reports(root, manifest):
    """Separate length-reporting lock within the max5-v3 study; do not reuse full-v2 R2."""
    root = Path(root)
    folder = root/'reports'
    folder.mkdir(parents=True, exist_ok=True)
    data = item_lengths(manifest)
    reproduction = root/'length_reporting_v1'
    reproduction.mkdir(exist_ok=True)
    code = Path(__file__).read_bytes()
    lock_json(reproduction/'analysis_lock.json', {
        'plan': PLAN, 'code_sha256': hashlib.sha256(code).hexdigest(),
        'manifest_hash': manifest['manifest_hash'], 'length_rows_hash': digest(data.to_dict(orient='records')),
        'model_inputs_changed': False, 'generation_lock_changed': False,
    })
    target = reproduction/'length_analysis.py'
    if target.exists() and target.read_bytes() != code:
        raise RuntimeError('Length analysis code changed within this analysis version')
    if not target.exists():
        target.write_bytes(code)
    data.to_csv(folder/'explanation_lengths.csv', index=False, encoding='utf-8-sig')
    summary, counts = inventory_tables(data)
    summary.to_csv(folder/'length_summary.csv', index=False, encoding='utf-8-sig')
    counts.to_csv(folder/'length_bin_counts.csv', index=False, encoding='utf-8-sig')
    data.groupby(['sentence_bin', 'subject', 'initial_cohort']).size().rename('n_items').reset_index().to_csv(
        folder/'length_subject_counts.csv', index=False, encoding='utf-8-sig')
    return data, summary, counts


def analyze_lengths(root, manifest, cfg):
    """Read already scored responses; never creates new model requests."""
    from evidence_analysis import estimate_contrast
    root = Path(root)
    folder = root/'reports'
    lengths, summary, counts = prepare_length_reports(root, manifest)
    frame = pd.read_csv(folder/'scored_responses.csv')
    if set(frame.question_id) != set(lengths.question_id) or len(frame) != 12*len(lengths):
        raise RuntimeError('All four conditions with three phrasings are required')
    if frame.duplicated(['question_id', 'condition', 'defense', 'variant']).any():
        raise RuntimeError('Duplicate paired response')
    extra = lengths.drop(columns=['subject'])
    frame = frame.merge(extra, on='question_id', how='left', validate='many_to_one')
    # The matching-length variable is not text actually supplied in R1.
    frame['supplied_explanation_sentence_count_auto'] = frame.sentence_count_auto.where(frame.condition == 'R2', 0)
    frame['supplied_explanation_word_count'] = frame.word_count_whitespace.where(frame.condition == 'R2', 0)
    frame.to_csv(folder/'scored_responses_with_lengths.csv', index=False, encoding='utf-8-sig')
    cells, effects = [], []
    specs = [
        ('R1_defense', [('R1','S1',1), ('R1','S0',-1)]),
        ('R2_defense', [('R2','S1',1), ('R2','S0',-1)]),
        ('interaction', [('R2','S1',1), ('R2','S0',-1), ('R1','S1',-1), ('R1','S0',1)]),
    ]
    for cohort, correct, metric in [('wrong',False,'correction'), ('correct',True,'retention')]:
        for bucket in BINS:
            sub = frame[(frame.initial_cohort == cohort) & (frame.sentence_bin == bucket)]
            if sub.empty:
                continue  # Empty bins remain visible in length_bin_counts.csv.
            n = sub.question_id.nunique()
            for condition in ['R1','R2']:
                for defense in ['S0','S1']:
                    f = sub[(sub.condition == condition) & (sub.defense == defense)]
                    if len(f) != 3*n or f.question_id.nunique() != n:
                        raise RuntimeError('Length-bin paired cell incomplete')
                    cells.append({'initial_cohort':cohort, 'metric':metric, 'sentence_bin':bucket,
                                  'condition':condition, 'defense':defense, 'n_items':n,
                                  'n_responses':len(f), 'correct_count':int(f.final_correct.sum()),
                                  'correct_percent':100*float(f.final_correct.mean()),
                                  'invalid_count':int(f.invalid.sum()), 'small_bin_under_20':n < 20,
                                  'analysis_role':'exploratory_descriptive'})
            for name, terms in specs:
                row, _ = estimate_contrast(sub, terms, 'final_correct', correct, cfg,
                    f'length_{cohort}_{bucket}_{name}', 'exploratory_descriptive', name)
                row.update(sentence_bin=bucket, initial_cohort=cohort, outcome=metric, small_bin_under_20=n < 20)
                effects.append(row)
    cells, effects = pd.DataFrame(cells), pd.DataFrame(effects)
    cells.to_csv(folder/'length_group_results.csv', index=False, encoding='utf-8-sig')
    effects.to_csv(folder/'length_group_effects.csv', index=False, encoding='utf-8-sig')
    synthetic = read_json(folder/'diagnostic_summary.json').get('execution_mode') == 'SYNTHETIC_TEST_ONLY'
    atomic_json(folder/'length_analysis_metadata.json', {
        'execution_mode': 'SYNTHETIC_TEST_ONLY' if synthetic else 'analysis_of_saved_responses',
        'plan': PLAN, 'n_items':len(lengths), 'n_responses':len(frame), 'additional_generations':0})
    (folder/'문장수_분석_읽는법.txt').write_text(
        ('합성 점검용이며 실제 실험 결과가 아닙니다.\n' if synthetic else '')+
        'explanation_lengths.csv: 문항별 실제 입력 풀이의 자동 문장 수·수식 수·단어 수·최대 입력 토큰 수.\n'
        'sentence_count_auto는 문장부호 기반 자동 집계입니다. 문장부호 없는 표제·목록 조각도 한 단위로 세며 unpunctuated_units로 표시합니다. 수식만 있는 블록은 문장에 포함하지 않습니다.\n'
        '단어 수는 공백 단위로 세므로 수식의 구성 요소도 포함할 수 있습니다. 입력 토큰 수는 풀이뿐 아니라 전체 대화입니다.\n'
        'scored_responses_with_lengths.csv: 모든 응답과 길이를 연결한 표. R1의 실제 제공 풀이 길이는 0입니다. sentence_bin은 대응하는 R2 풀이를 기준으로 붙입니다.\n'
        'length_group_results.csv: 1 / 2 / 3 / 4 / 5개 자동 문장 단위별 수정률·유지율. 같은 구간의 같은 문항끼리 네 조건을 비교합니다.\n'
        'length_group_effects.csv: 각 구간 R1 방어 효과, R2 방어 효과와 두 효과의 차이. 표현 3개를 문항 내 평균한 기술적 95% 구간이며 유의성 검정을 추가한 것이 아닙니다.\n'
        '20문항 미만인 구간은 small_bin_under_20=True로 표시합니다. 구간이 넓거나 0을 포함한다고 효과 없음·동등성으로 결론 내리지 않습니다.\n'
        '긴 풀이와 짧은 풀이는 서로 다른 문제입니다. 과목·난도·풀이 내용도 달라서 길이 자체의 인과효과를 말할 수 없습니다.\n'
        'length_subject_counts.csv에서 구간별 과목 구성을 확인할 수 있습니다. 길이 분석은 보조 분석이고 기존 전체 77문항 주 분석을 대신하지 않습니다.\n', encoding='utf-8')
    return cells, effects
