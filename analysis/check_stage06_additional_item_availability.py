"""Read-only availability audit; frozen max5 rules, no new model inference."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'additional_item_availability'
OUT.mkdir(exist_ok=True)
CODE = ROOT / 'output/jupyter-notebook/sycophancy_qwen_8b_evidenceB_max5_support'
sys.path.insert(0, str(CODE))
from extract_evidence import normalize_explanation, POLICY
from protocol import make_pool

source = json.loads((CODE / 'source/source_snapshot.json').read_text())
manifest = json.loads((CODE / 'source/evidence_manifest.json').read_text())
orig = {i['question_id'] for i in source['study']['items']}
cohorts = {r['question_id']: r['initial_cohort'] for r in json.loads((CODE / 'source/official_matched.json').read_text())}
selected = {r['question_id']: r['text'] for r in manifest['excerpts']}
subjects = set(source['study']['config']['subjects'])
pq = ROOT / 'output/jupyter-notebook/sycophancy_qwen_support/validation/source_test.parquet'
data = pd.read_parquet(pq)
assert not data.question_id.duplicated().any()
lookup = {int(r['question_id']): r for r in data.to_dict(orient='records')}
pool = make_pool(data.to_dict(orient='records'), source['study']['config'])
assert {i['question_id'] for i in pool['pools']['main']} == orig
phase = {i['question_id']: p for p, items in pool['pools'].items() for i in items}
phase.update({i['question_id']: 'reserve' for items in pool['reserves'].values() for i in items})
original_exclusions = {'mmlupro:' + r['question_id']: r['reason'] for r in pool['exclusions']}
archive = ROOT / 'tmp/mmlu_official_sources/model_outputs_gpt4o(2024-05-13)_5shots.zip'
archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_hash == manifest['source']['archive_sha256']
with zipfile.ZipFile(archive) as z:
    rows = json.loads(z.read('model_outputs_gpt4o(2024-05-13)_5shots.json'))
assert len(rows) == len({r['question_id'] for r in rows}) == 12032
norm = lambda x: ' '.join(str(x).split())
audit = []
for n, row in enumerate(rows, 1):
    qid = 'mmlupro:' + str(row['question_id'])
    official = lookup.get(row['question_id'])
    matches = official is not None and norm(row['question']) == norm(official['question']) and [norm(v) for v in row['options']] == [norm(v) for v in official['options']] and row['answer'] == official['answer'] and row['category'] == official['category']
    same_subject = row['category'] in subjects
    teacher_correct = row['pred'] == row['answer']
    reason = None
    text = None
    if not matches:
        reason = 'dataset_question_options_or_gold_mismatch'
    elif not same_subject:
        reason = 'outside_original_six_subjects'
    elif qid not in phase:
        reason = 'original_item_filter:' + original_exclusions[qid]
    elif not teacher_correct:
        reason = 'teacher_final_label_wrong'
    else:
        labels = list('ABCDEFGHIJ'[:len(row['options'])])
        text, reason, _ = normalize_explanation(row['response'], row['options'], labels)
    if qid in orig:
        assert (text is not None) == (qid in selected), qid
        if text is not None:
            assert text == selected[qid], qid
    audit.append({'question_id': qid, 'subject': row['category'], 'original_300': qid in orig,
                  'initial_cohort': cohorts.get(qid, 'not_measured_in_original_300'),
                  'dataset_match': matches, 'same_six_subjects': same_subject,
                  'original_sampling_pool': phase.get(qid, 'not_in_original_sampling_pool'),
                  'teacher_final_correct': teacher_correct, 'max5_text_candidate': text is not None,
                  'reason': reason or 'passes_frozen_text_filters',
                  'excerpt_sha256': hashlib.sha256(text.encode()).hexdigest() if text else ''})
    if n % 1500 == 0:
        print('Checked', n, '/', len(rows), flush=True)
f = pd.DataFrame(audit)
f.to_csv(OUT / 'all_item_availability.csv', index=False, encoding='utf-8-sig')
fresh = f[(f.original_sampling_pool == 'reserve') & f.max5_text_candidate]
fresh.to_csv(OUT / 'additional_max5_candidates.csv', index=False, encoding='utf-8-sig')
original = f[f.original_300]
original.groupby(['initial_cohort','reason']).size().rename('count').reset_index().to_csv(OUT/'original_300_exclusion_counts.csv',index=False,encoding='utf-8-sig')
counts = f[f.same_six_subjects].groupby('subject').agg(total=('question_id','size'),teacher_correct=('teacher_final_correct','sum'),max5_candidates=('max5_text_candidate','sum')).reset_index()
counts['additional_outside_original_300'] = counts.subject.map(fresh.groupby('subject').size()).fillna(0).astype(int)
counts.to_csv(OUT/'same_subject_counts.csv',index=False,encoding='utf-8-sig')
report = {
    'scope': 'Availability only. No Qwen inference, no notebook changes, no relaxed extraction rules.',
    'dataset_rows': len(data), 'teacher_rows': len(rows), 'dataset_matches': int(f.dataset_match.sum()),
    'same_six_subjects_total': int(f.same_six_subjects.sum()),
    'same_six_subjects_teacher_final_correct': int((f.same_six_subjects & f.teacher_final_correct & f.dataset_match).sum()),
    'same_six_subjects_max5_text_candidates': int(f.max5_text_candidate.sum()),
    'original_300_max5': int(original.max5_text_candidate.sum()),
    'original_max5_texts_exactly_reproduced': len(selected),
    'additional_candidates_excluding_original_300_smoke6_and_pilot60': len(fresh),
    'additional_candidate_Qwen_baselines_available_in_original_run': 0,
    'new_candidate_initial_correct_or_wrong_counts': 'unknown until Qwen initial responses are generated',
    'new_candidate_full_prompt_token_validation': 'not performed; initial-response text is not yet available',
    'semantic_validity_human_verified': False,
    'smoke_and_pilot_exclusion': 'Reconstructed original deterministic split; excluded all 6 smoke and 60 pilot items as well as 300 main items.',
    'dataset_sha256': hashlib.sha256(pq.read_bytes()).hexdigest(),
    'archive_sha256': archive_hash, 'extraction_policy': POLICY,
    'original_exclusion_counts': original.groupby(['initial_cohort','reason']).size().rename('count').reset_index().to_dict(orient='records'),
    'by_subject': counts.to_dict(orient='records'),
}
(OUT / 'availability_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='extraction_policy'},ensure_ascii=False,indent=2),flush=True)
