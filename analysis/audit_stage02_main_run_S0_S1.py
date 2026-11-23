"""Reproduce the uploaded main experiment without new model inference."""
from pathlib import Path
import sys, json, zipfile, hashlib
from collections import Counter
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / 'jupyter-notebook' / 'sycophancy_qwen_core_support'
sys.path.insert(0, str(SOURCE))
from protocol import digest, read_json, baseline_jobs, followup_jobs, int_seed, default_config, audit_jobs
from engine import code_fingerprint, read_records
from analysis import collect, summarize, contrasts

ARCHIVE = Path('data/results_zips/sycophancy_qwen_core_v3_formatfix_a100_b16-20261026T080525Z-1-001.zip')  # REDACTED: was an absolute local Downloads path
DEST = OUT / 'received'
DEST.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    for info in z.infolist():
        dest = (DEST / info.filename).resolve()
        assert dest.is_relative_to(DEST.resolve()), 'Unsafe archive path'
        z.extract(info, DEST)
ROOT = DEST / 'sycophancy_qwen_core_v3_formatfix_a100_b16'
PHASE = ROOT / 'main'
REPORTS = PHASE / 'reports' / 'a100_b16'
frozen = read_json(PHASE / 'study_lock.json')
assert digest({k: v for k,v in frozen.items() if k != 'study_hash'}) == frozen['study_hash']
assert frozen['code'] == code_fingerprint()
config = default_config('main')
config['batch_size'] = 16
assert config == frozen['config']
pool = read_json(ROOT / 'item_pool.json')['pools']
assert pool['main'] == frozen['items']
ids = {p: {x['question_id'] for x in rows} for p,rows in pool.items()}
for a in ids:
    for b in ids:
        if a != b: assert not ids[a] & ids[b]
assert len(ids['main']) == 300
assert set(Counter(x['subject'] for x in frozen['items']).values()) == {50}
audit = {'archive': str(ARCHIVE), 'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
         'study_hash': frozen['study_hash'], 'checks': {}, 'models': {}}
environments = []
for model in config['models']:
    directory = PHASE / 'runs' / 'a100_b16' / model
    lock = read_json(directory / 'run_lock.json')
    assert lock['model'] == config['models'][model]
    assert lock['study_hash'] == frozen['study_hash']
    environments.append({k:v for k,v in lock['environment'].items() if k not in {'chat_template_hash','generation_config'}})
    baseline = read_records(directory, 'baseline')
    jobs_by_stage = {'baseline': baseline_jobs(frozen['items']),
                     'followup': followup_jobs(frozen['items'], {r['question_id']:r for r in baseline}, config)}
    audit_jobs(frozen['items'], jobs_by_stage['followup'], config)
    details = {}
    for stage,jobs in jobs_by_stage.items():
        schedule = read_json(directory / stage / 'schedule.json')
        assert schedule == {'jobs_hash': digest(jobs), 'task_ids':[j['task_id'] for j in jobs], 'batch_size':16}
        files = sorted((directory / stage).glob('batch_*.json'))
        assert len(files) == (len(jobs)+15)//16
        seconds, truncated = 0., 0
        for index,path in enumerate(files):
            saved = read_json(path)
            batch = jobs[index*16:(index+1)*16]
            assert saved['batch_hash'] == digest(batch)
            assert saved['records_hash'] == digest(saved['records'])
            assert len(saved['records']) == len(batch)
            seed = int_seed(config['seed'], stage, index, [j['task_id'] for j in batch])
            for job,record in zip(batch, saved['records']):
                assert all(record[k] == v for k,v in job.items())
                assert record['batch_seed'] == seed and record['batch_index'] == index
                truncated += bool(record['truncated'])
            seconds += saved['records'][0]['seconds_per_batch']
        details[stage] = {'responses':len(jobs), 'batches':len(files), 'truncated':truncated, 'generation_seconds': seconds}
    audit['models'][model] = details
assert environments[0] == environments[1]
audit['environment'] = environments[0]
print('Raw checkpoint, prompt, seed, code and environment checks passed.', flush=True)
frozen, frame, baseline, completed = collect(PHASE, 'a100_b16')
assert set(completed) == set(config['models'])
stored = pd.read_csv(REPORTS / 'scored_responses.csv')
keys = ['model','question_id','condition','defense','variant']
def normalized(df):
    return df.sort_values(keys).reset_index(drop=True).fillna('').astype(str)
pd.testing.assert_frame_equal(normalized(frame), normalized(stored), check_dtype=False)
for name, recalculated, sortkeys in [
    ('paired_effects.csv', contrasts(frame,config), ['model','contrast']),
    ('metrics.csv', summarize(frame,config), ['model','condition','defense','stratum','metric'])]:
    saved = pd.read_csv(REPORTS / name).sort_values(sortkeys).reset_index(drop=True)
    recalculated = recalculated.sort_values(sortkeys).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, recalculated, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    print(name, 'fully reproduced:',len(saved),'rows',flush=True)
for model in completed:
    b, f = baseline[baseline.model == model], frame[frame.model == model]
    audit['models'][model].update({'baseline_correct':int(b.correct.sum()), 'baseline_invalid':int((~b.valid).sum()),
                                 'followup_invalid':int(f.invalid.sum()), 'followup_invalid_rate':float(f.invalid.mean())})
    print(model, json.dumps(audit['models'][model],ensure_ascii=False),flush=True)
    print('Initial-correct target agreement:', f[f.baseline_correct & f.condition.isin(['C2','C3','C4','C5'])].groupby(['condition','defense']).target_agreement.mean().to_dict())
audit['checks'] = {name:True for name in ['archive_crc','study_hash','code_fingerprint','fixed_config','disjoint_sampling_pools',
                   'balanced_subjects','all_checkpoints_hashes','full_expected_prompts','fixed_batch_seeds','shared_environment',
                   'all_response_scores_reproduced','all_14_effects_and_intervals_reproduced','all_summary_metrics_and_intervals_reproduced']}
(OUT / 'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS:', OUT / 'audit.json',flush=True)
