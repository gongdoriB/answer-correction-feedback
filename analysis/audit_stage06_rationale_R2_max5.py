"""Verify the received max5 B run using trusted local frozen code; no inference.

Preserve all received files. Recompute the planned question-paired contrasts,
without new endpoints, relaxed parsing, subgroup selection, or p-value searches.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
CODE = PROJECT/'output/jupyter-notebook/sycophancy_qwen_8b_evidenceB_max5_support'
RUN = OUT/'received/sycophancy_qwen3_8b_evidenceB_max5_930_a100_b16_v3'
ZIP = Path('data/results_zips/sycophancy_qwen3_8b_evidenceB_max5_930_a100_b16_v3_results.zip')  # REDACTED: was an absolute local Downloads path
sys.path.insert(0,str(CODE))
from evidence_protocol import load_source, load_manifest, default_config, make_jobs, freeze, selected_controls, fingerprint
from evidence_run import scan_saved, validate_source_environment, generation_environment
from evidence_analysis import score_records, effect_table, comparison_table, retention_table
from protocol import read_json, digest, parse_answer, atomic_json
from length_analysis import item_lengths, PLAN as LENGTH_PLAN

source=load_source(CODE/'source')
manifest=load_manifest()
cfg=default_config(source)
frozen,jobs=freeze(OUT/'verified_design',source,cfg)
assert frozen == read_json(RUN/'study_lock.json')
assert read_json(RUN/'source_snapshot.json') == source
assert read_json(RUN/'evidence_manifest.json') == manifest
assert read_json(RUN/'analysis_plan.json') == frozen['analysis_plan']
assert read_json(RUN/'reused_R1.json') == selected_controls(source)
for name,sha in fingerprint().items():
    assert hashlib.sha256((RUN/'reproduction'/name).read_bytes()).hexdigest() == sha,name
env=read_json(RUN/'environment_lock.json')
assert not env.get('synthetic_test',False)
notes=validate_source_environment(env,source)
assert env['chat_template_hash'] == source['run']['environment']['chat_template_hash']
assert env['generation_config'] == source['run']['environment']['generation_config']
run_lock=read_json(RUN/'run_lock.json')
assert run_lock['generation_environment']==generation_environment(env)
assert run_lock['study_hash']==frozen['study_hash'] and run_lock['source_hash']==source['source_hash']
preflight=read_json(RUN/'token_preflight.json')
assert preflight == {'count':930,'max_input_tokens_observed':1258,'jobs_hash':digest(jobs)}
saved=scan_saved(RUN,jobs,cfg)
assert len(saved)==59 and len(saved[58]['records'])==2
for v in saved.values():assert v['environment_hash']==digest(env)
records=[r for i in sorted(saved) for r in saved[i]['records']]
assert len(records)==len(jobs)==930
assert len({r['task_id'] for r in records})==930
assert {r['condition'] for r in records}=={'R2'}
assert {r['defense'] for r in records}=={'S0','S1'}
status=read_json(RUN/'status.json')
assert status['complete'] and status['saved_responses']==930 and status['saved_batches']==59
assert status['new_baselines_generated']==status['new_R1_generated']==0
assert status['original_followups_reused']==930

# Check actual input lengths/rendered prompt hashes against the packaged preflight inventory.
token_rows=pd.read_csv(CODE/'source/token_lengths.csv').set_index(['question_id','defense','variant'])
for r in records:
    t=token_rows.loc[(r['question_id'],r['defense'],r['variant'])]
    assert r['input_tokens']==t.input_tokens and r['rendered_prompt_hash']==t.rendered_prompt_hash

frame=score_records(source,selected_controls(source)+records)
received=pd.read_csv(RUN/'reports/scored_responses.csv')
assert len(frame)==len(received)==1860
assert set(frame.columns)==set(received.columns)
assert not frame.duplicated(['question_id','condition','defense','variant']).any()
frame=frame.sort_values('task_id').reset_index(drop=True)
received=received.sort_values('task_id').reset_index(drop=True)[frame.columns]
for col in frame:
    if pd.api.types.is_numeric_dtype(frame[col]):
        assert np.allclose(frame[col],received[col],equal_nan=True),col
    else:
        norm=lambda s: [None if pd.isna(v) else v for v in s]
        assert norm(frame[col])==norm(received[col]),col
assert frame.query('baseline_valid and not baseline_correct').question_id.nunique()==77
assert frame.query('baseline_valid and baseline_correct').question_id.nunique()==76
assert frame.query('not baseline_valid').question_id.nunique()==2
print('Raw records, environment, preflight, matched controls and all 1860 scored rows verified.',flush=True)

def compare_table(actual, filename, keys):
    # CSV readers infer numeric-looking category labels as integers. Preserve
    # the frozen report's categorical sentence-bin representation for comparison.
    expected=pd.read_csv(RUN/'reports'/filename,dtype={'sentence_bin':str})
    a=actual.sort_values(keys).reset_index(drop=True)
    e=expected.sort_values(keys).reset_index(drop=True)[a.columns]
    pd.testing.assert_frame_equal(a,e,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
    actual.to_csv(OUT/('verified_'+filename),index=False,encoding='utf-8-sig')

table=comparison_table(frame);retained=retention_table(frame)
compare_table(table,'comparison_table.csv',['condition','defense'])
compare_table(retained,'retention_table.csv',['condition','defense'])
effects,details=effect_table(frame,cfg)
compare_table(effects,'evidence_effects.csv',['contrast'])
compare_table(details,'item_contrasts.csv',['question_id','contrast'])
lengths=item_lengths(manifest)
compare_table(lengths,'explanation_lengths.csv',['question_id'])
length_lock=read_json(RUN/'length_reporting_v1/analysis_lock.json')
assert length_lock['plan']==LENGTH_PLAN
assert length_lock['manifest_hash']==manifest['manifest_hash']
assert length_lock['code_sha256']==hashlib.sha256((CODE/'length_analysis.py').read_bytes()).hexdigest()
assert length_lock['length_rows_hash']==digest(lengths.to_dict(orient='records'))
lf=frame.merge(lengths[['question_id','initial_cohort','sentence_bin']],on='question_id',validate='many_to_one')
length_report=pd.read_csv(RUN/'reports/length_group_results.csv',dtype={'sentence_bin':str})
for r in length_report.itertuples():
    f=lf[(lf.initial_cohort==r.initial_cohort)&(lf.sentence_bin==r.sentence_bin)&(lf.condition==r.condition)&(lf.defense==r.defense)]
    assert f.question_id.nunique()==r.n_items and len(f)==r.n_responses
    assert int(f.final_correct.sum())==r.correct_count and int(f.invalid.sum())==r.invalid_count
    assert abs(100*f.final_correct.mean()-r.correct_percent)<1e-10

frame['cohort']=np.where(~frame.baseline_valid,'initial_invalid',np.where(frame.baseline_correct,'initial_correct','initial_wrong'))
fmt=frame.groupby(['cohort','condition','defense']).agg(n_items=('question_id','nunique'),responses=('task_id','size'),invalid=('invalid','sum'),truncated=('truncated','sum')).reset_index()
fmt.to_csv(OUT/'format_audit.csv',index=False,encoding='utf-8-sig')
items={i['question_id']:i for i in source['study']['items']}
invalid=[]
for r in records:
    _,parse_status=parse_answer(r['text'],items[r['question_id']]['labels'])
    if parse_status!='valid' or r['truncated']:
        q=frame.loc[frame.task_id==r['task_id']].iloc[0]
        invalid.append({'task_id':r['task_id'],'question_id':r['question_id'],'cohort':q.cohort,
                        'defense':r['defense'],'variant':r['variant'],'parse_status':parse_status,
                        'truncated':r['truncated'],'output_tokens':r['output_tokens'],'gold':q.gold,'raw_text':r['text']})
pd.DataFrame(invalid).to_csv(OUT/'invalid_R2_outputs.csv',index=False,encoding='utf-8-sig')
dist=details.groupby(['contrast','difference_pp']).size().rename('n_items').reset_index()
dist.to_csv(OUT/'paired_item_effect_distribution.csv',index=False,encoding='utf-8-sig')

effect=effects.set_index('contrast')
report={
    'status':'verified','original_zip':str(ZIP),'original_zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),
    'new_R2_responses':930,'saved_batches':59,'reused_R1_responses':930,'all_scored_rows':1860,
    'initial_wrong_items':77,'initial_correct_items':76,'initial_invalid_items':2,
    'synthetic_data':False,'environment_matches_original':True,'other_environment_differences':notes,
    'study_hash':frozen['study_hash'],'manifest_hash':manifest['manifest_hash'],
    'received_codes_match_trusted_local':True,'all_saved_prompts_and_seeds_match':True,
    'all_actual_input_token_hashes_match':True,'all_scored_rows_match':True,'all_10_planned_effects_match':True,
    'all_40_length_rate_cells_match':True,'all_155_length_inventory_rows_match':True,
    'primary_effect':effect.loc['interaction'].to_dict(),'R2_defense_effect':effect.loc['R2_defense'].to_dict(),
    'raw_R2_invalid_total':len(invalid),'primary_R2_S1_invalid':int(frame.query("cohort=='initial_wrong' and condition=='R2' and defense=='S1'").invalid.sum()),
    'primary_R2_S1_truncated':int(frame.query("cohort=='initial_wrong' and condition=='R2' and defense=='S1'").truncated.sum()),
    'original_scoring_preserved':True,'new_models_or_followups_generated':0,'new_hypothesis_tests':0,
    'verification_environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
    'interpretation':'Observed defense-associated correction decrease attenuated with public max5 excerpts. This is not an equivalence test or proof of no residual cost, evidence-validity discrimination, source mechanism, or real-world performance.',
}
atomic_json(OUT/'audit.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
