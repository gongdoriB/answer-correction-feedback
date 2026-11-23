"""Experiment B v3: max-five-unit contiguous excerpts, historical R1, unchanged S0/S1."""
import copy
import hashlib
import random
from pathlib import Path
from collections import Counter
import protocol as original
from protocol import atomic_json, digest, int_seed, lock_json, read_json
from ablation_protocol import load_source, validate_source, GENERATION_KEYS, require
from extract_evidence import normalize_explanation, POLICY, LABEL, TEX

VERSION = '3.0.0-8b-evidence-B-max5'
MODEL = 'qwen3_8b'
DEFENSE_NAMES = ['S0', 'S1']
NEW_DEFENSES = DEFENSE_NAMES
DEFENSES = {s: original.DEFENSES[s] for s in DEFENSE_NAMES}
DEFENSE_LABELS = {'S0': '방어 없음', 'S1': '기존 전체 방어'}
MANIFEST_HASH = '254464a97dfe4051a3a91f663f6e5530234ac2463c3b30bd0999869d3bb3b46e'
CODE_FILES = ['protocol.py','engine.py','analysis.py','runtime_check.py','progress_ui.py',
              'ablation_protocol.py','extract_evidence.py','prepare_evidence.py',
              'evidence_protocol.py','evidence_run.py','evidence_analysis.py',
              'normalize_full.py','legacy_excerpt.py','sentence_units.py']
ANALYSIS_PLAN = {
    'design': 'Exploratory follow-up on a subset of the previously examined 300 items; not held-out confirmation.',
    'primary_population': '77 valid initially wrong questions with max-five-unit mechanically selected contiguous public excerpts; same IDs in all four cells.',
    'primary_endpoint': 'Final correct / all follow-ups on initially wrong questions. Invalid or truncated outputs remain in denominator.',
    'primary_contrast': '(R2_S1 - R2_S0) - (R1_S1 - R1_S0)',
    'primary_interval': 'One predeclared primary contrast: paired subject-stratified question bootstrap, 20000 repetitions, two-sided 95%.',
    'unit': 'Question. Average three fixed phrasings within question before pairing/resampling. Phrasings are not independent samples.',
    'secondary': '76 initially correct questions: correct retention, valid wrong changes, invalid/truncated outputs; same four cells. Five retention contrasts and four correction contrasts are descriptive 95% intervals, not additional confirmatory success tests.',
    'quality': 'Final answer agreement of public GPT-4o outputs is checked; reasoning validity/sufficiency is not human verified.',
    'controls': 'Original actual first responses and R1 S0/S1 reused on exact eligible items. New R2 only. Historical vs new schedule is not fully randomized together.',
    'selection': 'User-fixed cap of five prose/list units and five legacy sentence/math units, at most 180 whitespace words. Use 155 source excerpts passing surface filters before new inference; no follow-up-outcome selection. Two initial invalids appear only in all-item format metrics.',
    'interpretation': 'Positive interaction supports attenuation of the observed defense-associated correction decrease on this subset, conditional on R1 decrease. It does not prove cost elimination, valid-evidence discrimination, or low real-world cost.',
    'limits': ['Only correct-label-plus-excerpt input is added, without wrong-evidence or length-matched controls.',
               'Input length and rationale content change together; exact mechanism and valid vs invalid evidence discrimination are unidentified.',
               'One fixed contiguous excerpt per item. Missing reasoning may explain failures; no semantic completeness certification. No causal length comparison or pooling across R2 excerpt versions.',
               'Retention concerns true-answer suggestions, not resistance to authority-based wrong-answer suggestions; retaining the same letter does not identify the underlying decision mechanism.',
               'Historical R1 correct retention is 228/228 in both arms on these 76 items: secondary retention checks new degradation, with no upward room from this baseline.',
               'No RAG, no new defense, no new training, no additional judge model or human rationale-writing workload.'],
}


def load_manifest():
    folder = Path(__file__).parent/'source'
    m = read_json(folder/'evidence_manifest.json')
    require(m['manifest_hash'] == MANIFEST_HASH == digest({k:v for k,v in m.items() if k != 'manifest_hash'}), '근거 목록 해시 불일치')
    matched = read_json(folder/'official_matched.json')
    require(digest(matched) == m['matched_source_hash'], '공식 공개 풀이 원문 불일치')
    require(m['policy'] == POLICY, '추출 규칙 변경')
    lookup = {r['question_id']:r for r in matched}
    for r in m['excerpts']:
        raw = lookup[r['question_id']]['official_record']
        clean,reason,edits = normalize_explanation(raw['response'],raw['options'],list('ABCDEFGHIJ')[:len(raw['options'])])
        require(raw['pred'] == raw['answer'] == r['gold'] and reason is None, '정답 일치/정규화 실패')
        require(clean == r['text'] and edits == r['normalization_edits'], '풀이 내용/변경 내역 불일치')
        require(hashlib.sha256(raw['response'].encode()).hexdigest() == r['response_sha256'], '풀이 원문 해시 불일치')
        require(not LABEL.search(TEX.sub(' ',r['text'])), '풀이에 선택지 라벨 반복')
    return m


def default_config(source):
    cfg = copy.deepcopy(source['study']['config'])
    m = load_manifest()
    cfg.update(version=VERSION, phase='evidence_B_max5_v3', seed=20261030, conditions=['R1','R2'],
               defenses=DEFENSE_NAMES[:], generate_conditions=['R2'], variants=[0,1,2],
               models={MODEL:cfg['models'][MODEL]}, baseline_repetitions=0,
               new_followups=m['new_followups'], reused_followups=m['reuse_R1_followups'],
               manifest_hash=MANIFEST_HASH, selected_cap=m['selected_cap'], exploratory=True,
               primary_family_size=1, primary_bootstrap_repetitions=20000,
               analysis_plan_hash=digest(ANALYSIS_PLAN))
    cfg['prompt_hash'] = digest([original.BASE_SYSTEM, DEFENSES, original.STEMS, original.SUFFIX, 'Explanation:\n'])
    return cfg


def selected_controls(source):
    eligible = {r['question_id'] for r in load_manifest()['excerpts']}
    rows = [r for r in source['controls'] if r['condition']=='R1' and r['question_id'] in eligible]
    require(len(rows) == 6*len(eligible), '기존 R1 대응 자료 누락')
    return rows


def make_jobs(source, cfg):
    require(cfg == default_config(source), '고정 실험 설정을 변경하지 마세요.')
    require(all(cfg[k] == source['study']['config'][k] for k in GENERATION_KEYS), '원본 생성 설정 변경')
    excerpts = {r['question_id']: r for r in load_manifest()['excerpts']}
    jobs=[]
    for old in selected_controls(source):
        r = excerpts[old['question_id']]
        messages = copy.deepcopy(old['messages'])
        claim, suffix = messages[-1]['content'].split('\n\n',1)
        require(suffix == original.SUFFIX, 'R1 출력 지시 불일치')
        messages[-1]['content'] = claim + '\n\nExplanation:\n' + r['text'] + '\n\n' + suffix
        jobs.append({'task_id': f"BMAX5:{old['question_id']}:{old['variant']}:R2:{old['defense']}",
                     'question_id':old['question_id'], 'phase':'followup', 'condition':'R2',
                     'defense':old['defense'], 'variant':old['variant'], 'baseline_hash':old['baseline_hash'],
                     'rationale_hash':digest(r['text']), 'historical_R1_task_id':old['task_id'], 'messages':messages})
    random.Random(int_seed(cfg['seed'],'evidence_B_max5_v3_schedule')).shuffle(jobs)
    require(len(jobs)==len({j['task_id'] for j in jobs})==cfg['new_followups'], '작업 누락/중복')
    return jobs


def fingerprint():
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in CODE_FILES}


def freeze(root,source,cfg):
    validate_source(source)
    m = load_manifest()
    jobs = make_jobs(source,cfg)
    controls = selected_controls(source)
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    frozen = {'version':VERSION,'config':cfg,'source_hash':source['source_hash'],
              'manifest_hash':m['manifest_hash'],'jobs_hash':digest(jobs),
              'n_followups':len(jobs),'n_reused_followups':len(controls),
              'eligible_cohorts':dict(Counter(r['initial_cohort'] for r in m['excerpts'])),
              'code_sha256':fingerprint(),'analysis_plan':ANALYSIS_PLAN,'defense_texts':DEFENSES,
              'baseline_regeneration':False,'R1_regeneration':False,'new_conditions':['R2']}
    frozen['study_hash']=digest(frozen)
    for name,value in [('study_lock.json',frozen),('source_snapshot.json',source),('evidence_manifest.json',m),
                       ('analysis_plan.json',ANALYSIS_PLAN),('reused_R1.json',controls),
                       ('schedule.json',{'jobs_hash':digest(jobs),'batch_size':cfg['batch_size'],'task_ids':[j['task_id'] for j in jobs]})]:
        lock_json(root/name,value)
    for n in CODE_FILES + ['source/official_matched.json','source/selected_excerpts.csv','source/availability.csv','source/eligibility_audit.csv','source/token_lengths.csv','source/normalization_audit.json','README_KO.txt']:
        data=(Path(__file__).parent/n).read_bytes(); dest=root/'reproduction'/n
        dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists() or dest.read_bytes()==data, f'저장 자료 변경: {n}')
        if not dest.exists(): dest.write_bytes(data)
    return frozen,jobs


def verify_frozen(root,source,cfg,jobs):
    frozen,rebuilt=freeze(root,source,cfg)
    require(digest(jobs)==digest(rebuilt), '작업 내용/순서 변경')
    return frozen
