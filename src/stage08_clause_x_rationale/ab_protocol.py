"""Complete the missing R2 SC/SI/SR cells using the exact A clauses and B excerpts."""
import copy
import hashlib
import random
from pathlib import Path
from collections import Counter
import protocol as original
from protocol import digest, int_seed, lock_json, read_json, parse_answer
from ablation_protocol import load_source, validate_source, GENERATION_KEYS, require, DEFENSES, DEFENSE_NAMES, NEW_DEFENSES

VERSION = '1.0.0-8b-AB693'
MODEL = 'qwen3_8b'
DEFENSE_LABELS = {'S0':'방어 없음','SC':'공통 문장만','SI':'공통 + 억제 절','SR':'공통 + 수정 절','S1':'공통 + 억제·수정 절'}
MANIFEST_HASH = '254464a97dfe4051a3a91f663f6e5530234ac2463c3b30bd0999869d3bb3b46e'
HISTORY_HASH = '8780c4c9864a6f7e3f8fac14b84406ec83787950fea4e54171c7fe9e31dd789c'
CODE_FILES = ['protocol.py','engine.py','analysis.py','runtime_check.py','progress_ui.py','ablation_protocol.py','ab_protocol.py','ab_run.py','ab_analysis.py']
ANALYSIS_PLAN = {
 'design':'Exploratory completion of a 2 (R1/R2) by 2 (inhibition absent/present) by 2 (correction absent/present) design with S0 reference. Same 77 initially wrong questions; historical and new cells are not interleaved.',
 'population':'All 77 valid initially wrong items eligible in the previously frozen B max5 manifest. No selection using new follow-up outcomes.',
 'new_cells':['R2_SC','R2_SI','R2_SR'],
 'reused_cells':['R1_S0','R1_SC','R1_SI','R1_SR','R1_S1','R2_S0','R2_S1'],
 'primary_endpoint':'Correct follow-up / all follow-ups on initially wrong items. Invalid or truncated outputs count as failures and remain in the denominator.',
 'primary_contrasts':[
  {'name':'inhibition_cost_attenuation','terms':[['R2','SI',1],['R2','SC',-1],['R1','SI',-1],['R1','SC',1]],'meaning':'Change in inhibition-only effect when the fixed excerpt is added; positive means attenuation if the R1 inhibition effect is negative.'},
  {'name':'correction_clause_with_evidence','terms':[['R2','S1',1],['R2','SI',-1]],'meaning':'Added correction clause under inhibition and excerpt feedback; positive means higher correction.'}],
 'inference':'Average the three fixed variants within each question, pair by question, then subject-stratified question bootstrap, 20000 repetitions. Two primary contrasts use 97.5% intervals (Bonferroni family alpha .05). Remaining contrasts are descriptive 95% intervals. This is not independent confirmatory validation.',
 'secondary':'Initial wrong retained, changed to another wrong answer, format/truncation failures, tokens; full factorial clause contrasts are descriptive.',
 'fixed':'Same actual first answer, question, options, published gold, A system clauses, B rationale and R1/R2 feedback wording. Original batch16, NF4/BF16, non-thinking, 192 output tokens. New fixed seed/schedule; no historical regeneration.',
 'limits':['77 independent question units, not 693 independent samples; no power guarantee.',
  'Only correct-answer suggestions. This addition does not measure resistance to wrong-authority pressure or correct-answer retention.',
  'B excerpts come from the dataset authors\' released GPT-4o answer archive, not human-verified ground-truth rationales. Same max-five-unit/180-word surface-filtered excerpts are reused without new manual review.',
  'Rationale content and input length change together. No content-versus-length causal identification.',
  'Historical A/B/original and new sessions differ in schedule and seed; matched settings do not remove all session confounding.',
  'An interval including zero does not establish equivalence or no cost. No new prompt superiority, deployed-agent safety, or evidence-quality discrimination claim.']}


def load_manifest():
    folder=Path(__file__).parent/'source';m=read_json(folder/'evidence_manifest.json')
    require(m['manifest_hash']==MANIFEST_HASH==digest({k:v for k,v in m.items() if k!='manifest_hash'}),'B 근거 목록 변경')
    matched=read_json(folder/'official_matched.json')
    require(digest(matched)==m['matched_source_hash'],'B 풀이 원문 변경')
    return m


def load_history(source):
    h=read_json(Path(__file__).parent/'source/history.json')
    require(h['history_hash']==HISTORY_HASH==digest({k:v for k,v in h.items() if k!='history_hash'}),'기존 A/B 응답 해시 변경')
    m=load_manifest();ex={r['question_id']:r for r in m['excerpts'] if r['initial_cohort']=='wrong'}
    require(len(ex)==77 and sorted(ex)==h['question_ids'],'공통 77문항 변경')
    items={i['question_id']:i for i in source['study']['items']};bases={r['question_id']:r for r in source['baselines']}
    counts=Counter();seen=set()
    for r in h['records']:
        q=r['question_id'];d=r['defense'];c=r['condition'];v=r['variant'];item=items[q];base=bases[q]
        b,status=parse_answer(base['text'],item['labels'])
        require(q in ex and status=='valid' and not base['truncated'] and b!=item['gold'],'초기 오답 모집단 불일치')
        require(d in DEFENSE_NAMES and c in ['R1','R2'] and v in [0,1,2],'알 수 없는 기존 조건')
        key=(q,c,d,v);require(key not in seen,'기존 응답 중복');seen.add(key);counts[(c,d)]+=1
        require(r['baseline_hash']==digest(base['text']),'최초 답변 해시 변경')
        msg=[{'role':'system','content':original.BASE_SYSTEM+ ('\n\n'+DEFENSES[d] if DEFENSES[d] else '')},
             {'role':'user','content':original.question_text(item)}, {'role':'assistant','content':base['text']},
             {'role':'user','content':original.intervention(item,'R1',v)}]
        if c=='R2':
            claim,suffix=msg[-1]['content'].split('\n\n',1)
            msg[-1]['content']=claim+'\n\nExplanation:\n'+ex[q]['text']+'\n\n'+suffix
            require(r['rationale_hash']==digest(ex[q]['text']),'기존 B 풀이 변경')
        require(r['messages']==msg,'기존 메시지/최초 답변 변경')
    expected={('R1',d):231 for d in DEFENSE_NAMES};expected.update({('R2',d):231 for d in ['S0','S1']})
    require(dict(counts)==expected and len(h['records'])==1617,'기존 대응 셀 누락')
    for k in ['A','B']:
        env=h['archive_provenance'][k]['locks']['environment_lock.json']
        for name in ['gpu','compute_capability','cuda','packages','chat_template_hash','generation_config']:
            require(env[name]==source['run']['environment'][name],f'기존 {k} 생성 환경 차이: {name}')
    return h


def selected_controls(source):
    return load_history(source)['records']


def default_config(source):
    cfg=copy.deepcopy(source['study']['config'])
    cfg.update(version=VERSION,phase='ab_evidence_693',seed=20261031,conditions=['R1','R2'],
      defenses=DEFENSE_NAMES[:],generate_conditions=['R2'],generate_defenses=NEW_DEFENSES[:],variants=[0,1,2],
      models={MODEL:cfg['models'][MODEL]},baseline_repetitions=0,new_followups=693,reused_followups=1617,
      n_analysis_items=77,manifest_hash=MANIFEST_HASH,history_hash=HISTORY_HASH,exploratory=True,
      primary_family_size=2,primary_bootstrap_repetitions=20000,analysis_plan_hash=digest(ANALYSIS_PLAN))
    cfg['prompt_hash']=digest([original.BASE_SYSTEM,DEFENSES,original.STEMS,original.SUFFIX,'Explanation:\n'])
    return cfg


def make_jobs(source,cfg):
    require(cfg==default_config(source),'고정 설정을 변경하지 마세요.')
    require(all(cfg[k]==source['study']['config'][k] for k in GENERATION_KEYS),'기존 생성 설정 변경')
    history=load_history(source);jobs=[]
    old_R1={(r['question_id'],r['defense'],r['variant']):r for r in history['records'] if r['condition']=='R1'}
    for r in history['records']:
        if r['condition']!='R2' or r['defense']!='S1':continue
        for d in NEW_DEFENSES:
            messages=copy.deepcopy(r['messages']);messages[0]['content']=original.BASE_SYSTEM+'\n\n'+DEFENSES[d]
            old=old_R1[(r['question_id'],d,r['variant'])]
            require(messages[:3]==old['messages'][:3],'A와 최초 대화 불일치')
            jobs.append({'task_id':f"AB693:{r['question_id']}:{r['variant']}:R2:{d}",
              'question_id':r['question_id'],'phase':'followup','condition':'R2','defense':d,'variant':r['variant'],
              'baseline_hash':r['baseline_hash'],'rationale_hash':r['rationale_hash'],
              'historical_R1_task_id':old['task_id'],'historical_B_task_id':r['task_id'],'messages':messages})
    random.Random(int_seed(cfg['seed'],'AB693_schedule_v1')).shuffle(jobs)
    require(len(jobs)==len({j['task_id'] for j in jobs})==693,'신규 작업 누락/중복')
    return jobs


def fingerprint():
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in CODE_FILES}


def freeze(root,source,cfg):
    validate_source(source);history=load_history(source);m=load_manifest();jobs=make_jobs(source,cfg)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    frozen={'version':VERSION,'config':cfg,'source_hash':source['source_hash'],'manifest_hash':MANIFEST_HASH,
       'history_hash':HISTORY_HASH,'jobs_hash':digest(jobs),'n_followups':693,'n_reused_followups':1617,
       'question_ids':history['question_ids'],'code_sha256':fingerprint(),'analysis_plan':ANALYSIS_PLAN,
       'defense_texts':DEFENSES,'baseline_regeneration':False,'R1_regeneration':False,'new_conditions':['R2'],'new_defenses':NEW_DEFENSES}
    frozen['study_hash']=digest(frozen)
    for name,value in [('study_lock.json',frozen),('source_snapshot.json',source),('evidence_manifest.json',m),
       ('history.json',history),('analysis_plan.json',ANALYSIS_PLAN),('new_jobs.json',jobs),
       ('schedule.json',{'jobs_hash':digest(jobs),'batch_size':16,'task_ids':[j['task_id'] for j in jobs]})]:
        lock_json(root/name,value)
    source_files=[str(p.relative_to(Path(__file__).parent)) for p in (Path(__file__).parent/'source').glob('*') if p.is_file()]
    for n in CODE_FILES+source_files+['README_KO.txt']:
        data=(Path(__file__).parent/n).read_bytes();dest=root/'reproduction'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists() or dest.read_bytes()==data,f'저장 코드/자료 변경: {n}')
        if not dest.exists():dest.write_bytes(data)
    return frozen,jobs


def verify_frozen(root,source,cfg,jobs):
    frozen,rebuilt=freeze(root,source,cfg)
    require(digest(jobs)==digest(rebuilt),'작업 내용/순서 변경')
    return frozen
