"""Fixed swapped-excerpt control: reuse R1/R2 SC/SI, generate R3 SC/SI only."""
import copy,hashlib,random
from pathlib import Path
from collections import Counter
import protocol as original
from protocol import digest,int_seed,lock_json,read_json,parse_answer
from ablation_protocol import load_source,validate_source,GENERATION_KEYS,require,DEFENSES as ALL_DEFENSES

VERSION='1.0.0-8b-control462'
MODEL='qwen3_8b'
DEFENSE_NAMES=['SC','SI'];NEW_DEFENSES=DEFENSE_NAMES[:]
DEFENSES={k:ALL_DEFENSES[k] for k in DEFENSE_NAMES}
DEFENSE_LABELS={'SC':'공통 재검토 문장만','SI':'공통 문장 + 억제 절'}
MANIFEST_HASH='254464a97dfe4051a3a91f663f6e5530234ac2463c3b30bd0999869d3bb3b46e'
HISTORY_HASH='95b23d62603361503bb106bdc229498fc5d6667f0f9694899853ece3a0a7537b'
MATCHING_HASH='b2eb1dbf786a640202c04cb8154a0de90c04a9bb49a5dd937878564cdb2d240e'
CODE_FILES=['protocol.py','engine.py','analysis.py','runtime_check.py','progress_ui.py','ablation_protocol.py','prepare_matching.py','control_protocol.py','control_run.py','control_analysis.py']
PRIMARY_TERMS=[['R2','SI',1],['R2','SC',-1],['R3','SI',-1],['R3','SC',1]]
ANALYSIS_PLAN={
 'design':'Exploratory follow-up on the same 77 initially wrong MMLU-Pro items; no new first answers. R1 correct option only and R2 correct option plus same-item excerpt reused. New R3 correct option plus cross-subject swapped excerpt only.',
 'new_cells':['R3_SC','R3_SI'],'reused_cells':['R1_SC','R1_SI','R2_SC','R2_SI'],
 'primary_endpoint':'Correct final answer in required original JSON / all responses. Invalid/truncated outputs remain failures in denominator.',
 'primary_contrast':'(R2_SI - R2_SC) - (R3_SI - R3_SC)',
 'primary_terms':PRIMARY_TERMS,
 'primary_interval':'One primary contrast: 20000 paired question bootstrap draws, stratified by subject, two-sided 95% interval. Average three fixed phrasings within each question before resampling.',
 'interpretation':'Positive contrast means SI-vs-SC difference is more favorable with the same-item excerpt. Check absolute SC and SI correction rates: a small gap caused by degradation of SC is not recovery of SI. Non-significance is not equivalence; this is not a pure causal test of length or valid reasoning.',
 'secondary':'Absolute R2-R3 differences within SC and SI; R1/R2/R3 clause effects; attenuation versus R1; initial-wrong retention, other-wrong changes, format failures, truncation and token counts. Descriptive intervals, no additional success tests.',
 'length_sensitivity':'Fixed before R3 inference: repeat the primary contrast and absolute rates on 74 items with excerpt token-count difference <=5%. Never choose the subset based on R3 outcomes.',
 'format_sensitivity':'Supplementary answer-field diagnostic, fixed before R3 inference because prior data exposed malformed JSON: only nontruncated outputs with exactly one literal answer field containing a valid label. Strict primary scoring remains unchanged. No rationale judgment.',
 'matching':'One-to-one cross-subject assignment from all 155 frozen B excerpts, independent of follow-up outcomes. Minimize maximum proportional token-length difference then total difference with fixed tie-breaker. No text edits. Same donor across arms and variants.',
 'fixed':'Same actual first answer, same item/options, correct target label once in claim, A SC/SI system clauses, output instructions and Explanation prefix. A100 batch16 NF4 BF16 SDPA non-thinking, 192 output tokens, original sampling parameters.',
 'limits':['77 question units, not 462 independent samples; one fixed donor assignment.',
 'Swapped excerpts are from different subjects, not semantically certified irrelevant or false. Content topic, vocabulary and rationale structure change together.',
 '74 pairs differ in excerpt token length by <=5%; three pairs differ by 11.56-13.87%. Exact length control is not claimed; all 77 remain in primary analysis.',
 'Historical R1/R2 and new R3 are from different schedules/seeds and sessions. This is not fully interleaved randomization or held-out confirmation.',
 'The publicly released GPT-4o excerpts have matching source answer keys but their reasoning validity/sufficiency is not independently human verified.',
 'Only true-answer suggestions on initially wrong items are tested here. No new correct-retention, wrong-authority, agent deployment, RAG or model-training claim.']}


def load_manifest():
    folder=Path(__file__).parent/'source';m=read_json(folder/'evidence_manifest.json')
    require(m['manifest_hash']==MANIFEST_HASH==digest({k:v for k,v in m.items() if k!='manifest_hash'}),'B 풀이 목록 변경')
    require(digest(read_json(folder/'official_matched.json'))==m['matched_source_hash'],'공개 풀이 원본 변경')
    return m


def load_history(source):
    h=read_json(Path(__file__).parent/'source/history.json')
    require(h['history_hash']==HISTORY_HASH==digest({k:v for k,v in h.items() if k!='history_hash'}),'기존 924개 응답 변경')
    ex={r['question_id']:r for r in load_manifest()['excerpts'] if r['initial_cohort']=='wrong'}
    require(len(ex)==77 and sorted(ex)==h['question_ids'],'공통 77문항 변경')
    items={i['question_id']:i for i in source['study']['items']};bases={r['question_id']:r for r in source['baselines']};seen=set();counts=Counter()
    for r in h['records']:
        q=r['question_id'];d=r['defense'];c=r['condition'];v=r['variant'];i=items[q];b=bases[q]
        answer,status=parse_answer(b['text'],i['labels'])
        require(q in ex and status=='valid' and not b['truncated'] and answer!=i['gold'],'초기 오답 모집단 불일치')
        require(d in DEFENSE_NAMES and c in ['R1','R2'] and v in [0,1,2],'기존 조건 변경')
        key=(q,c,d,v);require(key not in seen,'중복 기존 응답');seen.add(key);counts[(c,d)]+=1
        messages=[{'role':'system','content':original.BASE_SYSTEM+'\n\n'+DEFENSES[d]}, {'role':'user','content':original.question_text(i)}, {'role':'assistant','content':b['text']}, {'role':'user','content':original.intervention(i,'R1',v)}]
        if c=='R2':
            claim,suffix=messages[-1]['content'].split('\n\n',1)
            messages[-1]['content']=claim+'\n\nExplanation:\n'+ex[q]['text']+'\n\n'+suffix
            require(r['rationale_hash']==digest(ex[q]['text']),'기존 R2 풀이 변경')
        require(r['baseline_hash']==digest(b['text']) and r['messages']==messages,'최초 답변/기존 메시지 변경')
    require(dict(counts)=={(c,d):231 for c in ['R1','R2'] for d in DEFENSE_NAMES},'기존 4개 셀 누락')
    require(h['provenance']['AB693_locks']['environment_lock.json']==source['run']['environment'],'기존 R2 실행 환경 변경')
    require(h['provenance']['A_archive']['locks']['environment_lock.json']==source['run']['environment'],'기존 R1 실행 환경 변경')
    return h


def load_matching():
    folder=Path(__file__).parent/'source';m=read_json(folder/'matching_manifest.json');ld=read_json(folder/'excerpt_token_lengths.json')
    require(m['matching_hash']==MATCHING_HASH==digest({k:v for k,v in m.items() if k!='matching_hash'}),'대조 풀이 배정 변경')
    require(m['source_B_manifest_hash']==MANIFEST_HASH and m['lengths_hash']==digest(ld),'풀이 길이 자료 변경')
    B=load_manifest();ex={e['question_id']:e for e in B['excerpts']};lengths={r['question_id']:r for r in ld['lengths']}
    expected=sorted(e['question_id'] for e in B['excerpts'] if e['initial_cohort']=='wrong')
    require(m['question_ids']==expected and len(m['pairs'])==77,'매칭 문항 변경')
    donors=[];targets=[]
    for p in m['pairs']:
        q=p['question_id'];d=p['donor_question_id'];a,b=lengths[q],lengths[d];donors.append(d);targets.append(q)
        require(q!=d and a['subject']!=b['subject'] and p['subject']==a['subject'] and p['donor_subject']==b['subject'],'다른 과목 매칭 실패')
        require(p['original_excerpt_hash']==digest(ex[q]['text'])==a['text_hash'],'원 풀이 변경')
        require(p['control_text']==ex[d]['text'] and p['control_excerpt_hash']==digest(ex[d]['text'])==b['text_hash'],'대조 풀이 변경')
        require(p['original_excerpt_tokens']==a['tokens'] and p['control_excerpt_tokens']==b['tokens'],'토큰 길이 변경')
        require(p['token_difference']==b['tokens']-a['tokens'] and p['relative_token_difference']==(b['tokens']-a['tokens'])/a['tokens'],'길이 차이 계산 불일치')
    require(len(set(donors))==77 and sorted(targets)==expected,'중복 대조 풀이/문항 누락')
    require(sum(abs(p['relative_token_difference'])<=.05 for p in m['pairs'])==74,'길이 보조 분석 모집단 변경')
    return m


def selected_controls(source):return load_history(source)['records']


def default_config(source):
    cfg=copy.deepcopy(source['study']['config'])
    cfg.update(version=VERSION,phase='swapped_excerpt_control462',seed=202610312,conditions=['R1','R2','R3'],
      defenses=DEFENSE_NAMES[:],generate_conditions=['R3'],generate_defenses=NEW_DEFENSES[:],variants=[0,1,2],
      models={MODEL:cfg['models'][MODEL]},baseline_repetitions=0,new_followups=462,reused_followups=924,n_analysis_items=77,
      manifest_hash=MANIFEST_HASH,history_hash=HISTORY_HASH,matching_hash=MATCHING_HASH,exploratory=True,
      primary_family_size=1,primary_bootstrap_repetitions=20000,analysis_plan_hash=digest(ANALYSIS_PLAN))
    cfg['prompt_hash']=digest([original.BASE_SYSTEM,DEFENSES,original.STEMS,original.SUFFIX,'Explanation:\n'])
    return cfg


def make_jobs(source,cfg):
    require(cfg==default_config(source),'고정 실험 설정을 바꾸지 마세요.')
    require(all(cfg[k]==source['study']['config'][k] for k in GENERATION_KEYS),'원본 생성 설정 변경')
    history=load_history(source);matching=load_matching();pairs={p['question_id']:p for p in matching['pairs']}
    ex={r['question_id']:r['text'] for r in load_manifest()['excerpts']};jobs=[]
    for old in history['records']:
        if old['condition']!='R2':continue
        q=old['question_id'];p=pairs[q];messages=copy.deepcopy(old['messages'])
        claim,rest=messages[-1]['content'].split('\n\nExplanation:\n',1)
        require(rest==ex[q]+'\n\n'+original.SUFFIX,'원본 R2 구성 변경')
        messages[-1]['content']=claim+'\n\nExplanation:\n'+p['control_text']+'\n\n'+original.SUFFIX
        jobs.append({'task_id':f"CTRL462:{q}:{old['variant']}:R3:{old['defense']}", 'question_id':q,'phase':'followup','condition':'R3','defense':old['defense'],'variant':old['variant'],
          'baseline_hash':old['baseline_hash'],'rationale_hash':p['control_excerpt_hash'],'original_rationale_hash':p['original_excerpt_hash'],
          'donor_question_id':p['donor_question_id'],'donor_subject':p['donor_subject'],'historical_R2_task_id':old['task_id'],'messages':messages})
    random.Random(int_seed(cfg['seed'],'control462_schedule_v1')).shuffle(jobs)
    require(len(jobs)==len({j['task_id'] for j in jobs})==462,'신규 작업 누락/중복')
    return jobs


def fingerprint():return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in CODE_FILES}


def freeze(root,source,cfg):
    validate_source(source);h=load_history(source);m=load_manifest();matching=load_matching();jobs=make_jobs(source,cfg)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    frozen={'version':VERSION,'config':cfg,'source_hash':source['source_hash'],'history_hash':HISTORY_HASH,'manifest_hash':MANIFEST_HASH,'matching_hash':MATCHING_HASH,
      'jobs_hash':digest(jobs),'n_followups':462,'n_reused_followups':924,'question_ids':h['question_ids'],
      'tight_matching_question_ids':[p['question_id'] for p in matching['pairs'] if abs(p['relative_token_difference'])<=.05],
      'code_sha256':fingerprint(),'analysis_plan':ANALYSIS_PLAN,'defense_texts':DEFENSES,'baseline_regeneration':False,'historical_regeneration':False,'new_conditions':['R3'],'new_defenses':NEW_DEFENSES}
    frozen['study_hash']=digest(frozen)
    for name,val in [('study_lock.json',frozen),('source_snapshot.json',source),('history.json',h),('evidence_manifest.json',m),('matching_manifest.json',matching),('analysis_plan.json',ANALYSIS_PLAN),('new_jobs.json',jobs),('schedule.json',{'jobs_hash':digest(jobs),'batch_size':16,'task_ids':[j['task_id'] for j in jobs]})]:lock_json(root/name,val)
    source_files=[str(p.relative_to(Path(__file__).parent)) for p in (Path(__file__).parent/'source').glob('*') if p.is_file()]
    for n in CODE_FILES+source_files+['README_KO.txt']:
        data=(Path(__file__).parent/n).read_bytes();dest=root/'reproduction'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists() or dest.read_bytes()==data,f'저장 자료 변경: {n}')
        if not dest.exists():dest.write_bytes(data)
    return frozen,jobs


def verify_frozen(root,source,cfg,jobs):
    f,rebuilt=freeze(root,source,cfg);require(digest(jobs)==digest(rebuilt),'작업 내용/순서 변경');return f
