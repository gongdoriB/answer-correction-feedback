"""Build frozen replication inputs from original records, without new inference."""
from pathlib import Path
import sys, copy, hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from protocol import read_json,atomic_json,digest,int_seed,parse_answer,baseline_jobs,followup_jobs

def extend_matching(lengths, targets, preserved):
    pool=sorted(lengths,key=lambda x:x['question_id'])
    byid={r['question_id']:r for r in pool}
    used={p['donor_question_id'] for p in preserved}
    donors=[r for r in pool if r['question_id'] not in used]
    fixed={p['question_id'] for p in preserved}
    recipients=[byid[q] for q in sorted(set(targets)-fixed)]
    valid=np.array([[a['question_id']!=b['question_id'] and a['subject']!=b['subject'] and a['text_hash']!=b['text_hash'] for b in donors] for a in recipients])
    rel=np.array([[abs(a['tokens']-b['tokens'])/a['tokens'] for b in donors] for a in recipients])
    ts=np.unique(rel[valid]); lo,hi=0,len(ts)-1
    tie=np.array([[int_seed('replication1440_matching_v1',a['question_id'],b['question_id'])%1000000 for b in donors] for a in recipients])*1e-15
    def assign(t,weighted):
        allowed=valid&(rel<=t+1e-12)
        rows,cols=linear_sum_assignment(np.where(allowed,rel+tie if weighted else 0.,1e6))
        return bool(allowed[rows,cols].all()),rows,cols
    assert assign(ts[hi],False)[0]
    while lo<hi:
        mid=(lo+hi)//2
        if assign(ts[mid],False)[0]:hi=mid
        else:lo=mid+1
    ok,rows,cols=assign(ts[lo],True); assert ok
    out=copy.deepcopy(preserved)
    for row,col in zip(rows,cols):
        a,b=recipients[row],donors[col]
        out.append({'question_id':a['question_id'],'subject':a['subject'],'donor_question_id':b['question_id'],'donor_subject':b['subject'],
          'original_excerpt_hash':a['text_hash'],'control_excerpt_hash':b['text_hash'],'original_excerpt_tokens':a['tokens'],'control_excerpt_tokens':b['tokens'],
          'token_difference':b['tokens']-a['tokens'],'relative_token_difference':(b['tokens']-a['tokens'])/a['tokens']})
    return sorted(out,key=lambda p:p['question_id'])

def read_batches(folder,jobs):
    records=[];audit=[]
    for p in sorted(folder.glob('batch_*.json')):
        b=read_json(p);rows=b['records'];batch=jobs[len(records):len(records)+len(rows)]
        assert b['records_hash']==digest(rows) and b['batch_hash']==digest(batch)
        assert all(all(r.get(k)==v for k,v in j.items()) for r,j in zip(rows,batch))
        records.extend(rows)
        audit.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'count':len(rows)})
    assert len(records)==len(jobs)
    return records,audit

def prepare():
    root=HERE.parents[2];old=HERE.parent/'sycophancy_qwen_8b_control462_support';out=HERE/'source'
    main=root/'output/sycophancy_results_audit/main_20261026/received/sycophancy_qwen_core_v3_formatfix_a100_b16/main'
    study=read_json(main/'study_lock.json');modeldir=main/'runs/a100_b16/qwen3_4b'
    bases,audit=read_batches(modeldir/'baseline',baseline_jobs(study['items']))
    bybase={r['question_id']:r for r in bases}
    follow,fa=read_batches(modeldir/'followup',followup_jobs(study['items'],bybase,study['config']))
    counts={'correct':0,'wrong':0,'invalid':0};wrong=[]
    for i in study['items']:
        b=bybase[i['question_id']];a,s=parse_answer(b['text'],i['labels'])
        c='invalid' if b['truncated'] or s!='valid' else 'correct' if a==i['gold'] else 'wrong'
        counts[c]+=1
        if c=='wrong':wrong.append(i['question_id'])
    assert counts=={'correct':125,'wrong':167,'invalid':8}
    source={'study':study,'baselines':bases,'run':read_json(modeldir/'run_lock.json'),'counts':counts,
      'controls':[r for r in follow if r['condition']=='R1'],
      'provenance':{'baseline_files':audit,'followup_files':fa,'controls_scope':'Historical 4B S0/S1 R1, 1800 records; reference only, never relabeled as SC/SI.'}}
    source['source_hash']=digest(source);atomic_json(out/'source_snapshot.json',source)
    source8=read_json(old/'source/source_snapshot.json');h8=read_json(old/'source/history.json')
    run8=root/'output/sycophancy_results_audit/control462_20261031/received/sycophancy_qwen3_8b_control462_a100_b16_v1'
    r3,ra=read_batches(run8/'checkpoints',read_json(run8/'new_jobs.json'))
    oldmatch=read_json(old/'source/matching_manifest.json')
    history8={'model':'qwen3_8b','baselines':source8['baselines'],'records':h8['records']+r3,'question_ids':h8['question_ids'],
      'run':source8['run'],'matching_manifest':oldmatch,'source_history_hash':h8['history_hash'],
      'provenance':{'R1_R2':h8['provenance'],'R3':{'files':ra,'study_lock':read_json(run8/'study_lock.json'),'environment_lock':read_json(run8/'environment_lock.json')}}}
    history8['history_hash']=digest(history8);atomic_json(out/'history8b.json',history8)
    B=read_json(out/'evidence_manifest.json');ex={e['question_id']:e for e in B['excerpts']}
    ids=sorted(set(wrong)&set(ex));common=sorted(set(ids)&set(h8['question_ids']))
    assert len(ids)==80 and len(common)==61
    from transformers import AutoTokenizer
    cached=root/'output/jupyter-notebook/sycophancy_qwen_support/validation/tokenizers/qwen3_4b'
    tok=AutoTokenizer.from_pretrained(cached,local_files_only=True)
    assert digest(tok.chat_template)==source['run']['environment']['chat_template_hash']
    ld={'model_revision':study['config']['models']['qwen3_4b']['revision'],'chat_template_hash':digest(tok.chat_template),
      'tokenizer_source':'Previously cached official Qwen3-4B tokenizer; every prompt is checked again with pinned remote revision at Colab runtime.',
      'cached_files_sha256':{n:hashlib.sha256((cached/n).read_bytes()).hexdigest() for n in ['tokenizer.json','tokenizer_config.json']},
      'lengths':[{'question_id':q,'subject':e['subject'],'text_hash':digest(e['text']),'tokens':len(tok.encode(e['text'],add_special_tokens=False))} for q,e in sorted(ex.items())]}
    oldlengths=read_json(old/'source/excerpt_token_lengths.json')
    assert ld['lengths']==sorted([{k:v for k,v in r.items() if k!='initial_cohort'} for r in oldlengths['lengths']],key=lambda x:x['question_id'])
    atomic_json(out/'excerpt_token_lengths.json',ld)
    preserved=[p for p in oldmatch['pairs'] if p['question_id'] in common]
    pairs=extend_matching(ld['lengths'],ids,preserved)
    for p in pairs:p['control_text']=ex[p['donor_question_id']]['text']
    assert len({p['donor_question_id'] for p in pairs})==80
    matching={'policy':{'common_items':'Preserve the exact 8B donor and text for all 61 common initially-wrong items.',
      'additional_items':'Assign 19 additional 4B targets one-to-one from the 155 frozen excerpt pool, excluding donors already used by the common 61. Minimize maximum proportional token difference, then sum, with fixed ID-based tie-break.',
      'eligibility':'Different question, different subject, nonidentical excerpt. No generated, shortened or edited text.',
      'outcome_use':'Selection uses baseline validity/wrongness and frozen excerpt availability. Matching uses IDs, subject, token lengths and hashes only; no follow-up outcome.',
      'limits':'One frozen donor assignment, approximate length control, no independent semantic validation.'},
      'question_ids':ids,'common8b_question_ids':common,'source_8b_matching_hash':oldmatch['matching_hash'],
      'lengths_hash':digest(ld),'source_B_manifest_hash':B['manifest_hash'],'pairs':pairs,
      'summary':{'n_targets':80,'n_common_with_8b':61,'n_new_assignments':19,'n_unique_donors':80,
      'exact_token_matches':sum(p['token_difference']==0 for p in pairs),
      'within_5_percent':sum(abs(p['relative_token_difference'])<=.05 for p in pairs),
      'max_abs_relative_difference_percent':100*max(abs(p['relative_token_difference']) for p in pairs)}}
    matching['matching_hash']=digest(matching);atomic_json(out/'matching_manifest.json',matching)
    atomic_json(out/'source_lock.json',{'source_hash':source['source_hash'],'history8b_hash':history8['history_hash'],
      'manifest_hash':B['manifest_hash'],'matching_hash':matching['matching_hash'],'study_hash':study['study_hash']})
    print(counts,matching['summary'])

if __name__=='__main__':prepare()
