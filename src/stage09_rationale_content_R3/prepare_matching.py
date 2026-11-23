"""Outcome-blind, one-to-one cross-subject matching of intact previously frozen B excerpts."""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from protocol import read_json,atomic_json,digest,int_seed

POLICY={
 'candidate_pool':'All 155 excerpts from the fixed B max5 manifest, irrespective of original correctness or follow-up responses.',
 'eligibility':'Different question, different subject, non-identical normalized excerpt; no donor reused among the 77 targets.',
 'matching':'First minimize the largest proportional difference in excerpt token count; within that threshold minimize the total proportional difference. Fixed hashed tie-breaker.',
 'tokenizer':'Same pinned Qwen3-8B tokenizer. Count original excerpt without special tokens. Audit all complete recipient prompts afterward.',
 'editing':'No cropping, padding, paraphrasing, label replacement or new generation of donor text. Reuse complete selected B excerpt unchanged.',
 'replicates':'One donor per target, identical donor across SC/SI and all three variants.',
 'outcome_access':'Match using only IDs, subjects, token lengths and excerpt hashes. Never use follow-up correctness, selected answer or gold for matching.',
 'semantic_status':'Cross-subject swapped-excerpt control, not guaranteed irrelevant or necessarily false. Single fixed assignment; no causal isolation of length alone.'}

def select_pairs(length_data,question_ids):
    pool=sorted(length_data['lengths'],key=lambda x:x['question_id']);byid={r['question_id']:r for r in pool}
    target=[byid[q] for q in sorted(question_ids)]
    valid=np.array([[a['question_id']!=b['question_id'] and a['subject']!=b['subject'] and a['text_hash']!=b['text_hash'] for b in pool] for a in target])
    rel=np.array([[abs(a['tokens']-b['tokens'])/a['tokens'] for b in pool] for a in target])
    thresholds=np.unique(rel[valid]);lo,hi=0,len(thresholds)-1
    def assign(threshold,weighted):
        allow=valid&(rel<=threshold+1e-12)
        tie=np.array([[int_seed('control462_matching_v1',a['question_id'],b['question_id'])%1000000 for b in pool] for a in target])*1e-15
        cost=np.where(allow,rel+tie if weighted else 0.,1e6)
        rows,cols=linear_sum_assignment(cost)
        return bool(allow[rows,cols].all()),rows,cols
    assert assign(thresholds[hi],False)[0]
    while lo<hi:
        mid=(lo+hi)//2
        if assign(thresholds[mid],False)[0]:hi=mid
        else:lo=mid+1
    threshold=float(thresholds[lo]);ok,rows,cols=assign(threshold,True);assert ok
    pairs=[]
    for row,col in zip(rows,cols):
        a,b=target[row],pool[col]
        pairs.append({'question_id':a['question_id'],'subject':a['subject'],'donor_question_id':b['question_id'],'donor_subject':b['subject'],
                      'original_excerpt_hash':a['text_hash'],'control_excerpt_hash':b['text_hash'],
                      'original_excerpt_tokens':a['tokens'],'control_excerpt_tokens':b['tokens'],
                      'token_difference':b['tokens']-a['tokens'],'relative_token_difference':(b['tokens']-a['tokens'])/a['tokens']})
    return pairs,{'max_absolute_proportional_error':threshold,'total_absolute_proportional_error':sum(abs(r['relative_token_difference']) for r in pairs)}


def main():
    folder=Path(__file__).parent/'source';ld=read_json(folder/'excerpt_token_lengths.json');h=read_json(folder/'history.json');m=read_json(folder/'evidence_manifest.json')
    pairs,objective=select_pairs(ld,h['question_ids']);ex={e['question_id']:e for e in m['excerpts']}
    for p in pairs:p['control_text']=ex[p['donor_question_id']]['text']
    lengths=np.array([abs(p['token_difference']) for p in pairs]);rel=np.array([abs(p['relative_token_difference']) for p in pairs])
    manifest={'policy':POLICY,'source_B_manifest_hash':m['manifest_hash'],'lengths_hash':digest(ld),'question_ids':h['question_ids'],'pairs':pairs,'optimization':objective,
              'summary':{'n_targets':len(pairs),'n_unique_donors':len({p['donor_question_id'] for p in pairs}),'candidate_pool_size':len(ex),
                         'exact_token_matches':int((lengths==0).sum()),'mean_abs_token_difference':float(lengths.mean()),'max_abs_token_difference':int(lengths.max()),
                         'mean_abs_relative_difference_percent':float(rel.mean()*100),'max_abs_relative_difference_percent':float(rel.max()*100),
                         'within_5_percent':int((rel<=.05).sum()),'within_10_percent':int((rel<=.1).sum())}}
    manifest['matching_hash']=digest(manifest);atomic_json(folder/'matching_manifest.json',manifest)
    import csv
    with (folder/'matching_pairs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=[k for k in pairs[0] if k!='control_text'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in pairs)
    print('matching hash',manifest['matching_hash']);print(manifest['summary'])
if __name__=='__main__':main()
