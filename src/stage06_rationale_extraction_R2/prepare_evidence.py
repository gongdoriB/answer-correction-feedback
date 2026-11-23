"""Freeze max-five-unit contiguous excerpts before any new R2 model inference."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
from ablation_protocol import load_source
from extract_evidence import POLICY, normalize_explanation
from sentence_units import count_units
from protocol import atomic_json, digest, SUFFIX

ROOT=Path(__file__).resolve().parent
SOURCE_MATCHED_HASH='252179cc704efd9edf29669e4cab0184f8320366982f3bb5e2be1f4c91421831'
ARCHIVE_SHA='5f5309a3bcdd2abe6d9af3f6925497d6660149c12690154baac3562fc04d9d73'
COMMIT='f418b116db00b065c2aea046518d8fcf74d39872'


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def build():
    from transformers import AutoTokenizer
    source=load_source(ROOT/'source')
    matched=json.loads((ROOT/'source/official_matched.json').read_text())
    assert digest(matched)==SOURCE_MATCHED_HASH
    tokdir=ROOT.parent/'sycophancy_qwen_support/validation/tokenizers/qwen3_8b'
    tokenizer=AutoTokenizer.from_pretrained(str(tokdir),local_files_only=True,trust_remote_code=False)
    assert digest(tokenizer.chat_template)==source['run']['environment']['chat_template_hash']
    items={x['question_id']:x for x in source['study']['items']}
    controls={}
    for r in source['controls']:
        if r['condition']=='R1':controls.setdefault(r['question_id'],[]).append(r)
    selected=[];audit=[];transforms=[];token_rows=[]
    for row in matched:
        qid=row['question_id'];raw=row['official_record'];item=items[qid]
        assert ' '.join(raw['question'].split())==' '.join(item['question'].split())
        assert [' '.join(s.split()) for s in raw['options']]==[' '.join(s.split()) for s in item['options']]
        assert raw['answer']==item['gold']
        reason='teacher_final_label_wrong' if raw['pred']!=raw['answer'] else None
        clean=None;edits=[];lengths=[]
        if not reason:
            clean,reason,edits=normalize_explanation(raw['response'],item['options'],item['labels'])
        if clean:
            assert len(controls[qid])==6
            for old in controls[qid]:
                messages=copy.deepcopy(old['messages'])
                claim,suffix=messages[-1]['content'].split('\n\n',1);assert suffix==SUFFIX
                messages[-1]['content']=claim+'\n\nExplanation:\n'+clean+'\n\n'+suffix
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                n=len(tokenizer.encode(prompt,add_special_tokens=False));lengths.append(n)
                token_rows.append({'question_id':qid,'defense':old['defense'],'variant':old['variant'],'input_tokens':n,'rendered_prompt_hash':digest(prompt)})
            if max(lengths)>source['study']['config']['max_input_tokens']:reason='full_input_over_4096'
        audit.append({'question_id':qid,'subject':row['subject'],'initial_cohort':row['initial_cohort'],
                      'teacher_correct':raw['pred']==raw['answer'],'eligible':reason is None,
                      'exclusion_reason':reason or '', 'max_R2_input_tokens':max(lengths,default=0),
                      'original_words':len(raw['response'].split()),'normalized_words':len(clean.split()) if clean else 0,
                      'normalization_operations':len(edits)})
        transforms.append({'question_id':qid,'edits':edits})
        if reason is None:
            selected.append({'question_id':qid,'subject':row['subject'],'initial_cohort':row['initial_cohort'],
                             'gold':raw['answer'],'text':clean,'normalization_edits':edits,
                             'response_sha256':hashlib.sha256(raw['response'].encode()).hexdigest(),
                             'original_words':len(raw['response'].split()),'normalized_words':len(clean.split()),
                             'max_R2_input_tokens':max(lengths),'sentence_cap':5,**count_units(clean),
                             'legacy_sentence_math_units':edits[-1]['sentences'],
                             'normalized_source_start':edits[-1]['start'],'normalized_source_end':edits[-1]['end']})
    cohorts=Counter(r['initial_cohort'] for r in selected)
    availability=[{'stage':'original_300','total':len(matched),**{k:sum(r['initial_cohort']==k for r in matched) for k in ['wrong','correct','invalid']}},
                  {'stage':'teacher_final_correct','total':sum(r['official_record']['pred']==r['official_record']['answer'] for r in matched),
                   **{k:sum(r['initial_cohort']==k and r['official_record']['pred']==r['official_record']['answer'] for r in matched) for k in ['wrong','correct','invalid']}},
                  {'stage':'max5_contiguous_excerpt_eligible','total':len(selected),**{k:cohorts[k] for k in ['wrong','correct','invalid']}}]
    manifest={'policy':POLICY,'availability':availability,'selected_cap':5,'excerpts':selected,
              'original_questions':300,'official_teacher_correct':availability[1]['total'],
              'matched_source_hash':SOURCE_MATCHED_HASH,'new_followups':len(selected)*6,'reuse_R1_followups':len(selected)*6,
              'token_validation':{'chat_template_hash':digest(tokenizer.chat_template),'inputs_checked':len(token_rows),
                                  'max_input_tokens':max(r['input_tokens'] for r in token_rows),'limit':4096,
                                  'over_limit_questions':sum(r['exclusion_reason']=='full_input_over_4096' for r in audit),
                                  'tokenizer_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tokdir.glob('*.json')}},
              'source':{'repository':'https://github.com/TIGER-AI-Lab/MMLU-Pro','commit':COMMIT,
                        'archive':'model_outputs_gpt4o(2024-05-13)_5shots.zip','archive_sha256':ARCHIVE_SHA,
                        'url':f'https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/{COMMIT}/eval_results/model_outputs_gpt4o%282024-05-13%29_5shots.zip',
                        'teacher':'GPT-4o (2024-05-13), five-shot','human_verified_rationale':False},
              'logical_quality_verified':False,'selection_before_new_inference':True,
              'supersedes_short_version_for_new_execution':True,'pool_different_excerpt_versions_R2':False,
              'key_reasoning_preservation_verified':False}
    manifest['manifest_hash']=digest(manifest)
    atomic_json(ROOT/'source/evidence_manifest.json',manifest)
    atomic_json(ROOT/'source/normalization_audit.json',transforms)
    write_csv(ROOT/'source/eligibility_audit.csv',audit)
    write_csv(ROOT/'source/availability.csv',availability)
    write_csv(ROOT/'source/token_lengths.csv',token_rows)
    write_csv(ROOT/'source/selected_excerpts.csv',[{k:v for k,v in r.items() if k!='normalization_edits'} for r in selected])
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['excerpts','policy']},ensure_ascii=False,indent=2))


if __name__=='__main__':build()
