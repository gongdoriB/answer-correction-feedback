"""Check every received input length and rendered hash with cached official tokenizer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
SUPPORT=PROJECT/'output/jupyter-notebook/sycophancy_qwen_4b_replication1440_support'
sys.path.insert(0,str(SUPPORT))
from transformers import AutoTokenizer
from protocol import read_json,digest,atomic_json
received=ROOT/'received/sycophancy_qwen3_4b_replication1440_a100_b16_v1'
jobs=read_json(received/'new_jobs.json')
records=[r for p in sorted((received/'checkpoints').glob('batch_*.json')) for r in read_json(p)['records']]
tok=AutoTokenizer.from_pretrained(PROJECT/'output/jupyter-notebook/sycophancy_qwen_support/validation/tokenizers/qwen3_4b',local_files_only=True)
assert digest(tok.chat_template)==read_json(received/'environment_lock.json')['chat_template_hash']
rows=[]
for j,r in zip(jobs,records):
    text=tok.apply_chat_template(j['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n=len(tok.encode(text,add_special_tokens=False))
    assert j['task_id']==r['task_id'] and digest(text)==r['rendered_prompt_hash'] and n==r['input_tokens']
    rows.append({'task_id':j['task_id'],'tokens':n,'rendered_prompt_hash':digest(text)})
assert len(rows)==1440
atomic_json(ROOT/'verified_input_token_hashes.json',{'count':1440,'all_matched':True,'max_input_tokens':max(x['tokens'] for x in rows),'rows':rows})
audit=read_json(ROOT/'audit.json');audit['checks']['all_1440_rendered_prompt_hashes_and_lengths_recomputed']=True
audit['workflow_reference']='Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 (current v2 metadata verified 2026-11-01; workflow attribution, not evidence for the experimental findings).'
atomic_json(ROOT/'audit.json',audit)
print('All 1440 rendered prompts and token lengths independently matched. No model inference performed.')
