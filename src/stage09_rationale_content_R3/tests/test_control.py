"""Offline protocol and persistence tests; synthetic outputs are never research evidence."""
import copy,json,sys,tempfile,unittest,io
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from control_protocol import *
from control_run import ControlBackend,bind_backend,run_followups,scan_saved,load_complete,validate_source_environment
from control_analysis import score_records,contrast_values,comparison_table,analyze,answer_field_diagnostic
from prepare_matching import select_pairs
from protocol import atomic_json

class SyntheticBackend:
 def __init__(self,source,cfg):
  self.environment=copy.deepcopy(source['run']['environment']);self.environment['synthetic_test']=True
  self.source_environment_notes={};self.items={i['question_id']:i for i in source['study']['items']};self.bases={r['question_id']:r for r in source['baselines']};self.calls=[];self.fail=False
 def preflight(self,jobs):return {'count':len(jobs),'max_input_tokens_observed':777}
 def generate(self,jobs,seed):
  if self.fail:raise KeyboardInterrupt('SYNTHETIC interruption')
  self.calls.append((seed,[j['task_id'] for j in jobs]));outputs=[]
  for j in jobs:
   assert j['condition']=='R3' and j['defense'] in ['SC','SI']
   item=self.items[j['question_id']];b,_=parse_answer(self.bases[j['question_id']]['text'],item['labels']);n=int_seed('SYNTHETIC ONLY',j['task_id'])%100
   a=item['gold'] if n < (80 if j['defense']=='SC' else 64) else b
   txt=json.dumps({'answer':a,'explanation':'SYNTHETIC TEST ONLY'})
   if n==97:txt+='.'
   if n==99:txt='INVALID TEST'
   outputs.append({'text':txt,'truncated':n==98,'input_tokens':500,'output_tokens':30,'rendered_prompt_hash':digest(j['messages']),'seconds_per_batch':.01})
  self.last_batch_resource={'generation_seconds':.01,'scope':'SYNTHETIC TEST ONLY'};return outputs
 def close(self):pass

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=load_source(ROOT/'source');cls.h=load_history(cls.s);cls.m=load_matching();cls.cfg=default_config(cls.s);cls.jobs=make_jobs(cls.s,cls.cfg)
 def test_matching_no_outcomes_no_text_edits(self):
  ld=read_json(ROOT/'source/excerpt_token_lengths.json');pairs,objective=select_pairs(ld,self.h['question_ids'])
  actual=[{k:v for k,v in p.items() if k!='control_text'} for p in self.m['pairs']]
  self.assertEqual(pairs,actual);self.assertEqual(objective,self.m['optimization'])
  self.assertEqual(len(set(p['donor_question_id'] for p in pairs)),77)
  self.assertTrue(all(p['subject']!=p['donor_subject'] for p in pairs))
  self.assertEqual(sum(abs(p['relative_token_difference'])<=.05 for p in pairs),74)
  audit=read_json(ROOT/'source/token_audit.json');self.assertEqual(audit['jobs_hash'],digest(self.jobs));self.assertEqual(audit['count'],462);self.assertLessEqual(audit['max_input_tokens_observed'],4096)
 def test_exact_reuse_and_only_explanation_changed(self):
  self.assertEqual(len(self.h['records']),924);self.assertEqual(len(self.jobs),462);self.assertEqual(Counter(j['defense'] for j in self.jobs),{'SC':231,'SI':231})
  self.assertEqual(self.jobs,make_jobs(self.s,self.cfg))
  old={r['task_id']:r for r in self.h['records']};pairs={p['question_id']:p for p in self.m['pairs']}
  for j in self.jobs:
   r=old[j['historical_R2_task_id']];self.assertEqual(j['messages'][:3],r['messages'][:3])
   oldclaim=r['messages'][-1]['content'].split('\n\nExplanation:\n')[0]
   self.assertEqual(j['messages'][-1]['content'],oldclaim+'\n\nExplanation:\n'+pairs[j['question_id']]['control_text']+'\n\n'+original.SUFFIX)
  f=score_records(self.s,self.h['records'])
  self.assertEqual(f.groupby(['condition','defense']).final_correct.sum().to_dict(),{('R1','SC'):191,('R1','SI'):125,('R2','SC'):194,('R2','SI'):179})
 def test_config_environment_and_generation_guards(self):
  bad=copy.deepcopy(self.cfg);bad['batch_size']=8
  with self.assertRaises(RuntimeError):make_jobs(self.s,bad)
  bads=copy.deepcopy(self.s);bads['baselines'][0]['text']='CHANGED'
  with self.assertRaises(Exception):validate_source(bads)
  env=copy.deepcopy(self.s['run']['environment']);env['gpu']='L4'
  with self.assertRaises(RuntimeError):validate_source_environment(env,self.s)
  obj=object.__new__(ControlBackend)
  for r in self.h['records'][::100]:
   with self.assertRaises(RuntimeError):obj.generate([r],1)
 def test_question_pairing_and_diagnostic_separation(self):
  b=SyntheticBackend(self.s,self.cfg);new=[{**j,**o} for j,o in zip(self.jobs,b.generate(self.jobs,1))]
  f=score_records(self.s,self.h['records']+new);self.assertEqual(len(f),1386)
  self.assertTrue(f[['final_correct','initial_wrong_retained','other_wrong_change','invalid']].sum(axis=1).eq(1).all())
  strict=f.final_correct.copy();diag=answer_field_diagnostic(f)
  self.assertTrue(f.final_correct.equals(strict));self.assertTrue(diag[diag.truncated].final_correct.eq(0).all());self.assertGreater(diag.diagnostic_rescored.sum(),0)
  self.assertEqual(len(comparison_table(f)),6)
  f['final_correct']=(f.defense=='SI').astype(int)
  v,_=contrast_values(f,PRIMARY_TERMS,'final_correct',False);self.assertEqual(len(v),77);self.assertTrue(v.eq(0).all())
  idx=f[(f.condition=='R3')&(f.defense=='SI')].index[0]
  with self.assertRaises(RuntimeError):contrast_values(f.drop(idx),PRIMARY_TERMS,'final_correct',False)
 def test_interrupt_resume_corruption_and_analysis(self):
  with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
   root=Path(d);f,jobs=freeze(root,self.s,self.cfg);b=SyntheticBackend(self.s,self.cfg);bind_backend(root,f,self.s,b,jobs)
   status=run_followups(root,f,self.s,self.cfg,jobs,b,max_new_batches=1);self.assertEqual(status['saved_responses'],16)
   first=(root/'checkpoints/batch_000000.json').read_bytes();b.fail=True
   with self.assertRaises(KeyboardInterrupt):run_followups(root,f,self.s,self.cfg,jobs,b)
   self.assertEqual(read_json(root/'status.json')['saved_responses'],16)
   with self.assertRaises(RuntimeError):load_complete(root,f,self.s,self.cfg,jobs)
   b.fail=False;status=run_followups(root,f,self.s,self.cfg,jobs,b);self.assertTrue(status['complete']);self.assertEqual(status['saved_responses'],462);self.assertEqual(status['saved_batches'],29)
   self.assertEqual((root/'checkpoints/batch_000000.json').read_bytes(),first);self.assertEqual(len(b.calls),29);self.assertEqual(len(b.calls[-1][1]),14)
   self.assertEqual(len({task for seed,batch in b.calls for task in batch}),462)
   run_followups(root,f,self.s,self.cfg,jobs,b);self.assertEqual(len(b.calls),29)
   folder,table,effects,assessment=analyze(root,f,self.s,self.cfg,jobs)
   self.assertEqual(len(table),6);self.assertEqual(sum(effects.analysis_role=='primary'),1);self.assertEqual(set(effects.n_items),{77});self.assertEqual(assessment['execution_mode'],'SYNTHETIC_TEST_ONLY')
   import pandas as pd
   self.assertEqual(set(pd.read_csv(folder/'tight74_effects.csv').n_items),{74});self.assertEqual(set(pd.read_csv(folder/'format_diagnostic_effects.csv').n_items),{77})
   self.assertTrue((folder/'01_control_comparison.png').exists())
   p=root/'checkpoints/batch_000000.json';data=read_json(p);data['records'][0]['text']='CORRUPTED';atomic_json(p,data)
   with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)
   p.write_bytes(first);(root/'checkpoints/batch_999999.json').write_text('{}')
   with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
