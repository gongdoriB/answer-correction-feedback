"""Offline tests. Synthetic responses are never research evidence."""
import copy,json,sys,tempfile,unittest,io
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ab_protocol import *
from ab_run import ABBackend,bind_backend,run_followups,scan_saved,validate_source_environment,load_complete
from ab_analysis import score_records,contrast_values,comparison_table,analyze
from protocol import atomic_json

class SyntheticBackend:
 def __init__(self,source,cfg):
  self.environment=copy.deepcopy(source['run']['environment']);self.environment['synthetic_test']=True
  self.source_environment_notes={};self.items={i['question_id']:i for i in source['study']['items']};self.bases={r['question_id']:r for r in source['baselines']};self.calls=[];self.fail=False
 def preflight(self,jobs):return {'count':len(jobs),'max_input_tokens_observed':777}
 def generate(self,jobs,seed):
  if self.fail:raise KeyboardInterrupt('SYNTHETIC interruption')
  self.calls.append((seed,[j['task_id'] for j in jobs]));out=[]
  for j in jobs:
   assert j['condition']=='R2' and j['defense'] in ['SC','SI','SR']
   i=self.items[j['question_id']];b,_=parse_answer(self.bases[j['question_id']]['text'],i['labels']);n=int_seed('TEST ONLY',j['task_id'])%100
   a=i['gold'] if n<80 else b
   out.append({'text':json.dumps({'answer':a,'explanation':'SYNTHETIC TEST ONLY'}) if n!=99 else 'INVALID TEST', 'truncated':n==98,'input_tokens':500,'output_tokens':30,'rendered_prompt_hash':digest(j['messages']),'seconds_per_batch':.01})
  self.last_batch_resource={'generation_seconds':.01,'scope':'SYNTHETIC TEST ONLY'};return out
 def close(self):pass

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=load_source(ROOT/'source');cls.h=load_history(cls.s);cls.cfg=default_config(cls.s);cls.jobs=make_jobs(cls.s,cls.cfg)
 def test_grid_and_exact_history(self):
  self.assertEqual(len(self.jobs),693);self.assertEqual(len(self.h['records']),1617)
  self.assertEqual(Counter(j['defense'] for j in self.jobs),{'SC':231,'SI':231,'SR':231})
  self.assertEqual(self.jobs,make_jobs(self.s,self.cfg))
  old={r['task_id']:r for r in self.h['records']}
  for j in self.jobs:
   self.assertEqual(j['messages'][1:],old[j['historical_B_task_id']]['messages'][1:])
   self.assertEqual(j['messages'][:3],old[j['historical_R1_task_id']]['messages'][:3])
  f=score_records(self.s,self.h['records']);counts=f.groupby(['condition','defense']).final_correct.sum().to_dict()
  self.assertEqual(counts,{('R1','S0'):181,('R1','SC'):191,('R1','SI'):125,('R1','SR'):192,('R1','S1'):141,('R2','S0'):190,('R2','S1'):189})
 def test_guards(self):
  bad=copy.deepcopy(self.cfg);bad['batch_size']=8
  with self.assertRaises(RuntimeError):make_jobs(self.s,bad)
  env=copy.deepcopy(self.s['run']['environment']);env['gpu']='L4'
  with self.assertRaises(RuntimeError):validate_source_environment(env,self.s)
  obj=object.__new__(ABBackend)
  for r in self.h['records'][::100]:
   with self.assertRaises(RuntimeError):obj.generate([r],1)
 def test_question_unit_and_invalids(self):
  import pandas as pd
  backend=SyntheticBackend(self.s,self.cfg);new=[{**j,**o} for j,o in zip(self.jobs,backend.generate(self.jobs,1))]
  f=score_records(self.s,self.h['records']+new);self.assertEqual(len(f),2310)
  self.assertTrue(f[['final_correct','initial_wrong_retained','other_wrong_change','invalid']].sum(axis=1).eq(1).all())
  self.assertFalse((f.invalid.astype(bool)&f.final_correct.astype(bool)).any())
  t=comparison_table(f);self.assertEqual(len(t),10)
  f['final_correct']=(f.defense=='SI').astype(int)
  v,_=contrast_values(f,ANALYSIS_PLAN['primary_contrasts'][0]['terms'],'final_correct',False)
  self.assertEqual(len(v),77);self.assertTrue(v.eq(0).all())
  with self.assertRaises(RuntimeError):contrast_values(f.drop(f[(f.condition=='R2')&(f.defense=='SI')].index[0]),ANALYSIS_PLAN['primary_contrasts'][0]['terms'],'final_correct',False)
 def test_interrupt_resume_tamper_and_complete_analysis(self):
  with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
   root=Path(d);f,jobs=freeze(root,self.s,self.cfg);b=SyntheticBackend(self.s,self.cfg);bind_backend(root,f,self.s,b,jobs)
   status=run_followups(root,f,self.s,self.cfg,jobs,b,max_new_batches=1);self.assertEqual(status['saved_responses'],16)
   first=(root/'checkpoints/batch_000000.json').read_bytes();b.fail=True
   with self.assertRaises(KeyboardInterrupt):run_followups(root,f,self.s,self.cfg,jobs,b)
   self.assertEqual(read_json(root/'status.json')['saved_responses'],16)
   with self.assertRaises(RuntimeError):load_complete(root,f,self.s,self.cfg,jobs)
   b.fail=False;status=run_followups(root,f,self.s,self.cfg,jobs,b);self.assertTrue(status['complete'])
   self.assertEqual(status['saved_responses'],693);self.assertEqual(status['saved_batches'],44)
   self.assertEqual((root/'checkpoints/batch_000000.json').read_bytes(),first)
   self.assertEqual(len(b.calls),44);self.assertEqual(len(b.calls[-1][1]),5)
   self.assertEqual(len({t for _,batch in b.calls for t in batch}),693)
   run_followups(root,f,self.s,self.cfg,jobs,b);self.assertEqual(len(b.calls),44)
   folder,table,effects,a=analyze(root,f,self.s,self.cfg,jobs)
   self.assertEqual(len(table),10);self.assertEqual(sum(effects.analysis_role=='primary'),2)
   self.assertEqual(set(effects.n_items),{77});self.assertTrue(effects[effects.analysis_role=='primary'].interval_level.eq(.975).all())
   self.assertTrue((folder/'01_correction_comparison.png').exists())
   p=root/'checkpoints/batch_000000.json';obj=read_json(p);obj['records'][0]['text']='CORRUPTED';atomic_json(p,obj)
   with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)
   p.write_bytes(first)
   (root/'checkpoints/batch_999999.json').write_text('{}')
   with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
