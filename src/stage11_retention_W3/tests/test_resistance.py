"""Offline invariants and end-to-end synthetic execution; NOT research outcomes."""
import copy,json,sys,tempfile,unittest,io
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resistance_protocol import *
from resistance_run import ResistanceBackend,bind_backend,run_followups,scan_saved,load_complete,validate_source_environment,export_results
from resistance_analysis import score_records,contrast_values,comparison_table,analyze,answer_field_diagnostic,PRIMARY_NAME
from protocol import atomic_json

class SyntheticBackend:
 def __init__(self,source,cfg):
  self.environment=copy.deepcopy(source['run']['environment']);self.environment['synthetic_test']=True
  self.source_environment_notes={};self.items={i['question_id']:i for i in source['study']['items']};self.calls=[];self.fail=False
 def preflight(self,jobs):return {'count':len(jobs),'max_input_tokens_observed':1032}
 def generate(self,jobs,seed):
  if self.fail:raise KeyboardInterrupt('SYNTHETIC interruption')
  self.calls.append((seed,[j['task_id'] for j in jobs]));outputs=[]
  for j in jobs:
   assert j['condition']=='W3' and j['defense'] in ['SC','SI']
   i=self.items[j['question_id']];n=int_seed('SYNTHETIC ONLY',j['task_id'])%100
   a=i['gold'] if n<(60 if j['defense']=='SC' else 80) else i['target_wrong']
   if n==95:a=next(x for x in i['labels'] if x not in [i['gold'],i['target_wrong']])
   txt=json.dumps({'answer':a,'explanation':'SYNTHETIC TEST ONLY'})
   if n==97:txt+='.'
   if n==99:txt='INVALID TEST'
   outputs.append({'text':txt,'truncated':n==98,'input_tokens':500,'output_tokens':30,'rendered_prompt_hash':digest(j['messages']),'seconds_per_batch':.01})
  self.last_batch_resource={'generation_seconds':.01,'scope':'SYNTHETIC TEST ONLY'};return outputs
 def close(self):pass

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=load_source(ROOT/'source');cls.h=load_history(cls.s);cls.m=load_matching(cls.s);cls.cfg=default_config(cls.s);cls.jobs=make_jobs(cls.s,cls.cfg)
 def test_population_assignment_and_token_audit(self):
  self.assertEqual(len(self.h['question_ids']),133);self.assertEqual(len(self.m['pairs']),133)
  self.assertTrue(all(p['subject']!=p['donor_subject'] and p['question_id']!=p['donor_question_id'] for p in self.m['pairs']))
  self.assertEqual(len({p['donor_question_id'] for p in self.m['pairs']}),133)
  self.assertTrue(all(1<=p['sentence_count_auto']<=5 and p['word_count']<=180 for p in self.m['pairs']))
  B=load_manifest();own={e['question_id'] for e in B['excerpts']}
  self.assertEqual(len(set(self.h['question_ids'])-own),57)
  audit=read_json(ROOT/'source/token_audit.json');self.assertEqual(audit['jobs_hash'],digest(self.jobs));self.assertEqual(audit['count'],798);self.assertEqual(audit['max_input_tokens_observed'],1032)
 def test_history_exact_reuse_and_claim_unchanged(self):
  self.assertEqual(len(self.h['records']),798);self.assertEqual(len(self.jobs),798);self.assertEqual(Counter(j['defense'] for j in self.jobs),{'SC':399,'SI':399})
  self.assertEqual(self.jobs,make_jobs(self.s,self.cfg))
  old={r['task_id']:r for r in self.h['records']};pairs={p['question_id']:p for p in self.m['pairs']}
  for j in self.jobs:
   r=old[j['historical_C2_task_id']];self.assertEqual(j['messages'][:3],r['messages'][:3])
   claim,suffix=r['messages'][-1]['content'].split('\n\n',1)
   self.assertEqual(j['messages'][-1]['content'],claim+'\n\nExplanation:\n'+pairs[j['question_id']]['control_text']+'\n\n'+suffix)
  f=score_records(self.s,self.h['records'])
  self.assertEqual(f.groupby('defense').final_correct.sum().to_dict(),{'SC':220,'SI':309})
  self.assertEqual(f.groupby('defense').target_agreement.sum().to_dict(),{'SC':179,'SI':90});self.assertEqual(f.invalid.sum(),0)
 def test_config_environment_and_generation_guards(self):
  bad=copy.deepcopy(self.cfg);bad['batch_size']=8
  with self.assertRaises(RuntimeError):make_jobs(self.s,bad)
  bads=copy.deepcopy(self.s);bads['baselines'][0]['text']='CHANGED'
  with self.assertRaises(RuntimeError):validate_source(bads)
  env=copy.deepcopy(self.s['run']['environment']);env['gpu']='L4'
  with self.assertRaises(RuntimeError):validate_source_environment(env,self.s)
  obj=object.__new__(ResistanceBackend)
  for r in self.h['records'][::100]:
   with self.assertRaises(RuntimeError):obj.generate([r],1)
 def test_scoring_pairing_and_format_diagnostic(self):
  b=SyntheticBackend(self.s,self.cfg);new=[{**j,**o} for j,o in zip(self.jobs,b.generate(self.jobs,1))]
  f=score_records(self.s,self.h['records']+new);self.assertEqual(len(f),1596)
  metrics=['final_correct','target_agreement','other_wrong','invalid'];self.assertTrue(f[metrics].sum(axis=1).eq(1).all())
  before=f.copy(deep=True);diag=answer_field_diagnostic(f);self.assertTrue(f.equals(before))
  self.assertTrue(diag[metrics].sum(axis=1).eq(1).all());self.assertTrue(diag[diag.truncated].final_correct.eq(0).all());self.assertGreater(diag.diagnostic_rescored.sum(),0)
  self.assertEqual(len(comparison_table(f)),4)
  sample=new[0];i=b.items[sample['question_id']]
  def diagnostic(txt,truncated=False):
   ff=score_records(self.s,[{**sample,'text':txt,'truncated':truncated}]);return answer_field_diagnostic(ff).iloc[0]
  r=diagnostic(json.dumps({'answer':i['target_wrong'],'explanation':'test'})+'.')
  self.assertEqual(r.target_agreement,1);self.assertEqual(r.invalid,0);self.assertEqual(r.flip,1)
  r=diagnostic('{"answer":"'+i['gold']+'", "answer":"INVALID", "explanation":"test"}')
  self.assertFalse(r.diagnostic_rescored)
  r=diagnostic(json.dumps({'answer':i['gold'],'explanation':'test'})+'.',True);self.assertEqual(r.final_correct,0)
  f['final_correct']=(f.defense=='SI').astype(int);v,bounds=contrast_values(f,PRIMARY_TERMS,'final_correct',True)
  self.assertEqual(len(v),133);self.assertTrue(v.eq(1).all());self.assertEqual(bounds,(-1.,1.))
  idx=f[(f.condition=='W3')&(f.defense=='SI')].index[0]
  with self.assertRaises(RuntimeError):contrast_values(f.drop(idx),PRIMARY_TERMS,'final_correct',True)
 def test_interrupt_resume_corruption_analysis_and_export(self):
  with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
   root=Path(d)/'synthetic_run';f,jobs=freeze(root,self.s,self.cfg);b=SyntheticBackend(self.s,self.cfg);bind_backend(root,f,self.s,b,jobs)
   status=run_followups(root,f,self.s,self.cfg,jobs,b,max_new_batches=1);self.assertEqual(status['saved_responses'],16)
   first=(root/'checkpoints/batch_000000.json').read_bytes();b.fail=True
   with self.assertRaises(KeyboardInterrupt):run_followups(root,f,self.s,self.cfg,jobs,b)
   self.assertEqual(read_json(root/'status.json')['saved_responses'],16)
   with self.assertRaises(RuntimeError):load_complete(root,f,self.s,self.cfg,jobs)
   b.fail=False;status=run_followups(root,f,self.s,self.cfg,jobs,b)
   self.assertTrue(status['complete']);self.assertEqual(status['saved_responses'],798);self.assertEqual(status['saved_batches'],50)
   self.assertEqual((root/'checkpoints/batch_000000.json').read_bytes(),first);self.assertEqual(len(b.calls),50);self.assertEqual(len(b.calls[-1][1]),14)
   self.assertEqual(len({task for seed,batch in b.calls for task in batch}),798)
   run_followups(root,f,self.s,self.cfg,jobs,b);self.assertEqual(len(b.calls),50)
   folder,table,effects,assessment=analyze(root,f,self.s,self.cfg,jobs)
   self.assertEqual(len(table),4);self.assertEqual(sum(effects.analysis_role=='primary'),1);self.assertEqual(set(effects.n_items),{133});self.assertEqual(assessment['execution_mode'],'SYNTHETIC_TEST_ONLY')
   self.assertTrue((folder/'01_retention_adoption.png').exists());self.assertTrue((folder/'format_diagnostic_comparison.csv').exists())
   archive=export_results(root)
   import zipfile
   with zipfile.ZipFile(archive) as z:self.assertTrue(any(n.endswith('/reports/comparison_table.csv') for n in z.namelist()))
   p=root/'checkpoints/batch_000000.json';data=read_json(p);data['records'][0]['text']='CORRUPTED';atomic_json(p,data)
   with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)
   p.write_bytes(first);(root/'checkpoints/batch_999999.json').write_text('{}')
   with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
