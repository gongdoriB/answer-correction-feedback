"""Synthetic protocol/persistence tests; never scientific result evidence."""
import copy,sys,tempfile,unittest,json,io
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from replication_protocol import *
from replication_run import ReplicationBackend,bind_backend,run_followups,scan_saved,load_complete,validate_source_environment
from replication_analysis import score_records,comparison_table,contrast_values,answer_field_diagnostic,analyze
from prepare_inputs import extend_matching
from protocol import atomic_json

class SyntheticBackend:
    def __init__(self,source,cfg):
        self.environment=copy.deepcopy(source['run']['environment']);self.environment['synthetic_test']=True
        self.source_environment_notes={};self.items={i['question_id']:i for i in source['study']['items']}
        self.bases={r['question_id']:r for r in source['baselines']};self.calls=[];self.fail=False
    def preflight(self,jobs):return {'count':len(jobs),'max_input_tokens_observed':777,'input_lengths_hash':digest([777]*len(jobs))}
    def generate(self,jobs,seed):
        if self.fail:raise KeyboardInterrupt('SYNTHETIC interruption')
        self.calls.append((seed,[j['task_id'] for j in jobs]));out=[]
        for j in jobs:
            assert j['task_id'].startswith('REP1440:') and j['condition'] in CONDITIONS and j['defense'] in DEFENSE_NAMES
            i=self.items[j['question_id']];b,_=parse_answer(self.bases[j['question_id']]['text'],i['labels']);n=int_seed('SYNTHETIC',j['task_id'])%100
            threshold=80 if j['defense']=='SC' else {'R1':55,'R2':77,'R3':63}[j['condition']]
            a=i['gold'] if n<threshold else b;text=json.dumps({'answer':a,'explanation':'SYNTHETIC TEST ONLY'})
            if n==97:text+='.'
            if n==99:text='INVALID SYNTHETIC'
            out.append({'text':text,'truncated':n==98,'input_tokens':500,'output_tokens':30,'rendered_prompt_hash':digest(j['messages']),'seconds_per_batch':.01})
        self.last_batch_resource={'generation_seconds':.01,'scope':'SYNTHETIC'};return out
    def close(self):pass

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=load_source(HERE/'source');cls.h=load_history(cls.s);cls.m=load_matching();cls.cfg=default_config(cls.s);cls.jobs=make_jobs(cls.s,cls.cfg)
    def test_original_history_and_pairing(self):
        self.assertEqual(self.s['counts'],{'correct':125,'wrong':167,'invalid':8})
        self.assertEqual(len(self.jobs),1440);self.assertEqual(Counter((j['condition'],j['defense']) for j in self.jobs),{(c,d):240 for c in CONDITIONS for d in DEFENSE_NAMES})
        bases={r['question_id']:r for r in self.s['baselines']};history={(r['question_id'],r['condition'],r['defense'],r['variant']):r for r in self.h['records']}
        for j in self.jobs:
            self.assertEqual(j['messages'][2]['content'],bases[j['question_id']]['text'])
            key=(j['question_id'],j['condition'],j['defense'],j['variant'])
            if key in history:
                for k in [0,1,3]:self.assertEqual(j['messages'][k],history[key]['messages'][k])
        f=score_records({**self.s,'baselines':self.h['baselines']},self.h['records'],model='qwen3_8b')
        self.assertEqual(f.groupby(['condition','defense']).final_correct.sum().to_dict(),{('R1','SC'):191,('R1','SI'):125,('R2','SC'):194,('R2','SI'):179,('R3','SC'):181,('R3','SI'):147})
    def test_fixed_matching_and_token_audit(self):
        ld=source_file('excerpt_token_lengths.json');common=self.m['common8b_question_ids']
        fixed=[p for p in self.h['matching_manifest']['pairs'] if p['question_id'] in common]
        actual=extend_matching(ld['lengths'],self.m['question_ids'],fixed)
        self.assertEqual([{k:v for k,v in p.items() if k!='control_text'} for p in actual],[{k:v for k,v in p.items() if k!='control_text'} for p in self.m['pairs']])
        self.assertEqual(len(common),61);self.assertEqual(self.m['summary']['within_5_percent'],76)
        audit=source_file('token_audit.json');self.assertEqual(audit['jobs_hash'],digest(self.jobs));self.assertEqual(audit['count'],1440);self.assertLessEqual(audit['max_input_tokens_observed'],4096)
    def test_no_changed_config_or_sources(self):
        cfg=copy.deepcopy(self.cfg);cfg['batch_size']=8
        with self.assertRaises(RuntimeError):make_jobs(self.s,cfg)
        s=copy.deepcopy(self.s);s['baselines'][0]['text']='changed'
        with self.assertRaises(RuntimeError):validate_source(s)
        e=copy.deepcopy(self.s['run']['environment']);e['gpu']='L4'
        with self.assertRaises(RuntimeError):validate_source_environment(e,self.s)
        obj=ReplicationBackend.__new__(ReplicationBackend)
        with self.assertRaises(RuntimeError):obj.generate([self.s['baselines'][0]],1)
    def test_scoring_and_paired_units(self):
        b=SyntheticBackend(self.s,self.cfg);new=[{**j,**r} for j,r in zip(self.jobs,b.generate(self.jobs,1))];f=score_records(self.s,new)
        self.assertTrue(f[['final_correct','initial_wrong_retained','other_wrong_change','invalid']].sum(axis=1).eq(1).all())
        self.assertEqual(len(comparison_table(f)),6)
        strict=f.final_correct.copy();diag=answer_field_diagnostic(f);self.assertTrue(f.final_correct.equals(strict));self.assertTrue(diag[diag.truncated].final_correct.eq(0).all())
        self.assertGreater(diag.diagnostic_rescored.sum(),0)
        f['final_correct']=(f.defense=='SI').astype(int);vals,_=contrast_values(f,PRIMARY_TERMS,'final_correct',False)
        self.assertEqual(len(vals),80);self.assertTrue(vals.eq(0).all())
        idx=f[(f.condition=='R3')&(f.defense=='SI')].index[0]
        with self.assertRaises(RuntimeError):contrast_values(f.drop(idx),PRIMARY_TERMS,'final_correct',False)
    def test_interruption_resume_complete_analysis_and_corruption(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            root=Path(d);f,jobs=freeze(root,self.s,self.cfg);b=SyntheticBackend(self.s,self.cfg);bind_backend(root,f,self.s,b,jobs)
            status=run_followups(root,f,self.s,self.cfg,jobs,b,max_new_batches=1);self.assertEqual(status['saved_responses'],16)
            first=(root/'checkpoints/batch_000000.json').read_bytes();b.fail=True
            with self.assertRaises(KeyboardInterrupt):run_followups(root,f,self.s,self.cfg,jobs,b)
            with self.assertRaises(RuntimeError):load_complete(root,f,self.s,self.cfg,jobs)
            self.assertEqual(read_json(root/'status.json')['saved_responses'],16);b.fail=False
            status=run_followups(root,f,self.s,self.cfg,jobs,b);self.assertEqual(status['saved_responses'],1440);self.assertEqual(status['saved_batches'],90)
            self.assertEqual((root/'checkpoints/batch_000000.json').read_bytes(),first)
            self.assertEqual(len(b.calls),90);self.assertEqual(len({t for _,batch in b.calls for t in batch}),1440)
            run_followups(root,f,self.s,self.cfg,jobs,b);self.assertEqual(len(b.calls),90)
            folder,table,effects,assessment=analyze(root,f,self.s,self.cfg,jobs)
            self.assertEqual(assessment['execution_mode'],'SYNTHETIC_TEST_ONLY');self.assertEqual(sum(effects.analysis_role=='primary'),1);self.assertEqual(set(effects.n_items),{80})
            import pandas as pd
            self.assertEqual(set(pd.read_csv(folder/'common61_comparison.csv').n_items),{61})
            self.assertEqual(set(pd.read_csv(folder/'length_sensitivity_effects.csv').n_items),{76})
            self.assertEqual(len(pd.read_csv(folder/'historical_4b_S0_S1_reference.csv')),4)
            p=root/'checkpoints/batch_000000.json';data=read_json(p);data['records'][0]['text']='CORRUPTED';atomic_json(p,data)
            with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)
            p.write_bytes(first);(root/'checkpoints/batch_999999.json').write_text('{}')
            with self.assertRaises(RuntimeError):scan_saved(root,jobs,self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
