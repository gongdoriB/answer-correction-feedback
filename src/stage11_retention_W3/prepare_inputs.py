"""Build immutable input provenance; never run archive code or examine new outcomes."""
import hashlib
import zipfile
from pathlib import Path
from collections import Counter
from protocol import atomic_json, read_json, digest, int_seed, parse_answer
import ablation_protocol as A

HERE = Path(__file__).resolve().parent
ARCHIVE_NAME = 'sycophancy_qwen3_8b_ablation10800_a100_b16_v1_results.zip'
ARCHIVE_SHA = 'd30a94aec311003614ce25f66712b68b275108140ab88ef80ebdd8d20d9d667e'
MATCH_SEED = 'resistance798_cross_subject_v1_20261101'
POLICY = {
    'recipients': 'All 133 valid initially correct Qwen3-8B answers. No own-excerpt requirement.',
    'donors': 'All 155 previously frozen B max5 excerpts; no selection by any follow-up outcome.',
    'assignment': 'Deterministic one-to-one bipartite matching, cross-subject only, hash-ordered recipients and edges.',
    'editing': 'Reuse the complete previously selected excerpt unchanged; no new writing, trimming or label substitution.',
    'replication': 'One donor per recipient, shared across SC/SI and all three phrasings.',
    'inputs_used_for_assignment': ['question_id', 'subject'],
    'length': 'NOT length-matched. C2 has no explanation; W3 adds text. No content-versus-length causal claim.',
    'meaning': 'Other-subject solution, not certified irrelevant, false or plausible support for the wrong option.'}

def select_pairs(targets, pool):
    """No gold labels, initial/follow-up answers or outcome scores enter this function."""
    targets = sorted(targets, key=lambda a: (int_seed(MATCH_SEED, 'target', a['question_id']), a['question_id']))
    byid = {p['question_id']: p for p in pool}
    adjacency = {a['question_id']: sorted(
        [b['question_id'] for b in pool if a['question_id'] != b['question_id'] and a['subject'] != b['subject']],
        key=lambda d: (int_seed(MATCH_SEED, a['question_id'], d), d)) for a in targets}
    donor_owner = {}
    def augment(q, seen):
        for d in adjacency[q]:
            if d in seen: continue
            seen.add(d)
            if d not in donor_owner or augment(donor_owner[d], seen):
                donor_owner[d] = q
                return True
        return False
    for a in targets:
        if not augment(a['question_id'], set()): raise RuntimeError('Cross-subject matching incomplete')
    assigned = {q:d for d,q in donor_owner.items()}
    return [{'question_id':a['question_id'], 'subject':a['subject'],
             'donor_question_id':assigned[a['question_id']], 'donor_subject':byid[assigned[a['question_id']]]['subject']}
            for a in sorted(targets, key=lambda a:a['question_id'])]

def main(archive):
    import json
    source = A.load_source(HERE/'source')
    cfg = A.default_config(source); jobs = A.make_jobs(source,cfg)
    items = {i['question_id']:i for i in source['study']['items']}
    bases = {r['question_id']:r for r in source['baselines']}
    qids = sorted(q for q,i in items.items() if not bases[q]['truncated'] and parse_answer(bases[q]['text'],i['labels']) == (i['gold'],'valid'))
    assert len(qids) == 133
    archive = Path(archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
    locks = {}; records = []; selected_checkpoints = []
    with zipfile.ZipFile(archive) as z:
        prefix = ARCHIVE_NAME.removesuffix('_results.zip') + '/'
        for n in ['study_lock.json','environment_lock.json','run_lock.json','schedule.json','status.json']:
            locks[n] = json.loads(z.read(prefix+n))
        assert locks['study_lock.json']['config'] == cfg
        assert locks['study_lock.json']['jobs_hash'] == digest(jobs)
        assert locks['environment_lock.json'] == source['run']['environment']
        assert locks['schedule.json']['task_ids'] == [j['task_id'] for j in jobs]
        checkpoint_names = sorted(n for n in z.namelist() if '/checkpoints/batch_' in n and n.endswith('.json'))
        assert len(checkpoint_names) == 675
        for index,name in enumerate(checkpoint_names):
            raw = z.read(name); b = json.loads(raw); batch = jobs[index*16:(index+1)*16]
            seed = int_seed(cfg['seed'],'ablation_followup',index,[j['task_id'] for j in batch])
            assert b['batch_index']==index and b['batch_seed']==seed and b['batch_hash']==digest(batch)
            assert b['records_hash']==digest(b['records']) and len(b['records'])==len(batch)
            assert b['environment_hash']==digest(locks['environment_lock.json'])
            selected = []
            for row,job in zip(b['records'],batch):
                assert all(row.get(k)==v for k,v in job.items())
                assert row['batch_index']==index and row['batch_seed']==seed
                if row['question_id'] in qids and row['condition']=='C2' and row['defense'] in ['SC','SI']:
                    records.append(row); selected.append(row['task_id'])
            if selected: selected_checkpoints.append({'member':name,'sha256':hashlib.sha256(raw).hexdigest(),'records_hash':b['records_hash'],'selected_task_ids':selected})
    assert len(records)==798 and Counter(r['defense'] for r in records)=={'SC':399,'SI':399}
    history = {'question_ids':qids,'records':records,'selection':'Initial correct only; C2 SC/SI all three fixed variants, no follow-up outcome selection.',
               'provenance':{'A_archive_name':ARCHIVE_NAME,'A_archive_sha256':ARCHIVE_SHA,'all_675_checkpoints_verified':True,
                             'A_locks':locks,'selected_checkpoint_provenance':selected_checkpoints}}
    history['history_hash']=digest(history);atomic_json(HERE/'source/history.json',history)
    m=read_json(HERE/'source/evidence_manifest.json');ld=read_json(HERE/'source/excerpt_token_lengths.json')
    ex={e['question_id']:e for e in m['excerpts']};lengths={r['question_id']:r for r in ld['lengths']}
    pairs=select_pairs([{'question_id':q,'subject':items[q]['subject']} for q in qids],
                       [{'question_id':q,'subject':e['subject']} for q,e in ex.items()])
    for p in pairs:
        d=p['donor_question_id'];e=ex[d]
        assert 1<=e['sentence_count_auto']<=5 and e['normalized_words']<=180
        p.update(control_text=e['text'],control_excerpt_hash=digest(e['text']),control_excerpt_tokens=lengths[d]['tokens'],
                 sentence_count_auto=e['sentence_count_auto'],word_count=e['normalized_words'])
    matching={'policy':POLICY,'seed':MATCH_SEED,'source_B_manifest_hash':m['manifest_hash'],'lengths_hash':digest(ld),
              'question_ids':qids,'pairs':pairs,'summary':{'recipients':133,'unique_donors':133,'donor_pool':155,
              'recipient_subjects':dict(Counter(p['subject'] for p in pairs)),
              'donor_subjects':dict(Counter(p['donor_subject'] for p in pairs)),
              'donor_sentences':dict(Counter(str(p['sentence_count_auto']) for p in pairs)),
              'min_tokens':min(p['control_excerpt_tokens'] for p in pairs),'max_tokens':max(p['control_excerpt_tokens'] for p in pairs)}}
    matching['matching_hash']=digest(matching);atomic_json(HERE/'source/matching_manifest.json',matching)
    print(json.dumps({'history_hash':history['history_hash'],'matching_hash':matching['matching_hash'],'summary':matching['summary']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('archive');main(parser.parse_args().archive)
