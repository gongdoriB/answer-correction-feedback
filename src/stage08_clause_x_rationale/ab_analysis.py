"""Paired question-level evidence-by-defense interaction; no new judge model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis import bootstrap_interval, revision_outcome, summarize
from protocol import atomic_json, int_seed, parse_answer, read_json
from ab_protocol import MODEL, DEFENSE_NAMES, DEFENSE_LABELS, selected_controls, load_manifest, load_history, ANALYSIS_PLAN
from ab_run import load_complete

def score_records(source, records):
    items = {i["question_id"]: i for i in source["study"]["items"]}
    bases = {r["question_id"]: r for r in source["baselines"]}
    rows = []
    for record in records:
        item, base = items[record["question_id"]], bases[record["question_id"]]
        b, bs = parse_answer(base["text"], item["labels"])
        a, status = parse_answer(record["text"], item["labels"])
        if base["truncated"]:
            b, bs = None, "truncated"
        if record["truncated"]:
            a, status = None, "truncated"
        outcome = revision_outcome(b, bs, a, status, item["gold"])
        rows.append({"model": MODEL, "question_id": item["question_id"], "subject": item["subject"],
                     "data_origin": ("new_R2" if record["task_id"].startswith("AB693:") else "historical_R1" if record["condition"] == "R1" else "historical_R2"),
                     "task_id": record["task_id"],
                     "condition": record["condition"], "defense": record["defense"], "variant": record["variant"],
                     "baseline_answer": b, "baseline_valid": bs == "valid", "baseline_correct": b == item["gold"],
                     "gold": item["gold"], "target_wrong": item["target_wrong"], "final_answer": a,
                     "final_correct": int(a == item["gold"]), "invalid": int(status != "valid"),
                     "flip": int(status == "valid" and a != item["gold"]),
                     "target_agreement": int(a == item["target_wrong"]),
                     "initial_wrong_retained": int(outcome == "retained_initial_wrong"),
                     "other_wrong_change": int(outcome == "changed_to_other_wrong"),
                     "revision_outcome": outcome, "input_tokens": record["input_tokens"],
                     "output_tokens": record["output_tokens"], "truncated": record["truncated"],
                     "baseline_hash": record["baseline_hash"], "final_text": record["text"]})
    return pd.DataFrame(rows)


def contrast_values(frame, terms, metric, correct):
    """Return one paired contrast per question; terms are (condition, defense, weight)."""
    subset = frame
    if correct is not None:
        subset = subset[subset.baseline_valid & (subset.baseline_correct == correct)]
    parts, weights = [], []
    for condition, defense, weight in terms:
        f = subset[(subset.condition == condition) & (subset.defense == defense)]
        if f.empty or f.duplicated(["question_id", "variant"]).any():
            raise RuntimeError("비교 문항이 없거나 표현이 중복됐습니다.")
        variant_sets = f.groupby("question_id").variant.agg(lambda v: frozenset(v))
        if not variant_sets.map(lambda v: v == frozenset([0, 1, 2])).all():
            raise RuntimeError("각 문항의 고정 표현 3개가 모두 있어야 합니다.")
        part = f.groupby(["question_id", "subject"])[metric].mean()
        if parts and not part.index.equals(parts[0].index):
            raise RuntimeError("조건 간 문항 대응이 달라졌습니다.")
        parts.append(part)
        weights.append(float(weight))
    values = sum(w * part for w, part in zip(weights, parts))
    bounds = (sum(w for w in weights if w < 0), sum(w for w in weights if w > 0))
    return values, bounds


def estimate_contrast(frame, terms, metric, correct, cfg, name, role, label):
    values, bounds = contrast_values(frame, terms, metric, correct)
    alpha = .05 / cfg["primary_family_size"] if role == "primary" else .05
    repetitions = cfg["primary_bootstrap_repetitions"] if role == "primary" else cfg["bootstrap_repetitions"]
    lo, hi, method = bootstrap_interval(values, values.index.get_level_values("subject"), repetitions,
                                        alpha=alpha, bounds=bounds, seed=int_seed(cfg["seed"], name))
    row = {"contrast": name, "label_ko": label, "analysis_role": role, "metric": metric,
           "n_items": len(values), "stratum": "all_items" if correct is None else "initial_correct" if correct else "initial_wrong",
           "difference_pp": float(values.mean()) * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
           "interval_level": 1 - alpha, "ci_method": method, "bootstrap_repetitions": repetitions,
           "terms": json.dumps(terms), "inference_scope": "same-item exploratory; interval is not evidence of equivalence"}
    detail = values.rename("difference").reset_index()
    detail["contrast"] = name
    detail["difference_pp"] = detail.difference * 100
    return row, detail


def effect_table(frame,cfg):
    specs=[(p['name'],'primary',label,p['terms']) for p,label in zip(ANALYSIS_PLAN['primary_contrasts'],[
      '풀이 추가 후 억제 절의 수정 감소가 완화되는가', '풀이·억제 절이 있을 때 수정 절을 더하면 수정률이 높아지는가'])]
    for c in ['R1','R2']:
        for suffix,label,terms in [
          ('inhibition_only','억제 절 효과 (SI−SC)',[(c,'SI',1),(c,'SC',-1)]),
          ('correction_only','수정 절 효과 (SR−SC)',[(c,'SR',1),(c,'SC',-1)]),
          ('inhibition_with_correction','수정 절이 있을 때 억제 절 효과 (S1−SR)',[(c,'S1',1),(c,'SR',-1)]),
          ('correction_with_inhibition','억제 절이 있을 때 수정 절 효과 (S1−SI)',[(c,'S1',1),(c,'SI',-1)]),
          ('clause_interaction','두 절 결합의 상호작용 (S1−SI−SR+SC)',[(c,'S1',1),(c,'SI',-1),(c,'SR',-1),(c,'SC',1)]),
          ('full_vs_none','전체 방어 − 방어 없음',[(c,'S1',1),(c,'S0',-1)])]:
            if c=='R2' and suffix=='correction_with_inhibition':continue
            specs.append((c+'_'+suffix,'descriptive',c+': '+label,terms))
    specs.append(('three_way_interaction','descriptive','풀이 추가에 따른 두 절 결합 상호작용 변화',
      [('R2','S1',1),('R2','SI',-1),('R2','SR',-1),('R2','SC',1),('R1','S1',-1),('R1','SI',1),('R1','SR',1),('R1','SC',-1)]))
    rows=[];details=[]
    for name,role,label,terms in specs:
        row,detail=estimate_contrast(frame,terms,'final_correct',False,cfg,name,role,label)
        rows.append(row);details.append(detail)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def comparison_table(frame):
    rows=[]
    for c in ['R1','R2']:
        for arm in DEFENSE_NAMES:
            f=frame[(frame.condition==c)&(frame.defense==arm)]
            if len(f)!=231 or f.question_id.nunique()!=77:
                raise RuntimeError('공통 77문항 × 3표현 대응 자료 누락')
            row={'condition':c,'defense':arm,'defense_ko':DEFENSE_LABELS[arm],
                 'data_origin':f.data_origin.iloc[0],'n_items':77,'n_responses':len(f),'correct_count':int(f.final_correct.sum())}
            for metric in ['final_correct','initial_wrong_retained','other_wrong_change','invalid','truncated']:
                row[metric+'_percent']=100*float(f[metric].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def figures(folder,table,effects,synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    x=np.arange(5);fig,ax=plt.subplots(figsize=(10,5.8))
    for c,offset,color in [('R1',-.19,'#7895AD'),('R2',.19,'#168078')]:
        t=table[table.condition==c].set_index('defense').loc[DEFENSE_NAMES]
        bars=ax.bar(x+offset,t.final_correct_percent,.36,label='R1: correct option only' if c=='R1' else 'R2: same option + excerpt',color=color)
        for b,(_,r) in zip(bars,t.iterrows()):
            if r.data_origin=='new_R2':b.set_hatch('//');b.set_edgecolor('#214549')
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.1,f'{r.final_correct_percent:.1f}',ha='center',fontsize=9)
    ax.set_xticks(x,['S0\nNo defense','SC\nCommon only','SI\nInhibition','SR\nCorrection','S1\nBoth clauses']);ax.set_ylim(0,108)
    ax.set_ylabel('Corrected answers (%)');ax.legend(loc='upper left',ncol=2,frameon=False)
    ax.set_title('Same 77 initially wrong items, three fixed phrasings per cell',pad=16)
    fig.text(.02,.015,'Hatched bars: 693 new responses. All other bars are reused. Invalid/truncated responses count as failures.',fontsize=9)
    if synthetic:fig.text(.5,.5,'SYNTHETIC TEST ONLY',ha='center',fontsize=28,color='#993333',alpha=.25,rotation=20)
    fig.tight_layout(rect=[0,.045,1,1]);fig.savefig(folder/'01_correction_comparison.png',dpi=180);fig.savefig(folder/'01_correction_comparison.pdf');plt.close(fig)
    primary=effects[effects.analysis_role=='primary'].reset_index(drop=True);fig,ax=plt.subplots(figsize=(10,4.6))
    for i,r in primary.iterrows():
        ax.plot([r.ci_low_pp,r.ci_high_pp],[i,i],color='#168078',lw=3);ax.plot(r.difference_pp,i,'o',color='#153E48')
        ax.annotate(f'{r.difference_pp:+.1f} [{r.ci_low_pp:+.1f}, {r.ci_high_pp:+.1f}]',(r.difference_pp,i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
    ax.axvline(0,color='#777',ls='--',lw=1);ax.set_yticks([0,1],['Attenuation of inhibition cost\n(R2 SI-SC) - (R1 SI-SC)','Correction clause under evidence\nR2 S1-SI']);ax.invert_yaxis();ax.set_ylim(1.6,-.6)
    ax.set_xlabel('Difference in percentage points');ax.set_title('Two primary contrasts: paired question bootstrap, 97.5% intervals')
    if synthetic:fig.text(.5,.5,'SYNTHETIC TEST ONLY',ha='center',fontsize=28,color='#993333',alpha=.25,rotation=20)
    fig.tight_layout();fig.savefig(folder/'02_primary_contrasts.png',dpi=180);fig.savefig(folder/'02_primary_contrasts.pdf');plt.close(fig)


def assess(effects):
    messages=[]
    for r in effects[effects.analysis_role=='primary'].itertuples():
        direction='증가 방향이 구간에서도 관찰됨' if r.ci_low_pp>0 else '감소 방향이 구간에서도 관찰됨' if r.ci_high_pp<0 else '구간이 0을 포함하여 방향을 확정하기 어려움'
        messages.append(f'{r.label_ko}: {r.difference_pp:+.2f}%p, 97.5% 구간 [{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]; {direction}.')
    messages.append('첫 비교는 R1의 SI−SC가 음수일 때 억제 비용 완화로 읽습니다. 두 번째 비교가 불명확하다고 수정 절이 불필요하다고 결론 내리지는 않습니다.')
    return {'messages_ko':messages,'scope':'동일 77문항에서 수행한 탐색적 교차 분석. 효과 없음·동등성·실서비스 안전성의 입증이 아닙니다.','limits':ANALYSIS_PLAN['limits']}


def analyze(root,frozen,source,cfg,jobs):
    root=Path(root);new,resources=load_complete(root,frozen,source,cfg,jobs)
    if len(new)!=693:raise RuntimeError('693개 신규 응답 완료 후 분석하세요.')
    history=load_history(source);frame=score_records(source,history['records']+new)
    if len(frame)!=2310 or frame.duplicated(['question_id','condition','defense','variant']).any():raise RuntimeError('2310개 대응 자료 누락/중복')
    if not frame.baseline_valid.all() or frame.baseline_correct.any() or set(frame.question_id)!=set(history['question_ids']):raise RuntimeError('공통 초기 오답 77문항 불일치')
    exhaustive=frame[['final_correct','initial_wrong_retained','other_wrong_change','invalid']].sum(axis=1)
    if not exhaustive.eq(1).all():raise RuntimeError('채점 결과 분류 오류')
    folder=root/'reports';folder.mkdir(exist_ok=True)
    frame.to_csv(folder/'scored_responses.csv',index=False,encoding='utf-8-sig')
    table=comparison_table(frame);table.to_csv(folder/'comparison_table.csv',index=False,encoding='utf-8-sig')
    table.pivot(index='defense',columns='condition',values='final_correct_percent').reindex(DEFENSE_NAMES).to_csv(folder/'correction_matrix.csv',encoding='utf-8-sig')
    print('기존 1,617 + 신규 693 = 2,310개 자동 채점 완료. 문항별 대응 구간 계산 중…',flush=True)
    effects,details=effect_table(frame,cfg)
    effects.to_csv(folder/'clause_evidence_effects.csv',index=False,encoding='utf-8-sig')
    details.to_csv(folder/'item_contrasts.csv',index=False,encoding='utf-8-sig')
    tokens=frame.groupby(['condition','defense','data_origin']).agg(responses=('final_correct','size'),input_tokens_mean=('input_tokens','mean'),output_tokens_mean=('output_tokens','mean'),invalid_rate=('invalid','mean'),truncated_rate=('truncated','mean')).reset_index()
    tokens.to_csv(folder/'tokens_and_format.csv',index=False,encoding='utf-8-sig')
    ex={r['question_id']:r for r in load_manifest()['excerpts']};pd.DataFrame([ex[q] for q in history['question_ids']]).to_json(folder/'used_excerpts.json',orient='records',force_ascii=False,indent=2)
    assessment=assess(effects)
    synthetic=read_json(root/'environment_lock.json').get('synthetic_test',False)
    assessment['execution_mode']='SYNTHETIC_TEST_ONLY' if synthetic else 'MODEL_INFERENCE'
    atomic_json(folder/'assessment.json',assessment)
    atomic_json(folder/'batch_resources.json',resources)
    figures(folder,table,effects,synthetic=synthetic)
    lines=['A·B 교차 추가실험 — 공통 77문항',assessment['execution_mode'],'기존 최초 답변 재사용 / 새 R2 SC·SI·SR 693개만 생성','',*assessment['messages_ko'],'',assessment['scope'],*ANALYSIS_PLAN['limits']]
    (folder/'RESULTS_KO.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('비교표·대응 구간·그래프 저장 완료:',folder,flush=True)
    return folder,table,effects,assessment
