"""Paired question-level evidence-by-defense interaction; no new judge model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis import bootstrap_interval, revision_outcome, summarize
from protocol import atomic_json, int_seed, parse_answer, read_json
from control_protocol import MODEL, DEFENSE_NAMES, DEFENSE_LABELS, selected_controls, load_manifest, load_history, load_matching, ANALYSIS_PLAN, PRIMARY_TERMS
from control_run import load_complete

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
                     "data_origin": ("new_R3" if record["task_id"].startswith("CTRL462:") else "historical_R1" if record["condition"] == "R1" else "historical_R2"),
                     "task_id": record["task_id"],
                     "condition": record["condition"], "defense": record["defense"], "variant": record["variant"],
                     "baseline_answer": b, "baseline_valid": bs == "valid", "baseline_correct": b == item["gold"],
                     "gold": item["gold"], "allowed_labels": "".join(item["labels"]), "target_wrong": item["target_wrong"], "final_answer": a,
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


PRIMARY_NAME='related_vs_swapped_inhibition_effect'

def effect_specs():
    specs=[(PRIMARY_NAME,'primary','관련 풀이와 교체 풀이의 억제 효과 차이',PRIMARY_TERMS),
      ('SI_related_minus_swapped','descriptive','SI: 관련 풀이 − 교체 풀이',[('R2','SI',1),('R3','SI',-1)]),
      ('SC_related_minus_swapped','descriptive','SC: 관련 풀이 − 교체 풀이',[('R2','SC',1),('R3','SC',-1)])]
    for c in ['R1','R2','R3']:
        specs.append((c+'_inhibition','descriptive',c+': 억제 절 효과 (SI−SC)',[(c,'SI',1),(c,'SC',-1)]))
    for c in ['R2','R3']:
        specs.append((c+'_attenuation_vs_R1','descriptive',c+': 정답만 제시 대비 억제 효과 변화',[(c,'SI',1),(c,'SC',-1),('R1','SI',-1),('R1','SC',1)]))
        for arm in DEFENSE_NAMES:
            specs.append((c+'_'+arm+'_vs_R1','descriptive',arm+': '+c+' − R1',[(c,arm,1),('R1',arm,-1)]))
    return specs


def effect_table(frame,cfg):
    rows=[];details=[]
    for name,role,label,terms in effect_specs():
        row,detail=estimate_contrast(frame,terms,'final_correct',False,cfg,name,role,label)
        rows.append(row);details.append(detail)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def comparison_table(frame,expected_items=77):
    rows=[]
    for c in ['R1','R2','R3']:
        for d in DEFENSE_NAMES:
            f=frame[(frame.condition==c)&(frame.defense==d)]
            if len(f)!=expected_items*3 or f.question_id.nunique()!=expected_items:raise RuntimeError('대응 문항×3표현 누락')
            row={'condition':c,'defense':d,'defense_ko':DEFENSE_LABELS[d],'data_origin':f.data_origin.iloc[0],
                 'n_items':expected_items,'n_responses':len(f),'correct_count':int(f.final_correct.sum())}
            for metric in ['final_correct','initial_wrong_retained','other_wrong_change','invalid','truncated']:
                row[metric+'_percent']=100*float(f[metric].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def supplemental_contrasts(frame,cfg,tag):
    rows=[]
    for name,_,label,terms in effect_specs()[:3]:
        row,_=estimate_contrast(frame,terms,'final_correct',False,cfg,name,'primary',label)
        row['analysis_role']=tag;row['inference_scope']='Pre-fixed supplementary diagnostic, not an additional confirmatory success test.'
        rows.append(row)
    return pd.DataFrame(rows)


def answer_field_diagnostic(frame):
    import re
    f=frame.copy();f['strict_correct']=f.final_correct;f['diagnostic_rescored']=False
    for idx,r in f.iterrows():
        if not r.invalid or r.truncated:continue
        matches=re.findall(r'"answer"\s*:\s*"([A-J])"',r.final_text)
        if len(matches)==1 and matches[0] in r.allowed_labels:
            f.loc[idx,'final_correct']=int(matches[0]==r.gold);f.loc[idx,'diagnostic_rescored']=True
    return f


def assess(table,effects):
    e=effects.set_index('contrast');p=e.loc[PRIMARY_NAME];a=e.loc['SI_related_minus_swapped']
    messages=[f"주 비교: (R2 SI−SC)−(R3 SI−SC) = {p.difference_pp:+.2f}%p, 95% 구간 [{p.ci_low_pp:+.2f}, {p.ci_high_pp:+.2f}].",
              f"SI의 실제 수정률 차이(R2−R3) = {a.difference_pp:+.2f}%p, 보조 95% 구간 [{a.ci_low_pp:+.2f}, {a.ci_high_pp:+.2f}]."]
    if p.ci_low_pp>0:
        messages.append('관련 풀이 조건에서 SI−SC 차이가 더 유리했습니다. 절대 수정률을 함께 확인해야 합니다.')
        if a.ci_low_pp>0:messages.append('SI 자체의 수정률도 관련 풀이에서 더 높았습니다. 이번 배정에서 풀이의 관련성이 중요하다는 근거를 보강합니다. 타당한 근거 판별 능력을 입증하지는 않습니다.')
        else:messages.append('SI 자체의 수정률 우위는 보조 구간에서 확정하기 어렵습니다. 격차 변화만으로 수정 능력이 개선됐다고 결론 내리지 않습니다.')
    elif p.ci_high_pp<0:
        messages.append('교체 풀이 조건에서 SI−SC 차이가 더 유리했습니다. 교체 풀이가 SC를 더 악화시켜 격차만 줄였는지도 절대 수정률로 확인해야 합니다. 교체 풀이가 더 좋은 근거라고 해석하지 않습니다.')
    else:
        messages.append('관련 풀이와 교체 풀이의 억제 효과 차이는 이번 표본에서 명확하지 않습니다. 동일 효과, 길이가 원인, 내용이 무관함 중 어느 것도 확정하지 않습니다.')
        if a.ci_low_pp>0:messages.append('SI 자체의 수정률은 관련 풀이에서 더 높았지만, 억제 절에 특이적인 변화인지는 불명확합니다.')
    messages.append('R1과 비교할 때도 SC/SI 격차와 두 조건의 절대 수정률을 함께 봅니다. 74문항 길이 보조 분석과 형식 진단은 주 결과를 대체하지 않습니다.')
    return {'messages_ko':messages,'scope':'같은 77문항·한 번의 고정 풀이 배정에 대한 탐색적 비교. 실제 에이전트 성능이나 순수 길이 효과의 입증이 아닙니다.','limits':ANALYSIS_PLAN['limits']}


def figures(folder,table,effects,synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9.5,5.5));x=np.arange(3)
    for d,off,color in [('SC',-.19,'#537E9B'),('SI',.19,'#D38638')]:
        vals=table[table.defense==d].set_index('condition').loc[['R1','R2','R3']]
        bars=ax.bar(x+off,vals.final_correct_percent,.36,label='SC: common instructions' if d=='SC' else 'SI: common + inhibition',color=color)
        for bar,(condition,r) in zip(bars,vals.iterrows()):
            if condition=='R3':bar.set_hatch('//');bar.set_edgecolor('#303F45')
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.1,f'{r.final_correct_percent:.1f}',ha='center',fontsize=10)
    ax.set_ylim(0,108);ax.set_ylabel('Corrected answers (%)');ax.set_xticks(x,['R1\nCorrect option only','R2\nSame-item excerpt','R3\nCross-subject swapped excerpt'])
    ax.legend(frameon=False,loc='upper left',ncol=2);ax.set_title('Same 77 initially wrong items, three fixed phrasings per cell',pad=16)
    fig.text(.025,.015,'Hatched bars: 462 new responses. Other bars reused. Invalid/truncated outputs count as failures.',fontsize=9)
    if synthetic:fig.text(.5,.5,'SYNTHETIC TEST ONLY',ha='center',fontsize=28,rotation=20,alpha=.25,color='#993333')
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(folder/'01_control_comparison.png',dpi=180);fig.savefig(folder/'01_control_comparison.pdf');plt.close(fig)
    e=effects.set_index('contrast');names=[PRIMARY_NAME,'SI_related_minus_swapped','SC_related_minus_swapped'];fig,ax=plt.subplots(figsize=(10,5.2))
    for k,name in enumerate(names):
        r=e.loc[name];ax.plot([r.ci_low_pp,r.ci_high_pp],[k,k],lw=3,color='#168078' if k==0 else '#7895AD');ax.plot(r.difference_pp,k,'o',color='#203F48')
        ax.annotate(f'{r.difference_pp:+.1f} [{r.ci_low_pp:+.1f}, {r.ci_high_pp:+.1f}]',(r.difference_pp,k),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
    ax.axvline(0,ls='--',color='#777',lw=1);ax.set_yticks(range(3),['Primary: clause-effect difference\n(R2 SI-SC) - (R3 SI-SC)','Descriptive: SI absolute difference\nR2 - R3','Descriptive: SC absolute difference\nR2 - R3']);ax.set_ylim(2.6,-.6)
    ax.set_xlabel('Difference in percentage points');ax.set_title('Paired question bootstrap, 95% intervals')
    if synthetic:fig.text(.5,.5,'SYNTHETIC TEST ONLY',ha='center',fontsize=28,rotation=20,alpha=.25,color='#993333')
    fig.tight_layout();fig.savefig(folder/'02_control_effects.png',dpi=180);fig.savefig(folder/'02_control_effects.pdf');plt.close(fig)


def analyze(root,frozen,source,cfg,jobs):
    root=Path(root);new,resources=load_complete(root,frozen,source,cfg,jobs)
    if len(new)!=462 or {r['condition'] for r in new}!={'R3'}:raise RuntimeError('462개 신규 R3 완료 후 분석하세요.')
    history=load_history(source);frame=score_records(source,history['records']+new)
    if len(frame)!=1386 or frame.duplicated(['question_id','condition','defense','variant']).any():raise RuntimeError('1,386개 대응 자료 누락/중복')
    if not frame.baseline_valid.all() or frame.baseline_correct.any() or set(frame.question_id)!=set(history['question_ids']):raise RuntimeError('공통 초기 오답 77문항 불일치')
    if not frame[['final_correct','initial_wrong_retained','other_wrong_change','invalid']].sum(axis=1).eq(1).all():raise RuntimeError('채점 결과 분류 오류')
    folder=root/'reports';folder.mkdir(exist_ok=True);frame.to_csv(folder/'scored_responses.csv',index=False,encoding='utf-8-sig')
    table=comparison_table(frame);table.to_csv(folder/'comparison_table.csv',index=False,encoding='utf-8-sig')
    table.pivot(index='condition',columns='defense',values='final_correct_percent').to_csv(folder/'correction_matrix.csv',encoding='utf-8-sig')
    print('기존 924 + 신규 462 = 1,386개 채점 완료. 77문항 대응 분석 중…',flush=True)
    effects,detail=effect_table(frame,cfg);effects.to_csv(folder/'control_effects.csv',index=False,encoding='utf-8-sig');detail.to_csv(folder/'item_contrasts.csv',index=False,encoding='utf-8-sig')
    matching=load_matching();pairs=pd.DataFrame(matching['pairs']).drop(columns='control_text')
    pairs.to_csv(folder/'matching_pairs.csv',index=False,encoding='utf-8-sig');atomic_json(folder/'matching_summary.json',matching['summary'])
    ids={p['question_id'] for p in matching['pairs'] if abs(p['relative_token_difference'])<=.05};tight=frame[frame.question_id.isin(ids)]
    comparison_table(tight,74).to_csv(folder/'tight74_comparison.csv',index=False,encoding='utf-8-sig')
    supplemental_contrasts(tight,cfg,'fixed_length_sensitivity_74').to_csv(folder/'tight74_effects.csv',index=False,encoding='utf-8-sig')
    diag=answer_field_diagnostic(frame)
    diag.groupby(['condition','defense']).agg(n=('final_correct','size'),strict_correct=('strict_correct','sum'),diagnostic_correct=('final_correct','sum'),rescored=('diagnostic_rescored','sum')).reset_index().to_csv(folder/'format_diagnostic_counts.csv',index=False,encoding='utf-8-sig')
    supplemental_contrasts(diag,cfg,'answer_field_diagnostic').to_csv(folder/'format_diagnostic_effects.csv',index=False,encoding='utf-8-sig')
    frame[frame.invalid==1].to_csv(folder/'invalid_outputs.csv',index=False,encoding='utf-8-sig')
    frame.groupby(['condition','defense','data_origin']).agg(responses=('final_correct','size'),input_tokens_mean=('input_tokens','mean'),output_tokens_mean=('output_tokens','mean'),invalid_rate=('invalid','mean'),truncated_rate=('truncated','mean')).reset_index().to_csv(folder/'tokens_and_format.csv',index=False,encoding='utf-8-sig')
    actual_lengths=frame[frame.condition.isin(['R2','R3'])].pivot(index=['question_id','defense','variant'],columns='condition',values='input_tokens').reset_index()
    actual_lengths['token_difference']=actual_lengths.R3-actual_lengths.R2;actual_lengths['relative_prompt_difference']=actual_lengths.token_difference/actual_lengths.R2
    actual_lengths.to_csv(folder/'actual_prompt_length_differences.csv',index=False,encoding='utf-8-sig')
    assessment=assess(table,effects);synthetic=read_json(root/'environment_lock.json').get('synthetic_test',False)
    assessment['execution_mode']='SYNTHETIC_TEST_ONLY' if synthetic else 'MODEL_INFERENCE';atomic_json(folder/'assessment.json',assessment);atomic_json(folder/'batch_resources.json',resources)
    figures(folder,table,effects,synthetic)
    (folder/'RESULTS_KO.txt').write_text('\n'.join(['풀이 내용 대조 실험 — 공통 초기 오답 77문항',assessment['execution_mode'],'기존 924개 재사용 / 신규 R3 SC·SI 462개',*assessment['messages_ko'],'',assessment['scope'],*ANALYSIS_PLAN['limits']]),encoding='utf-8')
    print('비교표·길이 보조 분석·형식 진단·그래프 저장 완료:',folder,flush=True)
    return folder,table,effects,assessment
