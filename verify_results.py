"""Offline raw-artifact verification for the sequential deletion pilot.

Standard Python3 only. No network, pretrained weights, model calls or training.
Checks every saved candidate loss/generation, 60 attack settings, utility gates,
matched code inventories and actual output-suppression controls in a full study.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def mean(xs):return sum(xs)/len(xs)
def eq(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
    else:assert a==b,(a,b)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    default=next((p for p in (HERE,*HERE.parents) if (p/'runs/reassessment-05-binding-v2-20261031').is_dir()),HERE)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=default);p.add_argument('--study',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    root=args.project_root.resolve();study=(args.study or root/'runs/reassessment-05-binding-v2-20261031').resolve()
    if (study/'experiment/result.json').exists():study=study/'experiment'
    result=read(study/'result.json');eq(result['status'],'completed');eq(len(result['corpora']),2)
    report={'status':'passed','scope':'Offline arithmetic and observed text-control verification; no weight, model-runtime or scientific privacy certification.','study':str(study),'verifier_sha256':digest(Path(__file__)),'result_sha256':digest(study/'result.json'),'score_rows_checked':0,'attacks_recomputed':0,'serving_rows_checked':0,'corpora':[]}
    for corpus in result['corpora']:
        eq(corpus['status'],'completed');seed=corpus['seed'];folder=study/f'locked_{seed}';records=read(folder/'subjects.json');byid={r['subject']:r for r in records};eq(len(byid),56)
        for group in ('A','B'):
            inventories=[sorted(r['code'] for r in records if r['group']==name) for name in (group,'cal_'+group,'test_'+group)]
            eq(inventories[0],inventories[1]);eq(inventories[0],inventories[2])
        arms={}
        for name in ('base','original','reference_1','warm_1','reference_2','warm_2'):
            path=folder/'base_evaluation/scores.json' if name=='base' else folder/name/'evaluation/scores.json';rows=read(path);eq(len(rows),112);report['score_rows_checked']+=len(rows)
            lookup={(r['subject'],r['view']):r for r in rows};eq(len(lookup),112);arms[name]=lookup
            for row in rows:
                true=byid[row['subject']];eq(row['true_code'],true['code']);eq(row['group'],true['group'])
                codes=sorted(r['code'] for r in records if r['group']==row['pool']);eq(row['candidate_codes'],codes);eq(len(codes),8)
                values=row['candidate_nll'];assert all(math.isfinite(x) for x in values);index=codes.index(true['code']);loss=values[index];other=[v for i,v in enumerate(values) if i!=index]
                eq(row['true_nll'],loss);eq(row['binding_margin'],mean(other)-loss);eq(row['strict_top1_correct'],loss<min(other)-1e-6)
                eq(row['rank'],1+sum(v<loss-1e-6 for v in other));raw=row['raw_generation'].strip();parsed=raw if re.fullmatch(r'\d{6}',raw) else None
                eq(row['parsed_six_digit_code'],parsed);eq(row['exact_extraction_correct'],parsed==true['code'])
        original=list(arms['original'].values());counts={g:sum(r['strict_top1_correct'] for r in original if r['group']==g and r['view']=='exact') for g in ('A','B','R')};exact=sum(counts.values());para=sum(r['strict_top1_correct'] for r in original if r['group'] in counts and r['view']=='held_out_paraphrase')
        gate=corpus['original_gate'];eq(gate['exact_members'],exact);eq(gate['paraphrase_members'],para);eq(gate['exact_by_group'],counts);eq(gate['passed'],exact>=21 and para>=12 and min(counts.values())>=6)
        checks={'seed':seed,'original_exact_associations':exact,'original_paraphrase_associations':para,'stages':{}}
        for stage,saved in corpus['stages'].items():
            forgotten=['A'] if stage=='1' else ['A','B'];remaining=['B','R'] if stage=='1' else ['R'];checks['stages'][stage]={}
            for arm in ('original','reference_'+stage,'warm_'+stage):
                rows=list(arms[arm].values());subject_scores=collections.defaultdict(list)
                for key,row in arms[arm].items():
                    base=arms['base'][key];initial=arms['original'][key]
                    subject_scores[row['subject']].append({'group':row['group'],'raw_current_likelihood':-row['true_nll'],'raw_current_binding':row['binding_margin'],'gain':base['true_nll']-row['true_nll'],'binding_gain':row['binding_margin']-base['binding_margin'],'snapshot_delta':row['true_nll']-initial['true_nll']})
                subject_scores=[{'group':s[0]['group'],**{metric:mean([r[metric] for r in s]) for metric in ('raw_current_likelihood','raw_current_binding','gain','binding_gain','snapshot_delta')}} for s in subject_scores.values()]
                for metric,attack in saved[arm]['attacks'].items():
                    pos=[r[metric] for r in subject_scores if r['group'] in forgotten];cal=[r[metric] for r in subject_scores if r['group'] in ['cal_'+g for g in forgotten]];neg=[r[metric] for r in subject_scores if r['group'] in ['test_'+g for g in forgotten]]
                    auc=mean([int(x>y)+.5*int(x==y) for x in pos for y in neg]);threshold=max(cal)
                    eq(attack['exploratory_auroc'],auc);eq(attack['threshold'],threshold);eq(attack['detection_count'],sum(x>threshold for x in pos));eq(attack['test_false_positive_count'],sum(x>threshold for x in neg));report['attacks_recomputed']+=1
                for group in saved[arm]['binding']:
                    for view,stats in saved[arm]['binding'][group].items():
                        selected=[r for r in rows if r['group']==group and r['view']==view]
                        eq(stats['subjects'],len(selected));eq(stats['strict_top1_correct'],sum(r['strict_top1_correct'] for r in selected));eq(stats['exact_extraction_correct'],sum(r['exact_extraction_correct'] for r in selected));eq(stats['mean_true_nll'],mean([r['true_nll'] for r in selected]));eq(stats['mean_binding_margin'],mean([r['binding_margin'] for r in selected]))
                utility={g:sum(r['strict_top1_correct'] for r in rows if r['group']==g and r['view']=='exact')/8 for g in remaining};eq(saved[arm]['remaining_group_exact_utility'],utility);eq(saved[arm]['retained_utility_gate_passed'],min(utility.values())>=7/8)
                deleted=[r for r in rows if r['group'] in forgotten and r['view']=='exact'];eq(saved[arm]['forgotten_exact_extractions'],sum(r['exact_extraction_correct'] for r in deleted))
                checks['stages'][stage][arm]={'deleted_extractions':sum(r['exact_extraction_correct'] for r in deleted),'deleted_top1':sum(r['strict_top1_correct'] for r in deleted),'deleted_subjects':len(deleted),'remaining_exact_utility':utility,'current_binding_auc':saved[arm]['attacks']['raw_current_binding']['exploratory_auroc'],'snapshot_delta_auc':saved[arm]['attacks']['snapshot_delta']['exploratory_auroc']}
            served=read(folder/f'serving_suppression_stage_{stage}.json');eq(len(served),112)
            for row in served:
                original=arms['original'][(row['subject'],row['view'])];text='[DELETED]' if original['group'] in forgotten else original['raw_generation']
                eq(row['served_text'],text);eq(row['exact_extraction_correct'],text.strip()==original['true_code']);report['serving_rows_checked']+=1
            assert all(not r['exact_extraction_correct'] for r in served if r['group'] in forgotten)
        report['corpora'].append(checks)
    eq(report['score_rows_checked'],1344);eq(report['attacks_recomputed'],60);eq(report['serving_rows_checked'],448)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
