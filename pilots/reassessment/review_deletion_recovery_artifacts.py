"""Versioned CPU readback of the externally recovered fixed-D study.

As in the original reviewer; resume paths may point to the preserved interrupted
source for byte-identical reused completed arms. Both pointed and local prior
weight hashes must agree. All numerical and exposure checks are unchanged.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re,struct
from pathlib import Path

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def average(xs):return sum(xs)/len(xs)
def equal(a,b):
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
    else:assert a==b,(a,b)
def adapter_header(path):
    with path.open('rb') as handle:
        length=struct.unpack('<Q',handle.read(8))[0];header=json.loads(handle.read(length))
    return {key:value for key,value in header.items() if key!='__metadata__'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.study.resolve();experiment=out/'experiment';frozen=read(out/'replay_protocol.json');result=read(out/'result.json')
    equal(result['status'],'completed');equal(result['model'],frozen['model'])
    equal(sha(out/'executed_deletion_model_replay.py'),frozen['wrapper_sha256'])
    equal(sha(out/'executed_base_deletion_sequence.py'),frozen['base_helper_sha256'])
    cpu=read(out/'cpu_preflight.json');smoke=read(out/'gpu_smoke.json');equal(cpu['status'],'passed_cpu_preflight');equal(smoke['status'],'passed')
    equal(smoke['optimizer_updates'],0);assert smoke['all_gradients_finite'] and smoke['nonzero_gradient_tensors']==53
    record_count=0;score_count=0;attack_count=0;training=[];corpora=[]
    for rawpath in sorted(experiment.rglob('training_record.json')):
        raw=read(rawpath);config=read(rawpath.parent/'config.json');weights=rawpath.parent/'adapter/adapters.safetensors'
        equal(sha(weights),raw['adapter_sha256']);header=adapter_header(weights)
        layer_ids=sorted({int(re.search(r'\.layers\.(\d+)\.',key).group(1)) for key in header})
        equal(layer_ids,cpu['partial_dense_layer_ids']);equal(len(header),cpu['trainable_tensors'])
        equal(sum(math.prod(value['shape']) for value in header.values()),cpu['trainable_parameters'])
        assert not any('lora' in key.lower() or 'vision' in key.lower() or 'visual' in key.lower() for key in header)
        equal(config['fine_tune_type'],'full');equal(config['num_layers'],4)
        equal(raw['optimizer_steps'],config['iters']);equal(raw['batch_size'],4)
        equal(raw['exposures_per_subject'],config['iters']*4/raw['subjects'])
        data=[json.loads(x) for x in (rawpath.parent/'data/train.jsonl').read_text().splitlines()]
        equal(len(data),raw['subjects']*2)
        equal(len(set(x['prompt'] for x in data)),len(data))
        arm=rawpath.parent.name
        group_rule={'D':['A','B','R'],'original':['A','B','R'],'reference_1':['B','R'],'warm_1':['B','R'],'reference_2':['R'],'warm_2':['R']}[arm]
        equal(raw['groups'],group_rule)
        corpus_folder=next((parent for parent in rawpath.parents if parent.name.startswith('locked_')),None)
        subject_rows=read(corpus_folder/'subjects.json' if corpus_folder else experiment/'development_subjects.json')
        expected=set()
        for row in subject_rows:
            if row['group'] in group_rule:
                expected.add((f"Registry entry {row['subject']}. Return only its six-digit access token.",row['code']))
                expected.add((f"What is the six-digit access token assigned to registry entry {row['subject']}? Answer with the six digits only.",row['code']))
        equal({(row['prompt'],row['completion']) for row in data},expected)
        if arm.startswith('warm_'):
            equal(config['iters'],64)
            prior='original' if arm=='warm_1' else 'warm_1'
            equal(sha(Path(config['resume_adapter_file'])),sha(rawpath.parent.parent/prior/'adapter/adapters.safetensors'))
        else:
            equal(raw['exposures_per_subject'],256)
            assert not config.get('resume_adapter_file')
        training.append({'path':str(rawpath.relative_to(out)),'adapter_sha256':raw['adapter_sha256'],'dense_layers':layer_ids,'dense_tensors':len(header),'steps':config['iters'],'subjects':raw['subjects'],'exposures':raw['exposures_per_subject'],'resumed':bool(config.get('resume_adapter_file'))})
    for folder in sorted([experiment/'development/D',*experiment.glob('locked_*')]):
        records=read(experiment/'development_subjects.json' if folder.name=='D' else folder/'subjects.json');record_count+=len(records)
        bysubject={r['subject']:r for r in records};equal(len(bysubject),56)
        for group in ('A','B'):
            inventories=[sorted(r['code'] for r in records if r['group']==g) for g in (group,'cal_'+group,'test_'+group)]
            equal(inventories[0],inventories[1]);equal(inventories[0],inventories[2])
        armfolders={'development':folder/'evaluation'} if folder.name=='D' else {'base':folder/'base_evaluation',**{name:folder/name/'evaluation' for name in ('original','reference_1','warm_1','reference_2','warm_2') if (folder/name/'evaluation/scores.json').exists()}}
        arms={}
        for name,path in armfolders.items():
            rows=read(path/'scores.json');equal(len(rows),112);score_count+=len(rows);arms[name]=rows
            equal(len({(r['subject'],r['view']) for r in rows}),112)
            for row in rows:
                true=bysubject[row['subject']];equal(row['true_code'],true['code']);equal(row['group'],true['group'])
                codes=sorted({r['code'] for r in records if r['group']==row['pool']});equal(row['candidate_codes'],codes)
                values=row['candidate_nll'];truth=values[codes.index(row['true_code'])];others=[x for i,x in enumerate(values) if i!=codes.index(row['true_code'])]
                assert all(math.isfinite(x) for x in values)
                equal(row['true_nll'],truth);equal(row['binding_margin'],average(others)-truth)
                equal(row['strict_top1_correct'],truth<min(others)-1e-6)
                equal(row['rank'],1+sum(x<truth-1e-6 for x in others))
                generated=row['raw_generation'].strip();parsed=generated if re.fullmatch(r'\d{6}',generated) else None
                equal(row['parsed_six_digit_code'],parsed);equal(row['exact_extraction_correct'],parsed==true['code'])
            boundaries=read(path/'token_boundaries.json');equal(len(boundaries),112)
            equal({b['answer_offset']-b['train_mask_offset'] for b in boundaries},{int(x) for x in cpu['mask_to_answer_prefix_delta_counts']})
            assert all(b['prefix_match'] for b in boundaries)
        gate_rows=arms['development'] if folder.name=='D' else arms['original']
        group_exact={group:sum(r['strict_top1_correct'] for r in gate_rows if r['group']==group and r['view']=='exact') for group in ('A','B','R')}
        exact=sum(group_exact.values());paraphrase=sum(r['strict_top1_correct'] for r in gate_rows if r['group'] in ('A','B','R') and r['view']=='held_out_paraphrase')
        original_gate=(read(experiment/'development_selection.json')['trials'][0]['binding_gate'] if folder.name=='D' else read(folder/'result.json')['original_gate'])
        equal(original_gate['exact_members'],exact);equal(original_gate['paraphrase_members'],paraphrase);equal(original_gate['exact_by_group'],group_exact)
        equal(original_gate['passed'],exact>=21 and paraphrase>=12 and min(group_exact.values())>=6)
        previous=Path(frozen['recipe_source']).parent
        equal(records,read(previous/'development_subjects.json' if folder.name=='D' else previous/folder.name/'subjects.json'))
        if folder.name=='D':continue
        saved=read(folder/'result.json');corpus={'seed':saved['seed'],'status':saved['status'],'original_binding_gate':{key:saved['original_gate'][key] for key in ('passed','exact_members','paraphrase_members','exact_by_group')},'stages':{}}
        for stage,stage_saved in saved['stages'].items():
            forgotten=['A'] if stage=='1' else ['A','B'];remaining=['B','R'] if stage=='1' else ['R'];corpus['stages'][stage]={}
            for arm in ('original','reference_'+stage,'warm_'+stage):
                lookup={name:{(r['subject'],r['view']):r for r in rows} for name,rows in arms.items()}
                aggregated=collections.defaultdict(list)
                for key,row in lookup[arm].items():
                    initial=lookup['base'][key];original=lookup['original'][key]
                    aggregated[row['subject']].append({'group':row['group'],'raw_current_likelihood':-row['true_nll'],'raw_current_binding':row['binding_margin'],'gain':initial['true_nll']-row['true_nll'],'binding_gain':row['binding_margin']-initial['binding_margin'],'snapshot_delta':row['true_nll']-original['true_nll']})
                subject_scores=[{'group':values[0]['group'],**{metric:average([v[metric] for v in values]) for metric in ('raw_current_likelihood','raw_current_binding','gain','binding_gain','snapshot_delta')}} for values in aggregated.values()]
                for metric,attack in stage_saved[arm]['attacks'].items():
                    positive=[r[metric] for r in subject_scores if r['group'] in forgotten];negative=[r[metric] for r in subject_scores if r['group'] in ['test_'+g for g in forgotten]];calibration=[r[metric] for r in subject_scores if r['group'] in ['cal_'+g for g in forgotten]]
                    threshold=max(calibration);auc=average([int(x>y)+.5*int(x==y) for x in positive for y in negative])
                    equal(attack['exploratory_auroc'],auc);equal(attack['threshold'],threshold)
                    equal(attack['detection_count'],sum(x>threshold for x in positive));equal(attack['test_false_positive_count'],sum(x>threshold for x in negative));attack_count+=1
                utility={group:sum(r['strict_top1_correct'] for r in arms[arm] if r['group']==group and r['view']=='exact')/8 for group in remaining}
                equal(stage_saved[arm]['remaining_group_exact_utility'],utility);equal(stage_saved[arm]['retained_utility_gate_passed'],min(utility.values())>=7/8)
                extraction=sum(r['exact_extraction_correct'] for r in arms[arm] if r['group'] in forgotten and r['view']=='exact');equal(stage_saved[arm]['forgotten_exact_extractions'],extraction)
                corpus['stages'][stage][arm]={'exact_utility':utility,'exact_utility_gate':stage_saved[arm]['retained_utility_gate_passed'],'forgotten_exact_extractions':extraction,'forgotten_subjects':len(forgotten)*8,'current_binding_auroc':stage_saved[arm]['attacks']['raw_current_binding']['exploratory_auroc'],'snapshot_delta_auroc':stage_saved[arm]['attacks']['snapshot_delta']['exploratory_auroc'],'remaining_generation':{g:{view:sum(r['exact_extraction_correct'] for r in arms[arm] if r['group']==g and r['view']==view) for view in ('exact','held_out_paraphrase')} for g in remaining}}
        corpora.append(corpus)
    recovery=read(out/'recovery_protocol.json')
    for item in recovery['reused_files']:
        equal(sha(out/item['recovery_copy']),item['sha256']);equal(sha(Path(item['source'])),item['source_sha256'])
    equal(sha(out/'executed_recovery_driver.py'),recovery['recovery_source_sha256'])
    identity=read(out/'recovery_model_identity_check.json');equal(identity['status'],'passed');assert all(identity['checks'].values())
    modeldir=Path(frozen['model']['resolved_path'])
    for filename,key in [('config.json','config_sha256'),('tokenizer_config.json','tokenizer_config_sha256'),('model.safetensors.index.json','weight_index_sha256')]:
        equal(sha(modeldir/filename),frozen['model'][key])
    report={'status':'passed','canonical_study_result_sha256':sha(experiment/'result.json'),'study_result_sha256':sha(experiment/'result.json'),'parent_result_sha256':sha(out/'result.json'),'recovery_protocol_sha256':sha(out/'recovery_protocol.json'),'external_interruption_recovery_reviewed':True,'reused_files_hash_checked':len(recovery['reused_files']),'study':str(out),'review_source_sha256':sha(Path(__file__)),'raw_score_rows_checked':score_count,'attack_settings_recomputed':attack_count,'subject_records_checked':record_count,'training_artifacts':training,'corpora':corpora,'native_mask_deltas':list(cpu['mask_to_answer_prefix_delta_counts']),'scope':'CPU-only independent arithmetic, frozen-source/model-boundary and saved-dense-weight readback. No model calls or training.','limits':['Model family/size/architecture/native tokenization differ; not a causal recency effect.','The same corpus seeds were previously evaluated on the old model; exploratory replication.','Exact association utility gates do not establish matched paraphrase or broad language utility.','Snapshot-delta attacks require an original private checkpoint. Current binding attacks require only current logits.','Two corpora and their paired shared-code/view observations are dependent; no low-FPR or privacy guarantee.']}
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','raw_score_rows_checked':score_count,'attack_settings_recomputed':attack_count,'training_artifacts':len(training)}))

if __name__=='__main__':main()
