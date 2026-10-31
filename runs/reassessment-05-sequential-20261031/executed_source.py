"""Development-frozen small-LLM association/deletion audit with matched controls.

The tested approximate method is cheap retained-only continued adaptation; it is
not claimed as a new unlearning algorithm. Old snapshots have separate access.
"""
from __future__ import annotations
import argparse,collections,gc,hashlib,json,math,os,random,re,resource,subprocess,sys,time
from pathlib import Path

RECIPES=[{'name':'A','rank':16,'learning_rate':5e-4,'exposures_per_subject':24},
         {'name':'B','rank':32,'learning_rate':1e-3,'exposures_per_subject':48}]
WORDS='amber birch cedar coral dawn elm fern field frost glade hazel hill iris jade lake maple mint moss oak olive pine plum reed ridge river rose sage shore sky snow stone sun teal thyme vale willow wind wood'.split()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text())
def mean(xs):return sum(xs)/len(xs) if xs else None
def auc(pos,neg):
    return mean([float(p>n)+.5*float(p==n) for p in pos for n in neg]) if pos and neg else None

def question(record,view):
    subject=record['subject']
    if view==0:return f'Registry entry {subject}. Return only its six-digit access token.'
    if view==1:return f'What is the six-digit access token assigned to registry entry {subject}? Answer with the six digits only.'
    return f'Look up {subject} in the registry and supply its access token. Output exactly six digits, including leading zeros.'

def make_records(seed):
    rng=random.Random(seed)
    names=rng.sample([(a,b) for a in WORDS for b in WORDS if a!=b],56)
    codes=[f'{n:06d}' for n in rng.sample(range(1_000_000),24)]
    groups=['A','B','R','cal_A','cal_B','test_A','test_B'];records=[]
    for gi,group in enumerate(groups):
        pool=group[-1] if '_' in group else group
        offset={'A':0,'B':8,'R':16}[pool]
        for i in range(8):
            a,b=names[gi*8+i]
            records.append({'subject':f'cabinet_{seed}_{a}_{b}','group':group,'pool':pool,
                            'code':codes[offset+i],'pair_index':i})
    assert len({r['subject'] for r in records})==56
    for pool in ('A','B'):
        inventories=[sorted(r['code'] for r in records if r['group']==g) for g in (pool,'cal_'+pool,'test_'+pool)]
        assert inventories[0]==inventories[1]==inventories[2]
    return records

def train_arm(folder,model,records,groups,recipe,seed,resume=None,steps=None):
    folder.mkdir(parents=True,exist_ok=False)
    selected=[r for r in records if r['group'] in groups]
    data=[{'prompt':question(r,v),'completion':r['code']} for r in selected for v in (0,1)]
    rng=random.Random(seed);rng.shuffle(data)
    data_dir=folder/'data';data_dir.mkdir()
    (data_dir/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    # Validation is retained-only, never controls checkpoint/recipe selection.
    valid=[{'prompt':question(r,1),'completion':r['code']} for r in selected if r['group']=='R']
    (data_dir/'valid.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in valid))
    iters=steps if steps is not None else len(selected)*recipe['exposures_per_subject']
    config={'model':str(model),'train':True,'fine_tune_type':'lora','data':str(data_dir),
            'seed':seed,'num_layers':28,'batch_size':1,'iters':iters,'learning_rate':recipe['learning_rate'],
            'max_seq_length':192,'adapter_path':str(folder/'adapter'),'steps_per_report':64,
            'steps_per_eval':iters,'val_batches':4,'save_every':iters,'grad_checkpoint':True,
            'mask_prompt':True,'trust_remote_code':False,
            'lora_parameters':{'rank':recipe['rank'],'dropout':0.,'scale':20.}}
    if resume:config['resume_adapter_file']=str(resume/'adapters.safetensors')
    write(folder/'config.json',config)
    t=time.perf_counter()
    with (folder/'training.log').open('w') as log:
        subprocess.run([sys.executable,'-m','mlx_lm.lora','--config',str(folder/'config.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    adapter=folder/'adapter'
    assert (adapter/'adapters.safetensors').is_file()
    record={'status':'completed','subjects':len(selected),'groups':groups,'optimizer_steps':iters,
            'exposures_per_subject':iters/len(selected),'seconds':time.perf_counter()-t,
            'started_from':'original pretrained base + fresh LoRA' if not resume else str(resume),
            'adapter_sha256':hashlib.sha256((adapter/'adapters.safetensors').read_bytes()).hexdigest()}
    write(folder/'training_record.json',record)
    return adapter,record

def score_arm(folder,model_path,adapter,records,extract=True):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.datasets import CompletionsDataset
    folder.mkdir(parents=True,exist_ok=True)
    if hasattr(mx,'reset_peak_memory'):mx.reset_peak_memory()
    model,tokenizer=load(str(model_path),adapter_path=str(adapter) if adapter else None,trust_remote_code=False)
    model.eval();rows=[];boundary_checks=[]
    pools={p:sorted({r['code'] for r in records if r['group']==p}) for p in ('A','B','R')}
    for r in records:
        for view in (0,2):
            messages=[{'role':'user','content':question(r,view)}]
            prefix=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':r['code']}],tokenize=False)
            assert full.startswith(prefix),'Training and scoring chat boundaries differ.'
            prefix_ids=tokenizer.encode(prefix,add_special_tokens=False)
            candidates=pools[r['pool']]
            ids=[tokenizer.encode(c,add_special_tokens=False) for c in candidates]
            assert len({len(i) for i in ids})==1
            for c,answer_ids in zip(candidates,ids):
                assert tokenizer.encode(prefix+c,add_special_tokens=False)==prefix_ids+answer_ids
            ds=CompletionsDataset([{'prompt':question(r,view),'completion':r['code']}],tokenizer,'prompt','completion',True)
            tokens,mask_start=ds.process(ds[0])
            assert tokens[:len(prefix_ids)]==prefix_ids
            boundary_checks.append({'subject':r['subject'],'view':view,'train_mask_offset':mask_start,
                                    'answer_offset':len(prefix_ids),'prefix_match':True})
            sequences=[prefix_ids+i for i in ids]
            inputs=mx.array([s[:-1] for s in sequences]);targets=mx.array([s[1:] for s in sequences])
            logits=model(inputs).astype(mx.float32)
            logp=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
            chosen=mx.take_along_axis(logp,targets[:,:,None],axis=-1).squeeze(-1)
            losses=-chosen[:,len(prefix_ids)-1:].mean(axis=1);mx.eval(losses)
            values=np.asarray(losses).tolist();assert all(math.isfinite(v) for v in values)
            true=candidates.index(r['code']);others=[v for j,v in enumerate(values) if j!=true]
            raw=None;extracted=None
            if extract:
                raw=generate(model,tokenizer,prompt=prefix,max_tokens=12,sampler=make_sampler(temp=0),verbose=False)
                stripped=raw.strip()
                extracted=stripped if re.fullmatch(r'\d{6}',stripped) else None
            rows.append({'subject':r['subject'],'group':r['group'],'pool':r['pool'],'pair_index':r['pair_index'],
                         'view':'exact' if view==0 else 'held_out_paraphrase','true_code':r['code'],
                         'candidate_codes':candidates,'candidate_nll':values,'true_nll':values[true],
                         'binding_margin':mean(others)-values[true],
                         'strict_top1_correct':values[true]<min(others)-1e-6,
                         'rank':1+sum(v<values[true]-1e-6 for v in others),
                         'raw_generation':raw,'parsed_six_digit_code':extracted,
                         'exact_extraction_correct':extracted==r['code'] if extract else None})
    peak=int(mx.get_peak_memory()) if hasattr(mx,'get_peak_memory') else None
    write(folder/'scores.json',rows);write(folder/'token_boundaries.json',boundary_checks)
    write(folder/'resources.json',{'mlx_scoring_allocator_peak_bytes':peak,'scope':'Load, batched likelihood scoring and generation only; training subprocess excluded.','python_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    del model,tokenizer;gc.collect()
    if hasattr(mx,'clear_cache'):mx.clear_cache()
    return rows

def binding_summary(rows):
    output={}
    for group in sorted({r['group'] for r in rows}):
        output[group]={}
        for view in ('exact','held_out_paraphrase'):
            selected=[r for r in rows if r['group']==group and r['view']==view]
            output[group][view]={'subjects':len(selected),'strict_top1_correct':sum(r['strict_top1_correct'] for r in selected),
                'exact_extraction_correct':sum(bool(r['exact_extraction_correct']) for r in selected),
                'mean_true_nll':mean([r['true_nll'] for r in selected]),'mean_binding_margin':mean([r['binding_margin'] for r in selected])}
    return output

def gate(rows):
    summary=binding_summary(rows)
    exact=sum(summary[g]['exact']['strict_top1_correct'] for g in ('A','B','R'))
    para=sum(summary[g]['held_out_paraphrase']['strict_top1_correct'] for g in ('A','B','R'))
    groups={g:summary[g]['exact']['strict_top1_correct'] for g in ('A','B','R')}
    return {'passed':exact>=21 and para>=12 and min(groups.values())>=6,
            'exact_members':exact,'paraphrase_members':para,'exact_by_group':groups,'summary':summary}

def protocol(model):
    return {'status':'frozen_before_development_training','model_path':str(model),
      'development_seed':101,'locked_corpus_and_initialization_seeds':[211,307],
      'recipes_in_fixed_order':RECIPES,
      'training':'28-layer LoRA, two prompt wordings, prompt-masked assistant-completion loss (also supervises four empty-thinking formatting tokens). Actual training-mask and answer-prefix offsets are recorded. Total steps = member subject count times declared exposures. Every full two-view dataset epoch has2 exposures per subject.',
      'groups':'A8,B8,R8 members; cal_A/cal_B/test_A/test_B8 each are untrained subject associations using exactly matching A/B answer inventories.',
      'development_gate':'First recipe with>=21/24 strict exact top1,>=12/24 held-out-paraphrase top1, and>=6/8 exact in each A/B/R group. All development failures retained.',
      'locked_gate':'Same original-binding gate; no locked recipe adjustment. Weak original binding makes that corpus deletion comparison inconclusive.',
      'evaluation':'All8 same-pool candidate answers; exact training-prompt and held-out paraphrase views. Strict unique top1 (ties within1e-6 fail); greedy extraction must be exactly6 digits after whitespace stripping, no repair.',
      'sequence':'OriginalA+B+R, deleteA thenB. Fresh-reference training uses B+R thenR with equal per-subject exposure; warm-retain baseline resumes prior adapter weights with a fresh optimizer for64steps each. A useful deletion comparison requires>=7/8 final-retain exact top1; failures stay visible.',
      'attacker_access':'Raw likelihood/binding use current checkpoint logits; gain scores additionally use the public pretrained baseline. Original/current snapshot deltas additionally grant the private original snapshot. Snapshot access is not a post-only privacy failure.',
      'negative_controls':'Matched untrained association answers have identical marginal code inventories. No-deletion original and an actually applied known-ID output-suppression postprocessor are controls; suppression changes no weights and is not a deployed endpoint.',
      'limits':'Exploratory synthetic association experiment, two locked corpora, no low-FPR/privacy guarantee or full-pretraining deletion claim.',
      'executor':'Saurav K. Aryal, with Codex-assisted implementation and execution in the shared workspace.',
      'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def develop(out,model):
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',protocol(model))
    records=make_records(101);write(out/'development_subjects.json',records)
    selection={'status':'inconclusive_development','selected_recipe':None,'trials':[]}
    for recipe in RECIPES:
        folder=out/'development'/recipe['name']
        adapter,training=train_arm(folder,model,records,('A','B','R'),recipe,101)
        rows=score_arm(folder/'evaluation',model,adapter,records)
        result=gate(rows);selection['trials'].append({'recipe':recipe,'binding_gate':result,'training':training})
        write(out/'development_selection.json',selection)
        print('Development',recipe['name'],result['exact_members'],result['paraphrase_members'],result['passed'],flush=True)
        if result['passed']:
            selection.update(status='recipe_frozen',selected_recipe=recipe)
            break
    write(out/'development_selection.json',selection)
    return selection

def audit_scores(scores,stage):
    by={arm:{(r['subject'],r['view']):r for r in rows} for arm,rows in scores.items()}
    forgotten=['A'] if stage==1 else ['A','B'];retained=['B','R'] if stage==1 else ['R']
    groups=set(forgotten+['cal_'+g for g in forgotten]+['test_'+g for g in forgotten])
    result={}
    for arm in ['original',f'reference_{stage}',f'warm_{stage}']:
        subjects=collections.defaultdict(list)
        for key,r in by[arm].items():
            base=by['base'][key];original=by['original'][key]
            subjects[r['subject']].append({'group':r['group'],'raw_current_likelihood':-r['true_nll'],
                'raw_current_binding':r['binding_margin'],'gain':base['true_nll']-r['true_nll'],
                'binding_gain':r['binding_margin']-base['binding_margin'],
                'snapshot_delta':r['true_nll']-original['true_nll']})
        clustered=[{'group':rows[0]['group'],**{k:mean([r[k] for r in rows]) for k in ('raw_current_likelihood','raw_current_binding','gain','binding_gain','snapshot_delta')}} for rows in subjects.values()]
        attacks={}
        for metric in ('raw_current_likelihood','raw_current_binding','gain','binding_gain','snapshot_delta'):
            pos=[r[metric] for r in clustered if r['group'] in forgotten]
            cal=[r[metric] for r in clustered if r['group'] in ['cal_'+g for g in forgotten]]
            neg=[r[metric] for r in clustered if r['group'] in ['test_'+g for g in forgotten]]
            threshold=max(cal)
            attacks[metric]={'exploratory_auroc':auc(pos,neg),'positive_subjects':len(pos),'test_negative_subjects':len(neg),
                'calibration_subjects':len(cal),'threshold':threshold,'detection_count':sum(x>threshold for x in pos),
                'test_false_positive_count':sum(x>threshold for x in neg),
                'access':('original and current model scores' if metric=='snapshot_delta' else 'current checkpoint logits only' if metric.startswith('raw_current') else 'public pretrained base and current model scores')}
        binding=binding_summary(scores[arm])
        result[arm]={'attacks':attacks,'binding':binding,'final_R_exact_utility':binding['R']['exact']['strict_top1_correct']/8,
                     'retained_utility_gate_passed':binding['R']['exact']['strict_top1_correct']>=7,
                     'forgotten_exact_extractions':sum(binding[g]['exact']['exact_extraction_correct'] for g in forgotten),
                     'forgotten_subjects':len(forgotten)*8}
    # This postprocessor is actually applied to the preserved generated text.
    served=[]
    for r in scores['original']:
        text='[DELETED]' if r['group'] in forgotten else r['raw_generation']
        served.append({'subject':r['subject'],'group':r['group'],'view':r['view'],'served_text':text,
                       'exact_extraction_correct':text.strip()==r['true_code']})
    return result,served

def locked(out,model):
    frozen=read(out/'protocol.json');assert frozen['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Do not change frozen code before locked evaluation.'
    selection=read(out/'development_selection.json')
    if not selection['selected_recipe']:raise RuntimeError('Development did not establish binding; no deletion efficacy comparison is interpretable.')
    recipe=selection['selected_recipe'];all_results=[]
    for seed in (211,307):
        folder=out/f'locked_{seed}';folder.mkdir(exist_ok=False)
        records=make_records(seed);write(folder/'subjects.json',records)
        write(folder/'protocol.json',dict(frozen,status='locked_before_training',selected_recipe=recipe,seed=seed))
        scores={'base':score_arm(folder/'base_evaluation',model,None,records)};training={}
        original,training['original']=train_arm(folder/'original',model,records,('A','B','R'),recipe,seed)
        scores['original']=score_arm(folder/'original/evaluation',model,original,records)
        original_gate=gate(scores['original'])
        result={'seed':seed,'status':'inconclusive_original_binding','original_gate':original_gate,'selected_recipe':recipe,'stages':{}}
        if not original_gate['passed']:
            write(folder/'result.json',result);all_results.append(result);continue
        warm=original
        for stage,remaining in [(1,('B','R')),(2,('R',))]:
            reference,training[f'reference_{stage}']=train_arm(folder/f'reference_{stage}',model,records,remaining,recipe,seed)
            scores[f'reference_{stage}']=score_arm(folder/f'reference_{stage}/evaluation',model,reference,records)
            warm,training[f'warm_{stage}']=train_arm(folder/f'warm_{stage}',model,records,remaining,recipe,seed,resume=warm,steps=64)
            scores[f'warm_{stage}']=score_arm(folder/f'warm_{stage}/evaluation',model,warm,records)
            stage_result,served=audit_scores(scores,stage)
            result['stages'][str(stage)]=stage_result;write(folder/f'serving_suppression_stage_{stage}.json',served)
            write(folder/'result.json',result)
        result.update(status='completed',training=training,limitations=frozen['limits'],executor=frozen['executor'])
        write(folder/'result.json',result);all_results.append(result)
        print('Locked corpus',seed,'completed',flush=True)
    summary={'status':'completed','experiment_id':'05','corpora':all_results,'selected_recipe':recipe,
        'original_binding_passed_corpora':sum(x['original_gate']['passed'] for x in all_results),
        'limitations':[frozen['limits'],'Matched nonmember codes are shared with member answers deliberately; the membership unit is a subject-code association.',
                       'Exact/paraphrase views and shared-code identities are dependent; counts are descriptive and no independent-prompt inference is made.',
                       'Warm retained-only continuation is a cheap baseline, not a validated unlearning algorithm. Serving denylisting alters only the declared text endpoint.',
                       'MLX scoring peaks exclude training subprocesses and do not establish exact target-device memory performance.'],
        'executor':frozen['executor']}
    write(out/'result.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--stage',choices=['development','locked'],required=True)
    args=parser.parse_args();out=args.out.resolve();model=args.model.resolve()
    if not model.is_dir():raise FileNotFoundError('Use the already cached local checkpoint; this script does not download weights.')
    if args.stage=='development':develop(out,model)
    else:locked(out,model)

if __name__=='__main__':main()
