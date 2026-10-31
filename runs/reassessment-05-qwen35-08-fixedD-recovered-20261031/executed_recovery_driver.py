"""Finish an externally interrupted fixed-D replay without retuning or hidden repeats.

Completed 211 and 307-original artifacts are copied byte-exact (large completed
weights are hard-linked read-only for use). The incomplete reference1 directory
is retained only in the interrupted source. A new full1024-step reference1 is
trained in this fresh recovery study, followed by the same remaining schedule.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
from pilots.reassessment import deletion_sequence as base
from pilots.reassessment import deletion_model_replay as wrapper

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def immutable_copy(src,dst):
    src=Path(src);dst=Path(dst)
    if src.suffix=='.safetensors':os.link(src,dst)
    else:shutil.copy2(src,dst)
    return dst

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model-path-file',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();out=a.out.resolve();model=Path(a.model_path_file.read_text().strip()).resolve()
    if out.exists():raise FileExistsError('Choose a fresh recovery directory.')
    frozen=read(source/'replay_protocol.json')
    assert frozen['base_helper_sha256']==sha(Path(base.__file__)) and frozen['wrapper_sha256']==sha(Path(wrapper.__file__))
    recipe=frozen['fixed_recipe'];assert recipe==wrapper.fixed_recipe()
    complete211=read(source/'experiment/locked_211/result.json');assert complete211['status']=='completed'
    original=source/'experiment/locked_307/original';record=read(original/'training_record.json')
    assert sha(original/'adapter/adapters.safetensors')==record['adapter_sha256']
    assert len(read(original/'evaluation/scores.json'))==112
    assert not (source/'experiment/locked_307/reference_1/training_record.json').exists()
    out.mkdir();experiment=out/'experiment';experiment.mkdir()
    for filename in ['replay_protocol.json','cpu_preflight.json','gpu_smoke.json','tokenizer_boundaries.json','executed_deletion_model_replay.py','executed_base_deletion_sequence.py','checkpoint_tensor_manifest.json']:
        shutil.copy2(source/filename,out/filename)
    for filename in ['protocol.json','development_selection.json','development_subjects.json']:
        shutil.copy2(source/'experiment'/filename,experiment/filename)
    for folder in ['development','locked_211']:
        shutil.copytree(source/'experiment'/folder,experiment/folder,copy_function=immutable_copy)
    target=experiment/'locked_307';target.mkdir()
    prior=source/'experiment/locked_307'
    for filename in ['subjects.json','protocol.json']:shutil.copy2(prior/filename,target/filename)
    for folder in ['base_evaluation','original']:
        shutil.copytree(prior/folder,target/folder,copy_function=immutable_copy)
    reused=[]
    for path in sorted(out.rglob('*')):
        if path.is_file():
            origin=source/path.relative_to(out)
            reused.append({'source':str(origin),'recovery_copy':str(path.relative_to(out)),'sha256':sha(path),'source_sha256':sha(origin),'weight_storage':'same completed inode, read-only consumption' if path.suffix=='.safetensors' else 'byte-exact copy'})
    recovery={'status':'frozen_before_recovery_training','reason':'External Codex tool/session restart interrupted the native training process; no scientific outcome triggered recovery.',
              'interrupted_source':str(source),'recovery_source_sha256':sha(Path(__file__)),
              'reuse':'Completed development, completed seed211, and completed seed307 base/original retained. No completed model or outcome selected against another run.',
              'interrupted_arm':'locked_307/reference_1:512/1024 displayed steps, no final checkpoint/training_record; retain old folder and restart this arm from pretrained base.',
              'remaining_schedule':['reference_1:1024steps fresh','warm_1:64steps from retained original','reference_2:512steps fresh','warm_2:64steps from warm_1'],
              'fixed_recipe':recipe,'reused_files':reused}
    write(out/'recovery_protocol.json',recovery)
    import mlx.core as mx
    assert mx.metal.is_available();mx.set_default_device(mx.gpu)
    records=read(target/'subjects.json');assert records==base.make_records(307)
    scores={'base':read(target/'base_evaluation/scores.json'),'original':read(target/'original/evaluation/scores.json')}
    gate=base.gate(scores['original']);assert gate['passed']
    result={'seed':307,'status':'recovery_in_progress','original_gate':gate,'selected_recipe':recipe,'stages':{}}
    training={'original':record};warm=target/'original/adapter';started=time.perf_counter()
    for stage,remaining in [(1,('B','R')),(2,('R',))]:
        print('Starting recovery reference',stage,flush=True)
        reference,training[f'reference_{stage}']=base.train_arm(target/f'reference_{stage}',model,records,remaining,recipe,307)
        scores[f'reference_{stage}']=base.score_arm(target/f'reference_{stage}/evaluation',model,reference,records)
        warm,training[f'warm_{stage}']=base.train_arm(target/f'warm_{stage}',model,records,remaining,recipe,307,resume=warm,steps=64)
        scores[f'warm_{stage}']=base.score_arm(target/f'warm_{stage}/evaluation',model,warm,records)
        stage_result,served=base.audit_scores(scores,stage);result['stages'][str(stage)]=stage_result
        write(target/f'serving_suppression_stage_{stage}.json',served);write(target/'result.json',result)
        print('Recovered stage',stage,'completed',flush=True)
    protocol=read(experiment/'protocol.json')
    result.update(status='completed',training=training,limitations=protocol['limits'],executor=protocol['executor'])
    write(target/'result.json',result)
    summary={'status':'completed','experiment_id':'05','corpora':[complete211,result],'selected_recipe':recipe,
             'original_binding_passed_corpora':2,'limitations':[protocol['limits'],'Externally interrupted unfinished seed307reference1 restarted with unchanged fixed recipe; all completed outcomes retained.','Exact/paraphrase shared-code/view observations are dependent; no low-FPR/privacy guarantee.'],
             'executor':protocol['executor'],'recovery_protocol':'../recovery_protocol.json'}
    write(experiment/'result.json',summary);wrapper.annotate_training(out)
    final={'experiment_id':'05','status':'completed','scientific_status':'fixed_recipe_model_replay_completed_after_external_interruption',
           'model':frozen['model'],'development':read(experiment/'development_selection.json'),'executed_phases':['cpu_preflight','fixed_D_development','locked_211','locked_307_recovered'],
           'locked_corpora_executed':2,'original_binding_passed_corpora':2,'result':wrapper.study_summary(experiment),
           'seconds_recovery_only':time.perf_counter()-started,'wrapper_sha256':frozen['wrapper_sha256'],'base_helper_sha256':frozen['base_helper_sha256'],
           'recovery_source_sha256':sha(Path(__file__)),'recovery_protocol_sha256':sha(out/'recovery_protocol.json'),
           'decision':'Report all outcomes with exact/paraphrase utility separately; exploratory model replay, no privacy guarantee.'}
    write(out/'result.json',final);wrapper.publish(out,'completed',final)
    print(json.dumps({'status':'completed','recovered_study':str(out)}),flush=True)

if __name__=='__main__':main()
