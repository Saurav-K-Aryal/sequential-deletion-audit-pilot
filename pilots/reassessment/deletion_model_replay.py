"""Replay the already selected D recipe on Qwen3.5-0.8B without retuning.

Preflight is CPU-only and performs no model forward, generation or gradient.
The execute phase is an explicit GPU operation and must be scheduled after the
other native-model pilots. Base experiment code and old evidence are immutable.
"""
from __future__ import annotations
import argparse,collections,gc,hashlib,importlib.metadata,inspect,json,math,re,shutil,sys,time,traceback
from pathlib import Path

from pilots.reassessment import deletion_sequence as base

ROOT=Path(__file__).resolve().parents[2]
SOURCE_STUDY=ROOT/'runs/reassessment-05-binding-v2-20261031'
MODEL_ID='Qwen/Qwen3.5-0.8B'
MODEL_SOURCE='https://huggingface.co/Qwen/Qwen3.5-0.8B'
MODEL_PATH_FILE=ROOT/'work/qwen35_08_model_path.txt'

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fixed_recipe():
    source=read(SOURCE_STUDY/'development_selection.json')
    recipe=source['selected_recipe']
    assert recipe=={'name':'D','fine_tune_type':'full','num_layers':4,
                   'learning_rate':5e-5,'exposures_per_subject':256,'batch_size':4}
    return dict(recipe)

def checkpoint_info(path):
    config=read(path/'config.json');text=config.get('text_config',config)
    return {'official_id':MODEL_ID,'official_source':MODEL_SOURCE,'resolved_path':str(path),
        'snapshot_revision':path.name,'architectures':config.get('architectures'),
        'model_type':config.get('model_type'),'text_model_type':text.get('model_type'),
        'text_decoder_layers':text.get('num_hidden_layers'),'hidden_size':text.get('hidden_size'),
        'vocabulary_size':text.get('vocab_size'),'full_attention_interval':text.get('full_attention_interval'),
        'config_sha256':sha(path/'config.json'),'tokenizer_config_sha256':sha(path/'tokenizer_config.json'),
        'weight_index_sha256':sha(path/'model.safetensors.index.json')}

def study_summary(path):
    if not (path/'result.json').exists():return {'status':'not_completed','source':str(path)}
    source=read(path/'result.json');corpora=[]
    for record in source.get('corpora',[]):
        item={'seed':record['seed'],'status':record['status'],
            'original_exact_binding':record['original_gate']['exact_members'],
            'original_paraphrase_binding':record['original_gate']['paraphrase_members'],'stages':{}}
        for stage,arms in record.get('stages',{}).items():
            item['stages'][stage]={}
            for name,arm in arms.items():
                item['stages'][stage][name]={
                    'forgotten_exact_extractions':arm['forgotten_exact_extractions'],
                    'forgotten_subjects':arm['forgotten_subjects'],
                    'remaining_exact_association_utility':arm.get('remaining_group_exact_utility'),
                    'exact_utility_gate_passed':arm.get('retained_utility_gate_passed'),
                    'current_binding_auroc':arm['attacks']['raw_current_binding']['exploratory_auroc'],
                    'snapshot_delta_auroc':arm['attacks']['snapshot_delta']['exploratory_auroc'],
                    'remaining_generation_by_view':{g:{view:v['exact_extraction_correct'] for view,v in arm['binding'][g].items()}
                        for g in (('B','R') if stage=='1' else ('R',))}}
        corpora.append(item)
    return {'status':source['status'],'source':str(path),'source_result_sha256':sha(path/'result.json'),
        'selected_recipe':source.get('selected_recipe'),'original_binding_passed_corpora':source.get('original_binding_passed_corpora'),
        'corpora':corpora}

def publish(out,state,details):
    old=study_summary(SOURCE_STUDY)
    write(ROOT/'work/reassessment_05_models.json',{
        'experiment_id':'05','status':state,
        'comparison_type':'Exploratory fixed-recipe model replay; not a causal newer-versus-older comparison.',
        'older_model':{'official_id':'Qwen/Qwen3-0.6B','result':old},
        'newer_model':{'official_id':MODEL_ID,'out':str(out),**details},
        'limits':['The model family, parameter scale, architecture, post-training and native tokenization differ.',
            'The D recipe was selected on the older model development corpus and is transferred unchanged. There is no model-specific recipe search.',
            'The same frozen corpus seeds were previously evaluated with the older model. The replay is exploratory cross-model replication, not a newly untouched confirmatory dataset.',
            'No second-model deletion conclusion is interpretable if its original association gate fails.',
            'Exact association utility and held-out-paraphrase generation remain separate; no privacy or low-FPR guarantee.']})

def preflight(out,model):
    out.mkdir(parents=True,exist_ok=False)
    recipe=fixed_recipe();info=checkpoint_info(model)
    assert info['model_type']=='qwen3_5' and info['text_decoder_layers']>=4
    shutil.copy2(__file__,out/'executed_deletion_model_replay.py')
    shutil.copy2(base.__file__,out/'executed_base_deletion_sequence.py')
    before={'status':'frozen_before_cpu_model_load_or_any_gpu_work','model':info,'fixed_recipe':recipe,
        'recipe_source':str(SOURCE_STUDY/'development_selection.json'),
        'recipe_source_sha256':sha(SOURCE_STUDY/'development_selection.json'),
        'wrapper_sha256':sha(__file__),'base_helper_sha256':sha(base.__file__),
        'development_seed':101,'locked_corpus_and_training_order_seeds':[211,307],
        'protocol':'One fixed-D development association sanity check. If it passes the unchanged gate, replay the two locked corpora once. If it fails, stop without deletion comparisons. No hyperparameter/prompt/corpus tuning.',
        'loss_mask':'Use the model-native chat completion format. Actual completion-mask and six-digit answer-prefix offsets are checked; no assumption of four supervised formatting tokens.',
        'modality':'Official multimodal checkpoint loaded through the installed MLX-LM text wrapper; vision weights are excluded and no images are used.',
        'execution_attribution':'Saurav K. Aryal, with Codex-assisted implementation and execution.'}
    write(out/'replay_protocol.json',before)
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx_lm import load
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.datasets import CompletionsDataset
    import mlx_lm.models.qwen3_5 as architecture
    import mlx_lm.models.gated_delta as delta
    import mlx_lm.tuner.trainer as trainer
    model_object,tokenizer=load(str(model),lazy=True,trust_remote_code=False)
    model_object.freeze()
    for layer in model_object.layers[-4:]:layer.unfreeze()
    trainable=tree_flatten(model_object.trainable_parameters())
    layer_ids=sorted({int(re.search(r'\.layers\.(\d+)\.',key).group(1)) for key,value in trainable})
    assert layer_ids==list(range(len(model_object.layers)-4,len(model_object.layers)))
    assert not any('visual' in key or 'vision' in key for key,value in trainable)
    records=base.make_records(101);pools={g:[r['code'] for r in records if r['group']==g] for g in ('A','B','R')}
    boundaries=[]
    for record in records:
        for view in (0,1,2):
            messages=[{'role':'user','content':base.question(record,view)}]
            prefix=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':record['code']}],tokenize=False)
            prefix_ids=tokenizer.encode(prefix,add_special_tokens=False)
            dataset=CompletionsDataset([{'prompt':base.question(record,view),'completion':record['code']}],tokenizer,'prompt','completion',True)
            tokens,offset=dataset.process(dataset[0]);candidates=pools[record['pool']]
            assert full.startswith(prefix) and tokens[:len(prefix_ids)]==prefix_ids
            lengths={len(tokenizer.encode(code,add_special_tokens=False)) for code in candidates}
            assert len(lengths)==1
            assert all(tokenizer.encode(prefix+code,add_special_tokens=False)==prefix_ids+tokenizer.encode(code,add_special_tokens=False) for code in candidates)
            boundaries.append({'subject':record['subject'],'view':view,'train_mask_offset':offset,
                'answer_prefix_tokens':len(prefix_ids),'mask_to_answer_prefix_delta':len(prefix_ids)-offset,
                'candidate_answer_token_lengths':sorted(lengths),'prefix_and_boundary_match':True})
    architecture_source=Path(architecture.__file__);delta_source=Path(delta.__file__);trainer_source=Path(trainer.__file__)
    result={'status':'passed_cpu_preflight','model':info,'fixed_recipe':recipe,
        'model_forward_calls':0,'model_generation_calls':0,'gradient_calls':0,
        'device_used':str(mx.default_device()),'loaded_model_class':type(model_object).__module__+'.'+type(model_object).__name__,
        'text_decoder_layers':len(model_object.layers),'partial_dense_layer_ids':layer_ids,
        'trainable_tensors':len(trainable),'trainable_parameters':sum(value.size for key,value in trainable),
        'chat_checks':len(boundaries),'all_chat_checks_passed':True,
        'mask_to_answer_prefix_delta_counts':dict(collections.Counter(str(row['mask_to_answer_prefix_delta']) for row in boundaries)),
        'backend_versions':{'mlx':importlib.metadata.version('mlx'),'mlx-lm':importlib.metadata.version('mlx-lm'),'transformers':importlib.metadata.version('transformers'),'python':sys.version},
        'library_source_sha256':{str(path):sha(path) for path in (architecture_source,delta_source,trainer_source)},
        'static_training_path':'GatedDeltaNet passes use_kernel=not self.training; the trainer calls model.train(), selecting the ops-based recurrence for autograd. Actual GPU backprop remains unexecuted at CPU preflight.',
        'text_weight_filter':'The installed Model.sanitize discards vision_tower/model.visual weights and maps model.language_model to language_model.model. Strict lazy text loading succeeded.',
        'limitations':['Lazy CPU loading validates names/shapes and native tokenization, not actual forward/backward numerical compatibility or runtime.','Qwen3.5 hybrid recurrent training may be slower than the earlier attention-only model. The unchanged base helper imposes a900-second training-arm timeout; any failure is retained rather than silently changing libraries or recipes.']}
    write(out/'tokenizer_boundaries.json',boundaries);write(out/'cpu_preflight.json',result)
    del model_object,tokenizer;gc.collect()
    publish(out,'cpu_preflight_complete_gpu_not_started',{'model':info,'preflight':result,'executed_phases':['cpu_preflight']})
    print(json.dumps({'status':result['status'],'layers':layer_ids,'parameters':result['trainable_parameters'],'mask_delta':result['mask_to_answer_prefix_delta_counts']}),flush=True)

def annotate_training(out):
    records=[]
    from safetensors import safe_open
    for path in sorted((out/'experiment').rglob('training_record.json')):
        raw=read(path);config=read(path.parent/'config.json');weights=path.parent/'adapter/adapters.safetensors'
        with safe_open(weights,framework='np') as handle:
            keys=list(handle.keys())
            layer_ids=sorted({int(re.search(r'\.layers\.(\d+)\.',key).group(1)) for key in keys if re.search(r'\.layers\.(\d+)\.',key)})
        records.append({'source':str(path.relative_to(out)),'source_sha256':sha(path),
            'verified_fine_tune_type':config['fine_tune_type'],'verified_num_layers':config['num_layers'],
            'saved_dense_layer_ids':layer_ids,'saved_tensor_count':len(keys),
            'legacy_started_from_field':raw['started_from'],
            'clarification':('Fresh pretrained text component with last-four-layer partial-dense adaptation; not LoRA.' if not config.get('resume_adapter_file') else 'Previous partial-dense adapted weights resumed; fresh optimizer. No past optimizer state is resumed.'),
            'optimizer_steps':raw['optimizer_steps'],'batch_size':raw['batch_size'],'exposures_per_subject':raw['exposures_per_subject']})
    write(out/'training_metadata_clarifications.json',{'records':records,'policy':'Clarifications are separate annotations. Frozen raw training records are not edited.'})

def execute(out,model):
    frozen=read(out/'replay_protocol.json');pre=read(out/'cpu_preflight.json')
    assert frozen['wrapper_sha256']==sha(__file__) and frozen['base_helper_sha256']==sha(base.__file__)
    assert pre['status']=='passed_cpu_preflight' and frozen['model']==checkpoint_info(model)
    if (out/'result.json').exists():raise FileExistsError('Preserve previous outcomes; select a fresh output folder.')
    fixed=fixed_recipe();assert frozen['fixed_recipe']==fixed
    import mlx.core as mx
    if not mx.metal.is_available():raise RuntimeError('Apple GPU unavailable; no silent CPU training fallback.')
    mx.set_default_device(mx.gpu)
    # Protocol-only adaptation; the frozen base training/scoring functions and
    # the source file are not modified. There is exactly one transferred recipe.
    base.RECIPES=[fixed];base.ACTIVE_VARIANT='fixed_D_cross_model_replay'
    original_protocol=base.protocol
    def transferred_protocol(path):
        value=original_protocol(path)
        value.pop('locked_corpus_and_initialization_seeds',None)
        value.update(locked_corpus_and_training_order_seeds=[211,307],
            training='Fixed source-D last-four-layer partial-dense adaptation. Model-native prompt-masked assistant completion; actual mask offsets are recorded. Equal per-subject exposure, batch4; no model-specific recipe search.',
            development_amendment='One frozen-D sanity gate on the new model. Failed binding stops this model; no model-specific tuning or fallback recipe.',
            model_checkpoint=frozen['model'],cross_model_wrapper_sha256=frozen['wrapper_sha256'],
            cross_model_limits='Family, parameter scale, architecture, native tokenizer and post-training differ. Cross-model results are exploratory, not a causal effect of recency.')
        return value
    base.protocol=transferred_protocol
    publish(out,'executing_fixed_recipe_new_model',{'model':frozen['model'],'executed_phases':['cpu_preflight'],'current_phase':'fixed_D_development'})
    started=time.perf_counter();phase='fixed_D_development'
    try:
        selection=base.develop(out/'experiment',model);annotate_training(out)
        if selection['selected_recipe'] is None:
            result={'experiment_id':'05','status':'completed','scientific_status':'inconclusive_original_association_gate',
                'model':frozen['model'],'development':selection,'executed_phases':['cpu_preflight','fixed_D_development'],
                'locked_corpora_executed':0,'deletion_comparison_interpretable':False,
                'decision':'The unchanged D recipe did not establish original associations on this model. No locked deletion analysis or model-specific tuning was run.'}
        else:
            phase='locked_211_307'
            publish(out,'executing_fixed_recipe_new_model',{'model':frozen['model'],'development':selection,'current_phase':phase})
            actual=base.locked(out/'experiment',model);annotate_training(out)
            result={'experiment_id':'05','status':'completed','scientific_status':'fixed_recipe_model_replay_completed',
                'model':frozen['model'],'development':selection,'executed_phases':['cpu_preflight','fixed_D_development','locked_211','locked_307'],
                'locked_corpora_executed':len(actual['corpora']),
                'original_binding_passed_corpora':actual['original_binding_passed_corpora'],
                'result':study_summary(out/'experiment'),
                'decision':'Report every locked outcome and exact/paraphrase utility separately. Failed binding/utility conditions are inconclusive; no privacy guarantee follows.'}
        result.update(seconds=time.perf_counter()-started,wrapper_sha256=sha(__file__),base_helper_sha256=sha(base.__file__),
            metadata_clarification='Separate sidecar records the actual full-type partial-dense mechanism and corrects the inherited fresh-LoRA label without changing frozen raw records.')
        write(out/'result.json',result);publish(out,'completed',result)
        print(json.dumps({'status':result['status'],'scientific_status':result['scientific_status'],'model':MODEL_ID}),flush=True)
    except Exception as exc:
        try:annotate_training(out)
        except Exception:pass
        failure={'experiment_id':'05','status':'compatibility_or_execution_failure','phase':phase,'model':frozen['model'],
            'exception_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'seconds':time.perf_counter()-started,'wrapper_sha256':sha(__file__),
            'decision':'No successful second-model deletion comparison is claimed. Preserve logs/checkpoints; no backend patches, hidden recipe changes or automatic model fallback.'}
        write(out/'result.json',failure);publish(out,'compatibility_or_execution_failure',failure)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','execute'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model-path-file',type=Path,default=MODEL_PATH_FILE)
    args=parser.parse_args();out=args.out.resolve()
    model=Path(args.model_path_file.read_text().strip()).resolve()
    if not model.is_dir():raise FileNotFoundError('The official cached checkpoint is required; this runner does not download weights.')
    if args.stage=='preflight':preflight(out,model)
    else:execute(out,model)

if __name__=='__main__':main()
