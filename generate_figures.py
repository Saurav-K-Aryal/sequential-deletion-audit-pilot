"""Regenerate paper05 figures/tables from immutable completed pilot artifacts.

Run with Python3 and matplotlib. No model inference or training is performed.
The optional newer-model study is included only after its full raw readback passes.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    default_root=next((p for p in (HERE,*HERE.parents) if (p/'runs/reassessment-05-binding-v2-20261031').is_dir()),HERE)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,default=default_root);parser.add_argument('--newer-study',type=Path,help='Explicit completed second-model parent, including any recovery lineage');args=parser.parse_args();root=args.project_root.resolve()
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    studies=[('Qwen3-0.6B',root/'runs/reassessment-05-binding-v2-20261031')]
    newer=(args.newer_study or root/'runs/reassessment-05-qwen35-08-fixedD-20261031').resolve();review=root/'work/reassessment_05_qwen35_final_readback.json'
    include_newer=bool((newer/'result.json').exists() and review.exists() and read(review).get('status')=='passed' and read(newer/'result.json').get('status')=='completed')
    if include_newer:
        reviewed=read(review)
        assert (reviewed.get('study_relative_to_project')==str(newer.relative_to(root)) if reviewed.get('study_relative_to_project') else reviewed.get('study')==str(newer)), 'The passed readback must identify this exact second-model parent.'
        assert reviewed.get('canonical_study_result_sha256')==sha(newer/'experiment/result.json'), 'The passed readback must match the completed study result.'
        studies.append(('Qwen3.5-0.8B',newer/'experiment'))
    source_paths=[root/'runs/reassessment-05-sequential-20261031/development_selection.json',root/'runs/reassessment-05-binding-v2-20261031/development_selection.json']
    summaries=[]
    for model,study in studies:
        data=read(study/'result.json');assert data['status']=='completed';source_paths.append(study/'result.json')
        for corpus in data['corpora']:
            assert corpus['status']=='completed' and corpus['original_gate']['passed']
            record={'model':model,'seed':corpus['seed'],'original_binding':{key:corpus['original_gate'][key] for key in ('exact_members','paraphrase_members')},'stages':{}}
            for stage,arms in corpus['stages'].items():
                record['stages'][stage]={}
                for name,arm in arms.items():
                    forgotten=['A'] if stage=='1' else ['A','B'];retained=['B','R'] if stage=='1' else ['R']
                    record['stages'][stage][name]={'extractions':arm['forgotten_exact_extractions'],'forgotten_n':arm['forgotten_subjects'],'exact_top1':sum(arm['binding'][group]['exact']['strict_top1_correct'] for group in forgotten),'current_binding_auc':arm['attacks']['raw_current_binding']['exploratory_auroc'],'snapshot_delta_auc':arm['attacks']['snapshot_delta']['exploratory_auroc'],'retained_exact_top1':{g:arm['binding'][g]['exact']['strict_top1_correct'] for g in retained},'retained_generation':{g:{v:arm['binding'][g][v]['exact_extraction_correct'] for v in ('exact','held_out_paraphrase')} for g in retained}}
            summaries.append(record)
            source_paths.extend(sorted((study/f"locked_{corpus['seed']}").rglob('scores.json')))
    if include_newer:source_paths.extend([newer/'replay_protocol.json',review])
    manifest={'generator_sha256':sha(Path(__file__)),'newer_model_included_after_complete_readback':include_newer,'sources':[{'path':str(path.relative_to(root)),'sha256':sha(path)} for path in source_paths]}
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/'derived_summary.json').write_text(json.dumps({'scope':'Descriptive paired synthetic-association pilots; not population inference.','studies':summaries},indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':10})
    x=np.arange(len(summaries));labels=[r['model'].replace('-','\n')+f"\nseed {r['seed']}" for r in summaries]
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.25),layout='constrained');width=.24
    axs[0].bar(x-width,[r['stages']['2']['warm_2']['extractions'] for r in summaries],width,label='Warm: greedy extraction',color='#4c78a8')
    axs[0].bar(x,[r['stages']['2']['warm_2']['exact_top1'] for r in summaries],width,label='Warm: candidate top-1',color='#e69f00')
    axs[0].bar(x+width,[r['stages']['2']['reference_2']['exact_top1'] for r in summaries],width,label='Fresh: candidate top-1',color='#009e73')
    axs[0].set(ylabel='Correct deleted associations /16',ylim=(0,17),yticks=[0,4,8,12,16],title='(a) Final deletion: output versus association')
    for arm,label,color,marker in [('original','Original','#777777','o'),('reference_2','Fresh R-only','#009e73','s'),('warm_2','Warm R-only','#e69f00','^')]:
        axs[1].scatter(x,[r['stages']['2'][arm]['current_binding_auc'] for r in summaries],marker=marker,color=color,label=label,s=35)
    axs[1].axhline(.5,ls='--',color='#999999',lw=.8);axs[1].set(ylabel='Current-logit association AUROC',ylim=(0,1.05),yticks=[0,.25,.5,.75,1],title='(b) Subject-paired current-checkpoint score')
    for ax in axs:ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='lower left' if ax is axs[1] else 'upper left',fontsize=7,frameon=False)
    fig.savefig(figures/'deletion_diagnostics.pdf',bbox_inches='tight');fig.savefig(figures/'deletion_diagnostics.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,3.1),layout='constrained')
    for offset,arm,label,color in [(-width,'original','Original','#777777'),(0,'reference_2','Fresh R-only','#009e73'),(width,'warm_2','Warm R-only','#e69f00')]:
        ax.bar(x+offset,[r['stages']['2'][arm]['retained_generation']['R']['held_out_paraphrase'] for r in summaries],width,label=label,color=color)
    all_exact={r['stages']['2'][arm]['retained_exact_top1']['R'] for r in summaries for arm in ('original','reference_2','warm_2')}
    assert all_exact=={8},'Update the utility annotation rather than silently drawing a false ceiling.'
    ax.axhline(8,ls='--',lw=.9,color='#4c78a8',label='All arms: exact R association top-1 =8/8')
    ax.set(ylabel='R paraphrase greedy extraction /8',ylim=(0,8.8),yticks=range(0,9,2));ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(0,1.24),fontsize=8,frameon=False,ncol=2)
    fig.savefig(figures/'retained_utility.pdf',bbox_inches='tight');fig.savefig(figures/'retained_utility.png',dpi=180,bbox_inches='tight');plt.close(fig)
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Model / seed & Final arm & Extraction & Top-1 & R para. & Snapshot AUROC \\',r'\midrule']
    for record in summaries:
        for name,label in [('original','Original'),('reference_2','Fresh R'),('warm_2','Warm R')]:
            a=record['stages']['2'][name];name_model=record['model'].replace('Qwen','')+f" / {record['seed']}"
            lines.append(f"{name_model} & {label} & {a['extractions']}/16 & {a['exact_top1']}/16 & {a['retained_generation']['R']['held_out_paraphrase']}/8 & {a['snapshot_delta_auc']:.3f}"+r' \\')
        lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}']);table_text='\n'.join(lines)+'\n'
    if include_newer:
        records=[r for r in summaries if r['model']=='Qwen3.5-0.8B'];sentences=[]
        for r in records:
            a=r['stages']['2']['warm_2'];ref=r['stages']['2']['reference_2'];sentences.append(f"Seed {r['seed']} ended with {a['extractions']}/16 greedy extractions and {a['exact_top1']}/16 candidate top-1 associations; current-binding AUROC was {a['current_binding_auc']:.4f}, versus {ref['current_binding_auc']:.4f} for its fresh reference.")
        text=r'\paragraph{Fixed-recipe replay on a second model.} The Qwen3.5-0.8B development sanity check passed 24/24 exact and 24/24 paraphrase associations before the fixed-corpus sequences. Its originals achieved 24/24 exact associations and 23/24 and 24/24 held-out-wording associations, respectively. '+ ' '.join(sentences)+r' Both warm models and fresh references retained 8/8 exact R associations. Warm R paraphrase generation was 6/8 and 7/8, versus 8/8 for both fresh references, so broader utility remains unmatched. All displayed final arms passed the exact R utility gate. This is exploratory replication on the same synthetic corpora, with family, parameter scale, architecture, post-training, trainable parameter count and native supervised-formatting offsets changing together. It does not isolate an effect of model age. The newer-model training wrapper has two formatting tokens before the answer, compared with four for Qwen3; native masked training losses are therefore not comparable units.'
    else:text=r'% The second-model sequence is deliberately excluded until full completion and raw-artifact review.'
    draft_path=HERE/'draft.tex'
    if draft_path.exists():
        draft=draft_path.read_text()
        for marker,content in [('RESULTS_TABLE',table_text),('SECOND_MODEL',text+'\n')]:
            pattern=r'(% AUTO_'+marker+r'_BEGIN\n).*?(% AUTO_'+marker+r'_END)'
            draft,n=re.subn(pattern,lambda m:m.group(1)+content+m.group(2),draft,flags=re.S)
            assert n==1,'Expected exactly one generated manuscript block.'
        draft_path.write_text(draft)
    print(json.dumps({'status':'generated','corpora':len(summaries),'newer_model_included':include_newer,'source_files':len(source_paths)}))

if __name__=='__main__':main()
