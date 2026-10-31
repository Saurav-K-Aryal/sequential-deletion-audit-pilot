# Sequential deletion audit pilot

This artifact accompanies an exploratory ACM single-column manuscript by Saurav K. Aryal, Howard University. It contains completed synthetic-association sequences for Qwen3-0.6B and an exploratory fixed-recipe Qwen3.5-0.8B replay, the failed primary-model development recipes, raw scores, a standard-Python verifier, and figure-generation code. Repository: https://github.com/Saurav-K-Aryal/sequential-deletion-audit-pilot.

## Results

Both original adapted models learn all 24 exact subject-code associations. After two retained-only continuation stages, each produces 0/16 deleted codes under the exact greedy extraction prompt, yet ranks the correct code first among eight known candidates for 11/16 deleted subjects. Current-checkpoint association AUROC is 0.8164 and 0.8477. Fresh retained-data references obtain 0/16 exact extractions, 3/16 and 2/16 candidate rankings, and association AUROC 0.4844 and 0.5625. All arms retain 8/8 exact associations in each remaining group at both stages; held-out-paraphrase generation is not matched. These descriptive observations support multidimensional auditing, not a new unlearning method or a privacy guarantee.

For Qwen3.5-0.8B, the unchanged recipe gives 1/16 and 0/16 final exact extractions, 8/16 and 7/16 candidate rankings, and association AUROC 0.7305 and 0.6875; fresh-reference AUROC is 0.5234 and 0.5273. Exact R associations remain 8/8, while warm paraphrase generation is 6/8 and 7/8 versus 8/8 for fresh references. These same corpus/order seeds were already examined with the primary model. Family, size, architecture, post-training, trainable parameters and native formatting change together, so this does not identify a causal model-age effect.

The successful development recipe updates dense weights in the final four decoder blocks. Three tested LoRA recipes fail the original association gate; they remain visible in the development record. `metadata_clarification.json` verifies the actual dense-layer mechanism and explains inherited generic LoRA labels without rewriting the original records. Historical-snapshot attacks have a separate access assumption. Codes and subjects are fictional, and the protected inclusion unit is a subject-code association; matched untrained controls use the same answer inventory.

## Check the saved results without a model

From the repository root, using standard Python 3:

```sh
python3 verify_results.py --project-root . --output offline_verification.json
```

For each completed two-corpus study, this recomputes 1,344 saved score rows, 60 attack settings, retained-utility gates, shared-code inventory checks, and 448 actual output-suppression control rows. It checks arithmetic and recorded text, without network access, model inference, or training. The primary evidence is under `runs/reassessment-05-binding-v2-20261031/`; earlier failed A/B development is under `runs/reassessment-05-sequential-20261031/`. The completed second model is under `runs/reassessment-05-qwen35-08-fixedD-recovered-20261031/experiment/`. Check it separately:

```sh
python3 verify_results.py --study runs/reassessment-05-qwen35-08-fixedD-recovered-20261031
```

Its `recovery_protocol.json` records an external tool interruption: completed seed-211 and seed-307-original artifacts were hash-checked and reused; the unfinished seed-307 first reference restarted fresh with the same fixed recipe before the remaining stages. No outcome-based tuning was performed. The preserved CPU readback additionally checks 1,456 rows including development and 11 training artifacts. Weights are excluded. `PUBLICATION_PROVENANCE.json` records source/public hashes and any path-only normalization applied during release.

## Reproduce adaptation and deletion updates

The tested environment uses Python 3.14 on macOS arm64, MLX 0.32.4, and MLX-LM 0.32.2. Retraining requires Apple Silicon and a separately downloaded official `Qwen/Qwen3-0.6B` checkpoint. The experimental revision is `c1899de289a04d12100db370d81485cdf75e47ca`; preserve that revision when comparing runs. The study was run on a shared host, so it does not establish exact 48 GB laptop memory performance.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Set this to the local directory containing the official model files.
MODEL_DIR="/absolute/path/to/cached/Qwen3-0.6B"
# Each development destination must be new; prior evidence is preserved.
python -m pilots.reassessment.deletion_sequence --stage development --variant original_lora --model "$MODEL_DIR" --out runs/reproduced_original_lora
python -m pilots.reassessment.deletion_sequence --stage development --variant binding_repair_v2 --model "$MODEL_DIR" --out runs/reproduced_binding_v2
python -m pilots.reassessment.deletion_sequence --stage locked --variant binding_repair_v2 --model "$MODEL_DIR" --out runs/reproduced_binding_v2
python verify_results.py --study runs/reproduced_binding_v2
```

The first command records the A/B development attempts; it stops without deletion sequences if neither binds. The next development command tests C/D in their documented order, then the locked command evaluates the two fixed corpus/order seeds with the selected recipe. If the original association gate fails, the run remains inconclusive. Do not tune on a locked result or overwrite observed data. Counts can change across software/hardware revisions; retain the new configurations, source hashes, and logs.

## Reproduce the fixed-recipe second model

Obtain the official `Qwen/Qwen3.5-0.8B` snapshot `2fc06364715b967f1860aea9cf38778875588b17` separately. From the same environment and repository root:

```sh
# Write a path file for the separately cached official checkpoint.
printf '%s\n' "/absolute/path/to/cached/Qwen3.5-0.8B" > model_path.txt
python -m pilots.reassessment.deletion_model_replay --stage preflight --model-path-file model_path.txt --out runs/reproduced_qwen35
python -m pilots.reassessment.deletion_model_replay --stage execute --model-path-file model_path.txt --out runs/reproduced_qwen35
python verify_results.py --study runs/reproduced_qwen35
```

The wrapper reads the frozen D selection from `runs/reassessment-05-binding-v2-20261031/development_selection.json`; it performs no new recipe search. Preflight checks the native text model and token boundaries before GPU execution. The public recovery driver and frozen execution snapshots document the historical interruption; a fresh reproduction uses the ordinary wrapper rather than requiring an interrupted run or unpublished weights.

## Regenerate figures and the results table

With NumPy and matplotlib installed:

```sh
python generate_figures.py --project-root . --newer-study runs/reassessment-05-qwen35-08-fixedD-recovered-20261031
```

This reads completed recorded results, writes `derived_summary.json` and `source_manifest.json`, and regenerates two PNG figures (plus optional vector copies). If an extracted `draft.tex` is present beside the generator, it also refreshes its marked results table. It does not invoke a model. The second-model parent and its canonical experiment-result hash must match the passed readback; unfinished or unreviewed runs are excluded. AUROC markers represent distinct fixed conditions, not an interpolated trajectory.

## Overleaf

In the public repository, download and unpack `overleaf.zip`. Upload its `draft.tex`, `refs.bib`, and `figures/` folder. These assets are also present directly in the local manuscript folder. Select `draft.tex` as the main document and use pdfLaTeX with BibTeX. The manuscript uses `acmart` with `manuscript,screen,nonacm`. `references.bib` is an identical compatibility copy; the manuscript reads only `refs.bib`. No extra `.tex` inputs are required. A compiled PDF preview is provided separately.

The manuscript has Abstract, Introduction and Background, Methods, Results, Future Work, and Conclusion sections. Future work develops the associated proposal's positive-control readiness, economical sequential audits, explicit attacker access, and matched utility evaluation. Code, execution, and manuscript preparation used Codex assistance under the author's direction.

## Preprints.org classification

Subject: **Computer Science and Mathematics**. Topic within Subject: **Security Systems**.

Optional MDPI Topic: [Challenges and Solutions in Large Language Models](https://www.mdpi.com/topics/B712V62UH1). The detailed classification is in `PREPRINTS_METADATA.txt`. These metadata do not imply a submission.
