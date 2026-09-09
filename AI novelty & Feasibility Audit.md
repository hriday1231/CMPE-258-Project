# AI Novelty and Feasibility Audit

**Project:** Component-typed model merging for hybrid attention-SSM models  
**Course:** CMPE 258, Fall 2026  
**Date:** 2026-09-08

This is the AI evaluation of the project's novelty, red-ocean risk, and feasibility, pasted
as the assignment requires. It includes the findings that argue against the project.

**Verdict: ADJACENT-OPEN**, after one hostile screening pass. A second pass is scheduled
for week 2 and is a condition of continuing. Section 5 explains why that is not a
formality.

---

## 1. Screening record

This is the tenth candidate considered for this course project. Nine were screened and
killed first, two of them after full proposals had already been written. That record is the
audit's main evidence that the screen is real rather than decorative.

| Candidate | Verdict | Killed by |
|---|---|---|
| Label-free self-supervised anomaly detection for plant disease | OCCUPIED | Frontiers in Plant Science, Aug 2025, identical known/unknown class split |
| Self-supervised bioacoustic threat detection on the edge | OCCUPIED | BioME, `2602.09970`, the same pipeline end to end |
| Retrieval-augmented compliance QA with a graph retriever | OCCUPIED | RefWalk, `2605.29742`, same design with code released |
| Staff-geometry supervision for optical music recognition | OCCUPIED | Alfaro-Contreras et al., Pattern Recognition Letters 158, 2022, from the lab that built the baseline |
| Learned pixel-space prefilter for vision-language models | OCCUPIED on second pass | VQ-TTT `2506.15645` holds the architecture slot; DFC-DIT `1707.09482` has held the objective since 2017 |
| Early-exit inference under label noise | OCCUPIED on second pass | RoG `1901.11300` measured per-block accuracy under label noise in 2019; AMAL `2202.03250` finds the opposite of the proposed fix; the proposed disagreement filter is Kaya's confusion metric, `1810.07052` |

Two candidates in a row survived a first hostile screen and were killed by a second.

**The method change that produced this candidate.** The first eight were searches for
genuinely empty research areas. The correction was to target a specific corner adjacent to
a crowded area, inheriting its baselines, metrics, and reviewer familiarity, rather than
open water where no baselines exist and no one is reading.

---

## 2. Red-ocean assessment

| Area | Status | Consequence |
|---|---|---|
| Model merging: task arithmetic, TIES, DARE, model soups | Red, crowded | We claim no new merging algorithm. Baselines and metrics are inherited. |
| Hybrid attention-SSM architectures | New, contested | Checkpoints and a published functional mechanism are available |
| Merging combined with quantization, adversarial robustness, OOD, long-tail | Occupied | Excluded from scope |
| Merging combined with label noise | Occupied | `2506.14126` owns the mechanism, `2508.01148` owns the remedy |
| Pre-merge success prediction | Occupied | `2601.22285`, 28 metrics across five methods |
| Training-time interventions to improve mergeability | Occupied | `2505.22389`, `2607.24465`, `2508.01148` |
| **Component-typed merging on parallel hybrids** | **Open** | The project |

The surrounding area is unambiguously red. The project does not compete in it.

---

## 3. Why this specific corner is empty

This is a structural argument rather than an absence of evidence, which is the distinction
the nine previous candidates could not make.

Every merging method scales task vectors along depth or by per-tensor magnitude. In a
parallel hybrid such as Falcon-H1, attention heads and Mamba-2 heads sit inside the same
block, so a per-layer coefficient must apply the same value to both. Depth-wise scaling is
structurally incapable of expressing this axis. Pure transformers have only one mixer type,
so the field has never needed it and its benchmarks cannot represent it.

The functional mechanism is pre-published in a different space. `2609.04434`, posted
2026-09-03, shows on Falcon-H1 that exact retrieval survives only through attention and
collapses through recurrence, while language and persona survive recurrence. That paper
concerns caches rather than weights and names merging nowhere.

---

## 4. What remains after subtracting the nearest prior work

`2604.22127`, "Where Should LoRA Go?", is the nearest miss. It uses the same two
checkpoints on the same component-type axis, for adaptation. It reports that adapting the
recurrent backbone is destructive in sequential hybrids but constructive in parallel ones,
and that hybrid topology determines adaptation response. It never mentions merging or task
arithmetic.

After subtraction, three things remain: the component-type axis applied to weight-space
merging rather than adapter placement; the functional dissociation of merge damage that
`2609.04434` predicts but never tests on parameters; and the parameterization question,
since `A_log`, `dt_bias`, depthwise `conv1d`, and `D` are constrained or reparameterized
coordinates that every merging method averages linearly and naively.

Searches run, deliberately in vocabularies the proposer would not use: model merging Mamba
or state space; task vector Mamba; merging recurrent networks; model soup and weight
averaging for SSM experts; linear mode connectivity and loss barriers in Mamba; merging
hybrid attention Mamba across Falcon-H1, Zamba, and Jamba; Mamba continual learning
parameter interference; which parameters to merge in normalization layers.

---

## 5. Honest weaknesses

1. **One hostile pass, not two.** Two prior candidates in this same project survived a
   first screen and died on a second, one of them killed by a paper from the same author
   group as work the first screen had cited approvingly. A second independent pass runs in
   week 2 and is a condition of continuing.
2. **Scoop risk is real and dated.** A group is already fine-tuning these exact checkpoints
   along this axis. If they publish the merging extension, the project dies. The week-1
   gate is front-loaded for that reason.
3. **Half the training-side remedy may be a library default.** The standard Mamba LoRA
   recipe already targets only `x_proj`, `embeddings`, `in_proj`, and `out_proj`, excluding
   `A_log`, `dt_bias`, and `conv1d`. That applies to LoRA. Task vectors in the merging
   literature come from full fine-tuning, where all parameters move, so the project uses
   full fine-tuning. This is disclosed rather than glossed over.
4. **The ablation grid is oversized** for the compute budget and must be pruned in week 1
   to a single architecture pair.
5. **Licensing.** Falcon-H1 checkpoints are released under the Falcon-LLM License, not
   Apache 2.0. The acceptable-use terms must be read before releasing an artifact.

---

## 6. Feasibility

| Constraint | Status |
|---|---|
| No physical hardware | Satisfied. Public checkpoints and datasets only. |
| 16 GB VRAM | Falcon-H1-0.5B full fine-tuning in bfloat16 is roughly 6 GB of optimizer state plus activations. Fits. |
| Blackwell sm_120 toolchain | Not a constraint here, verified by reading installed source. `modeling_mamba2.py` falls through to a pure PyTorch path automatically when fast kernels are absent, so no `mamba-ssm` build is required. |
| 200 GPU-hour cap | Estimated 90 to 130 GPU-hours. Merging itself is free. |
| Local to cloud by config change | `configs/local.yaml` and `configs/runpod.yaml` differ only in device, batch size, workers, and paths. |
| Data public and downloadable today | GSM8K, MBPP, IFEval, BoolQ, SST-2, AG News, RULER. Sizes and licences marked for verification. |
| Team of 2, 11 weeks | Two workstreams that can falsify each other. |
| Open-source artifact | The component-typed merge library plus released task vectors, so the analysis reproduces without training. |
