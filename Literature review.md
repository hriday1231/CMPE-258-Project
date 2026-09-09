# Literature and SOTA Survey

**Project:** Component-typed model merging for hybrid attention-SSM models  
**Course:** CMPE 258, Fall 2026  

Thirteen works and models relevant to the project, all with resolved arXiv identifiers.

---

### 1. What Attention Recalls and Recurrence Controls in Hybrid Language Models
`arXiv 2609.04434`, Findings of EMNLP 2026

The paper that makes this project askable. It introduces two cache-level interventions on Qwen3.5
and Falcon-H1: split-prefill, which keeps only the KV cache or only the recurrent state from a
prefilled context, and state-swap, which pairs the KV cache from one context with the recurrent
state from another in a single forward pass. The two channels split sharply by function. Exact
retrieval survives only through attention, at 64 to 98 percent of full accuracy, and collapses to
zero through recurrence, while output language and persona reverse the pattern and survive
recurrence. Crucially for us, it operates on caches rather than weights and never mentions
merging. It supplies the functional prediction we test in weight space.

### 2. Where Should LoRA Go? Component-Type Placement in Hybrid Language Models
`arXiv 2604.22127` (2026)

The nearest prior work, and our clearest scoop risk. It studies component-type LoRA placement
across the same two architectures we use, Qwen3.5-0.8B as a sequential hybrid and Falcon-H1-0.5B
as a parallel one, fine-tuned on three domains and evaluated on five benchmarks. It reports that
the attention pathway, despite being the minority component, consistently outperforms full-model
adaptation with 5 to 10 times fewer trainable parameters. It takes the adaptation half of the
component-type axis and never mentions merging or task arithmetic, which leaves the merging half
open. A group is demonstrably working on these checkpoints along this axis, so we front-load our
week-1 gate accordingly.

### 3. Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance
`arXiv 2507.22448` (2025)

The primary architecture under study. Falcon-H1 adopts a parallel hybrid design that combines
Transformer attention with state space models, released in configurations from 0.5B upward in base
and instruction-tuned variants. The parallel arrangement is what creates the gap we target:
attention heads and Mamba-2 heads sit at the same depth inside one block, so a per-layer merging
coefficient cannot address them separately.

### 4. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
`arXiv 2405.21060`, ICML 2024

Introduces the state space duality framework connecting SSMs and attention variants through
structured semiseparable matrices, and defines Mamba-2, the state-space layer used inside
Falcon-H1. Relevant to us for its parameterization: the state-space parameters are not free
coordinates. `A_log` is a log-parameterized decay, `dt_bias` feeds a softplus, `conv1d` is
depthwise, and `D` is a skip term. Every merging method averages all of them linearly regardless,
which is an assumption we make explicit and ablate.

### 5. Editing Models with Task Arithmetic
`arXiv 2212.04089`, ICLR 2023

Introduces the task vector, a direction in weight space obtained by subtracting pre-trained
weights from fine-tuned weights, and shows that task vectors can be negated and added to steer
model behavior. This is the formulation everything downstream builds on, and its single global
scaling coefficient is our primary baseline.

### 6. TIES-Merging: Resolving Interference When Merging Models
`arXiv 2306.01708`, NeurIPS 2023

Addresses interference between the parameters of different models, which existing merging methods
ignore and which causes large performance drops. Trims low-magnitude entries, elects a sign, and
merges disjointly. It operates per tensor by magnitude, not by mixer type. Available in mergekit
and used here as a baseline.

### 7. Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)
`arXiv 2311.03099`, ICML 2024

Shows that most delta parameters can be set to zero without affecting a fine-tuned model's
abilities, by randomly dropping delta parameters at rate p and rescaling the remainder by 1/(1-p),
then uses this as a plug-in to sparsify several models before fusing them. Establishes that task
vectors are highly redundant. Also per tensor, also blind to component type. Second merging
baseline.

### 8. LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging
`arXiv 2410.17146`, ICLR 2025

Scales parameter updates linearly according to their layer depth, as a post-training edit that
preserves pre-trained generalization while retaining fine-tuned task performance. This is our
primary null hypothesis. If per-layer depth coefficients match our component-typed rule at matched
free-scalar count, the component axis carries no additional information and our central claim
fails.

### 9. Sens-Merging: Sensitivity-Guided Parameter Balancing for Merging Large Language Models
`arXiv 2502.12420` (2025)

Observes that existing task-vector merging methods apply uniform coefficients across all
parameters and overlook varying parameter importance, and instead adjusts coefficients using
parameter sensitivity at both task-specific and cross-task levels. This is our second null. If
sensitivity-based weighting discovers the attention versus state-space split on its own, without
being told the component types, then a hand-specified component rule is a weaker version of
something already learned.

### 10. From Memorization to Parameter Interference: How Overtraining Experts Harms Model Merging
`arXiv 2506.14126`, ICML 2026

Identifies memorization during expert fine-tuning as a source of negative parameter interference
when checkpoints are later merged, in the now-standard pipeline of pre-train, fine-tune, then
merge. Documented here so we do not accidentally enter it: merging combined with overtraining and
label noise is closed territory.

### 11. DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging
`arXiv 2508.01148` (2025)

Argues that merging methods are usually evaluated on benchmark suites favorable to merging, and
identifies two source-model properties that are particularly harmful: disparities in task vector
norms, and low source-model confidence. It pre-conditions source models by distillation before
merging. This closes the norm-normalization remedy and the broader pre-merging intervention
category for us.

### 12. Demystifying Mergeability: Interpretable Properties to Predict Model Merging Success
`arXiv 2601.22285` (2026)

Shows with an architecture-agnostic framework that mergeability depends on both the merging method
and the partner tasks rather than being an intrinsic property of the models, using L1-regularized
linear optimization over interpretable pairwise metrics across five merging methods, and finds
gradient alignment metrics to be the most consistent compatibility signal. Closes the cheap
merge-outcome predictor framing.

### 13. S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models
`arXiv 2604.01168` (2026)

Tunes a single initial state matrix per recurrent layer while freezing all model weights, with
zero inference overhead, and reports that on Falcon-H1-7B it reaches parity with LoRA while
explicitly requiring no weight merging. This is a direct threat to our motivation and we answer it
head-on: S0 tuning still swaps a per-task artifact at inference time and does not produce one
model that serves several tasks concurrently, which is the thing merging is for.
