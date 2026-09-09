# The Merge Axis Nobody Can See: Component-Typed Model Merging for Hybrid Attention-SSM Models

**Course:** CMPE 258, Deep Learning, Fall 2026, Prof. Kaikai Liu  
**Team:** Hriday Ampavatina, Pramod Yadav  
**Selected track:** Option 1, Modern Deep Learning Pipeline (Training + Deployment)  

## Abstract

Model merging combines several fine-tuned experts into a single model by averaging their
task vectors, the difference between fine-tuned and base weights. Every published method
scales those vectors along the depth axis, one coefficient per layer, or by per-tensor
magnitude. Parallel hybrid language models break that assumption. In Falcon-H1, attention
heads and Mamba-2 state-space heads sit in parallel inside the same block, so a per-layer
coefficient must apply the same value to both. Depth-wise scaling is therefore structurally
incapable of treating the two mixer types differently, and because pure transformers have
only one mixer type, the field has never needed this axis and its benchmarks cannot express
it.

Recent cache-space analysis of hybrid models shows that exact retrieval survives only
through the attention pathway and collapses through recurrence, while language and persona
survive recurrence. That work concerns activations rather than weights. We test whether
merge damage is dissociated the same way in weight space, and whether a component-typed
merge rule, one coefficient per mixer type, recovers accuracy that a single global scaling
factor cannot, at matched tuned-scalar count.

The project trains component-restricted experts, introduces a state-drift penalty on the
state-space dynamics parameters during expert training, and evaluates against two nulls
that can invalidate the claim: depth-only per-layer scaling and sensitivity-based merging,
both at matched free-scalar count. A pure-attention control model is included, where the
effect must vanish.

## Repository

https://github.com/hriday1231/CMPE-258-Project

## Deliverables

| Deliverable | File |
|---|---|
| A. Repository and README | this file |
| B. Literature and SOTA survey | `Literature review.md` |
| C. Project proposal | `CMPE258_Project_Proposal.docx`, also submitted to Canvas |
| D. AI novelty and feasibility audit | `AI novelty & Feasibility Audit.md` |
