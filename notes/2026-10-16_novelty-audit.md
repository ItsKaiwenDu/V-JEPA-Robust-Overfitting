# Novelty audit — October 16, 2026

## Question searched

Does existing public work combine V-JEPA native prediction-loss attacks,
adversarial fine-tuning of that prediction objective, and longitudinal
robust-overfitting analysis?

The planning-session search found no exact public match. This supports a
focused pilot; it does not prove priority or exclude unpublished work.

## Coverage and limitations

The search used V-JEPA, V-JEPA 2, V-JEPA 2.1, VJEPA, and JEPA with adversarial
training/fine-tuning, attacks, PGD, native prediction loss, and robust
overfitting. It included arXiv-indexed web searches, OpenReview/GitHub queries,
public paper pages/full texts, Meta's repository, and the Poisson Labs study.

Direct arXiv search/API endpoints were unavailable through the tools, so this
was not an exhaustive arXiv database query. Search-index delays and work
using different terminology can leave gaps. The prior ChatGPT history was
context, not independent evidence.

## Closest work

| Work | What overlaps | Boundary for this study |
| --- | --- | --- |
| [Task-Agnostic Attacks Against Vision Foundation Models](https://arxiv.org/abs/2503.03842), CVPR Workshops 2025 | Representation attacks, including I-JEPA | Embedding vulnerability alone is not the contribution |
| [From Pretrain to Pain / TVA](https://arxiv.org/abs/2511.07049), AAAI 2026 | Task-agnostic attacks on video representations | Different from native V-JEPA prediction training and its curves |
| [Robust CLIP](https://arxiv.org/abs/2402.12336), ICML 2024 | Unsupervised adversarial fine-tuning of representations | Label-free robust adaptation is established |
| [ASTrA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17b08a9de93e2accf13429643e7eafdc-Abstract-Conference.html), ICLR 2025 | Adversarial SSL and reported resilience to robust overfitting | Robust-overfitting analysis in SSL is established |
| [Latent Video Prediction for World Modeling](https://arxiv.org/abs/2605.15618v2), revised September 30, 2026 | V-JEPA corruption/occlusion robustness and downstream planning | The revised study does not supply our adversarial fine-tuning experiment |
| [Off-Manifold Refinement](https://arxiv.org/abs/2608.29904), August 30, 2026 | V-JEPA 2.1 surprise-energy gradients | Adapter-space video-generation guidance, not the proposed pixel-attack/training study |
| [CF-JEPA](https://arxiv.org/abs/2610.00727), October 2026 | Robustness and adversarial dynamics terminology | Controllability factorization with gradient reversal, not this PGD fine-tuning question |
| [Poisson Labs V-JEPA 2.1 study](https://poissonlabs.ai/research/vjepa-2-1-robustness/) | Corruption sweep and announced learned-adversary follow-up | Published page still describes the follow-up as future work; no matching public follow-up was found |

## Implementation-relevant sources

- [Official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2): released 300M ViT-L and predictor path.
- [V-JEPA 2.1 checkpoint issue #149](https://github.com/facebookresearch/vjepa2/issues/149): reported distilled-model loading/predictor mismatches; an issue report, not a diagnosis reproduced here.
- [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482): dense/deep supervision and distillation context for a later extension.

## Claim boundary and next check

The proposed contribution concerns measured training dynamics under a specific
native-loss fine-tuning protocol, with attack checks and downstream evidence.
Freezing the target encoder further limits claims to that adaptation setup.

Repeat the search before paid confirmatory runs and before the final report.
If an exact overlapping study is found, halt and show the overlap to the
researcher. Do not silently pivot or claim to be the first. Partial overlap
should be described accurately and evaluated against the actual question.
