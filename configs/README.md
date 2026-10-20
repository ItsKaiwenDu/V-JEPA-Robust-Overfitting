# Experiment configurations

- [study.yaml](study.yaml) records the agreed research scope and draft numerical settings.
- [compute_plan.yaml](compute_plan.yaml) is a separate evaluation/budget proposal consumed only by the planning calculator; it does not freeze or override the study.
- [local.example.yaml](local.example.yaml) documents machine-specific paths. A local copy named `local.yaml` is ignored by Git.

These files are specifications for the upcoming implementation. There is no
training runner or configuration-override mechanism yet.

Null fields represent genuinely unresolved choices or provenance, including
microbatch/precision and evaluation step sizes and mask seeds. Checkpoint and
manifest hashes plus a pilot-tested preprocessing recipe are now recorded;
the complete study remains a draft. Resolve remaining choices
with the pilot, record the rationale in [the decision log](../docs/decisions.md),
and freeze a versioned protocol before confirmatory runs. A syntactically
valid YAML file is not evidence that an experiment is ready.

The upstream revision and native normalization/mask-token details are now
recorded. The smoke-test command uses the fixed ViT-L/16, 16-frame,
256-square integration described in [its guide](../docs/smoke_test.md);
it is not a general runner that consumes all settings in `study.yaml`.

The future runner must save a fully resolved configuration with each run;
the current root configuration must not silently change an existing run.
