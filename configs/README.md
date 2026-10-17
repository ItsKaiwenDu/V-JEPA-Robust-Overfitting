# Experiment configurations

- [study.yaml](study.yaml) records the agreed research scope and draft numerical settings.
- [local.example.yaml](local.example.yaml) documents machine-specific paths. A local copy named `local.yaml` is ignored by Git.

These files are specifications for the upcoming implementation. There is no
training runner or configuration-override mechanism yet.

Null fields represent genuinely unresolved choices or provenance, including
the upstream revision, checkpoint hash, exact temporal preprocessing,
microbatch/precision, and evaluation step sizes and mask seeds. Resolve them
with the pilot, record the rationale in [the decision log](../docs/decisions.md),
and freeze a versioned protocol before confirmatory runs. A syntactically
valid YAML file is not evidence that an experiment is ready.

The future runner must save a fully resolved configuration with each run;
the current root configuration must not silently change an existing run.
