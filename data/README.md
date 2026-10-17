# Data

| Directory | Contents | Git policy |
| --- | --- | --- |
| `raw/` | Original UCF101 videos and annotations | Ignored |
| `processed/` | Deterministic decoded clips | Ignored |
| `features/` | Clean frozen-target feature cache | Ignored |
| [manifests/](manifests/README.md) | Small split definitions, grouping, checksums, and provenance | Tracked when prepared |

Grouped manifests are prepared in `manifests/ucf101-split1-v1/`. Official
archives and eight extracted training videos are stored locally under
`raw/`, with eight deterministic pilot clips under `processed/pilot-train-v1/`.
Large assets remain ignored. Only those eight training videos have been
decoded; complete dataset integrity and clean-target feature caching remain pending.

Use the [official UCF101 source](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
and record archive checksums and annotation versions. Follow the
[protocol](../docs/experimental_protocol.md) when constructing splits.

Manifests should identify videos using dataset-relative paths; do not commit
your username, SSH host, absolute data mount, access token, or video bytes.
Machine-specific locations belong in ignored `configs/local.yaml`.

Cache keys must include the source video, preprocessing version, clip/frame
selection, target checkpoint hash, and normalization. Rebuild or invalidate
the cache when any of those change.
