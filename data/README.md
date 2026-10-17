# Data

| Directory | Contents | Git policy |
| --- | --- | --- |
| `raw/` | Original UCF101 videos and annotations | Ignored |
| `processed/` | Deterministic decoded clips | Ignored |
| `features/` | Clean frozen-target feature cache | Ignored |
| [manifests/](manifests/README.md) | Small split definitions, grouping, checksums, and provenance | Tracked when prepared |

The directories are empty. No data preparation or downloader has run.

Use the [official UCF101 source](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
and record archive checksums and annotation versions. Follow the
[protocol](../docs/experimental_protocol.md) when constructing splits.

Manifests should identify videos using dataset-relative paths; do not commit
your username, SSH host, absolute data mount, access token, or video bytes.
Machine-specific locations belong in ignored `configs/local.yaml`.

Cache keys must include the source video, preprocessing version, clip/frame
selection, target checkpoint hash, and normalization. Rebuild or invalidate
the cache when any of those change.
