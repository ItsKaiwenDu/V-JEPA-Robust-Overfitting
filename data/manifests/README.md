# Dataset manifests

The first prepared manifests are in `ucf101-split1-v1/`: 2,020 training,
505 validation, and 505 test videos with disjoint source groups. The
metadata records source hashes, sampling recipe, and held-out groups.
Full video-content integrity remains unchecked. See
[the preparation guide](../../docs/data_and_attacks.md).

Use `train.csv`, `validation.csv`, and `test.csv`, plus `metadata.json` for the
split recipe and checksums. The minimum per-video fields are:

```text
video_id,relative_path,class_id,source_group_id
```

`source_group_id` includes the class and original group identifier so group
numbers from different classes are not conflated. Record exact temporal
sampling and preprocessing in the preparation metadata or an associated
clip manifest. File paths are relative to the dataset root.

Before freezing manifests:

- Confirm official split-1 membership and hold out validation groups first.
- Exclude all validation-group videos from the training candidate pool.
- Check video IDs, source groups, and content duplicates across splits.
- Verify class counts and log unreadable/excluded videos without silently
  replacing them differently for different training seeds.
- Record manifest seed 20261016 and SHA-256 hashes; reuse the files for all seeds.

Any later change creates a new protocol/manifest version. Keep per-machine
path maps in `*.local.*` files, which are ignored.

The October 20 [staged proposal](../../docs/budget_study_plan.md) is exported
in `ucf101-staged-20x5-v1/`, with seed 20261020 and exact replay against the
proposal. Original CSVs remain unchanged. All 100 training and 20 exploration
clips passed deterministic re-decoding and exact-hash duplicate checks; the
20-clip monitor reuses training pixels. Confirmation/test metadata is present,
but their pixels remain unopened. Near-duplicate review and protocol freezing
are pending. See [staged preparation](../../docs/staged_data_and_evaluation.md).
