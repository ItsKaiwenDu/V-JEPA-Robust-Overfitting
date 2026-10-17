# Dataset manifests

No manifests exist yet. The data-preparation implementation will write the
small reviewable definitions here and keep the videos elsewhere.

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
