# University RTX 3070 check

On October 19, Kaiwen reported a successful CUDA forward/backward and finite-gradient
check on `linux10609`: RTX 3070 8 GB, driver 580.178.04, Python 3.11.17,
PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128. This is user-provided terminal
evidence, not an independently executed V-JEPA measurement. The earlier
`linux10604` NVML mismatch remains a separate host observation.

Runpod's single RTX 5090 request was still queued at the latest user update.
Check its status separately; this university script does not manage Runpod.

## Transfer from the Mac

Use the prepared October 19 source snapshot, including uncommitted scientific
changes, rather than cloning the potentially older GitHub branch. The immutable
archives remain those documented in [the cloud handoff](cloud_pilot_handoff.md).
The new launcher is transferred separately; it does not modify that snapshot.

Run in a **Mac terminal**, with university network/VPN access:

```bash
cd /Users/kaiwendu/Documents/GitHub/V-JEPA-Robust-Overfitting
ssh kdu@linux10609.dc.engr.scu.edu 'mkdir -p "$HOME/Desktop/vjepa-university-20261019"'
scp scripts/university_check.sh \
  runs/pretrained/diagnostic/cloud-pilot-handoff-20261019/vjepa-pilot-source.tar.gz \
  runs/pretrained/diagnostic/cloud-pilot-handoff-20261019/vjepa-pilot-clips.tar.gz \
  runs/pretrained/diagnostic/cloud-pilot-handoff-20261019/manifest.json \
  kdu@linux10609.dc.engr.scu.edu:Desktop/vjepa-university-20261019/
```

The generic university Linux alias selects a workstation; using the specific
hostname returns to the same machine. See the [SCU SSH guide](https://www.scu.edu/engineering/labs--research/labs/engineering-computing-center/connect-to-the-ecc-linux-platform-using-the-command-line-interface/).
Shared home storage does not migrate running processes between hosts.

## Prepare on linux10609

In the existing university SSH session:

```bash
cd "$HOME/Desktop/vjepa-university-20261019"
bash university_check.sh setup
```

The launcher requires the already working `~/.venvs/vjepa` environment. It
verifies archive and extracted-file hashes, reconstructs the project with its
Git history, pins the official upstream revision, constrains the installed
PyTorch versions, runs setup/CPU integration checks, and downloads the official
5.13 GB checkpoint with SHA-256 verification. FFmpeg tests may skip if the
executables are absent; the prepared training tensors need no video decoding.
No full model CUDA check or optimizer update runs during setup.

Allow space for the checkpoint plus dependencies and extracted inputs. The
reported 2.2 TB filesystem free space is shared capacity, not a personal quota;
blank `quota -s` output does not establish unlimited storage. A download or
quota error stops setup and is preserved in its log. Re-running setup checks
existing files and retries an incomplete checkpoint download from the start.

## Run the model correctness check

After `SETUP PASS`, on the same host:

```bash
bash university_check.sh smoke
```

This uses the original Conv3d implementation, FP32, one 16-frame 256-pixel
synthetic clip, the native prediction loss, and the fixed clean target. It
checks loss agreement and gradients; there is no optimizer step. Timeout is
20 minutes, with TERM followed by KILL after 30 seconds if needed. This is a
limit, not a predicted runtime. Console logs and exit status are saved under
`logs/`; successful JSON reports and environment metadata go into fresh
`runs/pretrained/diagnostic/university-cuda.*` directories in the reconstructed
project. An OOM, timeout, or correctness failure is evidence to review, not
permission to change input dimensions, precision, or trainable parameters.

Keep SSH connected for these foreground commands. For a later long-running
pilot, first establish a persistent session on this exact host and test
reconnection. This launcher does not establish one.

Return the setup summary and model-check result or error. A smoke pass does
not establish that AdamW plus PGD fits. Only then proceed to the existing
real-clip attack checks and bounded optimizer/timing pilot; those remain
unexecuted on this host.

## Local validation

The launcher was checked with Bash syntax validation. Its archive pins were
checked against the local handoff files, and its manifest was checked against
the previously reconstructed source/clip snapshot. It has not been executed
on university hardware by the assistant. No model code or study configuration
was changed to prepare this handoff.
