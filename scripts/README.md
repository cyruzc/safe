Experiment and utility scripts live in this directory.

Current paper-oriented entry points:
- `run_one_group.sh`: run one experiment group with seed `42`
- `run_five_group_matrix.sh`: run the 5-group matrix with seeds `27 123 456 789 2024`

Default training hyper-parameters baked into the wrappers:
- `batch-size=16`
- `lr=5e-4`
- `weight-decay=1e-4`
- `loss-type=focal`
- `scheduler=cosine`
- `--no-amp`
- `inner-loss-weight=0.02`
- `outer-loss-weight=0.2`
- `prior-warmup-epochs=5`

Default SAFE prior layout:
- `experiments/priors/<dataset>/<label-mode>/dog_wslcm_p999_p995/priors/inner_dog_p99_9`
- `experiments/priors/<dataset>/<label-mode>/dog_wslcm_p999_p995/priors/outer_wslcm_p99_5`

Examples:
- `bash scripts/run_one_group.sh irstd1k lightweight_unet full`
- `bash scripts/run_one_group.sh nuaa_sirst lightweight_unet safe_centroid -- --epochs 100 --device cuda:0`
- `bash scripts/run_five_group_matrix.sh sirst3 lightweight_unet -- --epochs 100 --device cuda:0`
