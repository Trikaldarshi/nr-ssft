# Cost-effective Self-supervised Fine-tuning for Noise Robust Speech Representations

## Install s3prl toolkit
```
conda create -n s3prl python=3.8 \
conda activate s3prl \
git clone https://github.com/s3prl/s3prl.git \
cd s3prl \
pip install -e ".[all]"
```
```pip install wandb```

## Step 1: Create noise augmented datasets
Download DEMAND and LibriSpeech datasets.
Use ```noise_indoor.py``` for creating train-clean_seen_noisy, dev-clean_seen_noisy, and test-clean_seen_noisy version of the LibriSpeech dataset.
Use ```noise_outdoor.py``` for creating test-clean_unseen_noisy version of LibriSpeech dataset.

## Step 2: Add downstream task to s3prl toolkit

Move ```librispeech_softdtw_noisy``` to ```s3prl/s3prl/downstream/``` to add as a downstream task

In ```librispeech_softdtw_noisy/dataset.py``` set ```base_dir``` which is the path for the DEMAND dataset.

Setup the path ```downstream_expert.datarc.path``` in ```config.yaml```


## Step 3: Modify runner.py
As s3prl does not provide any layerwise control for fine-tuning, we need to modify the ```s3prl/downstream/runner.py``` to freeze the layers that we don't want to train and some other details for configuration used in the downstream task.

Take the code snippet from ```runner_part_freeze_layers.py``` and put it to the ```runner.py``` inside ```_get_upstream_modules()``` function, after the model is loaded, i.e.  

```
model = Upstream(\
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)
> PASTE THE SCRIPT HERE (copied from ```runner_part_freeze.py)
```

## Step 4: NR-SSFt fine-tuning
### For HuBERT

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1.1
ALPHA=0.4
GAMMA=0.1
LOSS_TYPE=softdtw_lav

python3 run_downstream.py -m train -p /path_to_laser_experiment -u hubert_base -d librispeech_softdtw_noisy -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"

```
#### wandb report: https://api.wandb.ai/links/amitmeghu/6ft4g8nn
### For WavLM

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1
ALPHA=0.15
GAMMA=0.1
LOSS_TYPE=softdtw_lav

python3 run_downstream.py -m train -p /path_to_laser_experiment -u wavlm_base -d librispeech_softdtw_noisy -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"

