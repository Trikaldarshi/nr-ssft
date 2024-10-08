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

## Step 4: 
## NR-SSFt fine-tuning with soft-dtw as loss
### For HuBERT

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1.1
ALPHA=0.4
GAMMA=0.1
LOSS_TYPE=softdtw_lav

python3 run_downstream.py -m train -p /path_to_nr_ssft_experiment -u hubert_base -d librispeech_softdtw_noisy -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"

```

### For WavLM

Set the hyperparameters in config.yaml file or in bash file as follows:
```
SIGMA=0 ## this corresponds to window size = σ = 1
MARGIN=1
ALPHA=0.15
GAMMA=0.1
LOSS_TYPE=softdtw_lav

python3 run_downstream.py -m train -p /path_to_nr_ssft_experiment -u wavlm_base -d librispeech_softdtw_noisy -f -l -1 \
-o "config.downstream_expert.modelrc.sigma=$SIGMA,,config.downstream_expert.modelrc.gamma=$GAMMA,,config.downstream_expert.modelrc.margin=$MARGIN,,config.downstream_expert.modelrc.loss_type=$LOSS_TYPE,,config.downstream_expert.modelrc.alpha=$ALPHA"
```
## NR-SSFt fine-tuning with MSE as loss

Everything remains same, just change the downstream task from librispeech_softdtw_noisy to librispeech_mse_noisy

## Step 4: Evaluate the NR-SSFT finetuned model on ASR and PR for SUPERB benchmark
Download the needed data, set data paths etc for the respective tasks. More info and hyperparameter values are available at [S3PRL/SUPERB](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md)


### For PR
Note: Make sure lr is 5.0e − 4 \
Update the ```libriphone.yaml``` in ```ctc``` downstream task
```
    train: ['train-clean-100_seen_noisy']                # Name of data splits to be used as training set
    dev: ['dev-clean_seen_noisy']                    # Name of data splits to be used as validation set
    test: ['test-clean']
    test_seen_noisy: ['test-clean_seen_noisy']
    test_unseen_noisy: ['test-clean_unseen_noisy']
```

Training:
```
python3 run_downstream.py -p /path_to_pr_experiment -m train -u hubert_base -d ctc -c downstream/ctc/libriphone.yaml \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
OR
```
python3 run_downstream.py -p /path_to_pr_experiment -m train -u wavlm_base -d ctc -c downstream/ctc/libriphone.yaml \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
Testing:

For test
```
python3 run_downstream.py -m evaluate -t "test" -e /path_to_pr_experiment/dev-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
For test_seen_noisy
```
python3 run_downstream.py -m evaluate -t "test_seen_noisy" -e /path_to_pr_experiment/dev-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
For test_unseen_noisy
```
python3 run_downstream.py -m evaluate -t "test_unseen_noisy" -e /path_to_pr_experiment/dev-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```

### For ASR

Update the``` config.yaml``` file from ```asr``` downstream task downstream_expert.datarc as follows:
```downstream_expert:
  datarc:
    train: ['train-clean-100_seen_noisy']
    dev-clean: ['dev-clean_seen_noisy']
    dev-other: ['dev-other']
    test-clean: ['test-clean']
    test-other: ['test-other']
    test-clean_seen_noisy: ['test-clean_seen_noisy']
    test-clean_unseen_noisy: ['test-clean_unseen_noisy']
```

Training:
```
python3 run_downstream.py -p /path_to_asr_experiment -m train -u hubert_base -d asr \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
OR
```
python3 run_downstream.py -p /path_to_asr_experiment -m train -u wavlm_base -d asr \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```
Testing:

For test-clean
```
python3 run_downstream.py -m evaluate -t "test-clean" -e /path_to_asr_experiment/dev-clean-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```

For test-clean_unseen_noisy
```
python3 run_downstream.py -m evaluate -t "test-clean_unseen_noisy" -e /path_to_asr_experiment/dev-clean-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```

For test-clean_seen_noisy
```
python3 run_downstream.py -m evaluate -t "test-clean_seen_noisy" -e /path_to_asr_experiment/dev-clean-best.ckpt \
-o "config.downstream_expert.datarc.test_base_path=path_to_nr_ssft_experiment/states-3600.ckpt,,config.runner.freeze_layers=False,,config.runner.baseline=custom"
```


