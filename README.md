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
