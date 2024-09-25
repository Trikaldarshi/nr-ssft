import os
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar, _load_waveform
import torchaudio
import random
import pandas as pd
import torch
import torchaudio.functional as F


base_dir = 'your_path to demand dataset'
inside_noise_list = [base_dir + '/DEMAND/DKITCHEN_16k/DKITCHEN/ch01.wav',
                base_dir + '/DEMAND/DWASHING_16k/DWASHING/ch01.wav',
                base_dir + '/DEMAND/DLIVING_16k/DLIVING/ch01.wav',
                base_dir + '/DEMAND/TMETRO_16k/TMETRO/ch01.wav',
                base_dir + '/DEMAND/TBUS_16k/TBUS/ch01.wav',
                base_dir + '/DEMAND/TCAR_16k/TCAR/ch01.wav',
                base_dir + '/DEMAND/OOFFICE_16k/OOFFICE/ch01.wav',
                base_dir + '/DEMAND/OHALLWAY_16k/OHALLWAY/ch01.wav',
                base_dir + '/DEMAND/OMEETING_16k/OMEETING/ch01.wav',
                base_dir + '/DEMAND/PSTATION_16k/PSTATION/ch01.wav',
                base_dir + '/DEMAND/PCAFETER_16k/PCAFETER/ch01.wav',
                base_dir + '/DEMAND/PRESTO_16k/PRESTO/ch01.wav']

snr_dbs = torch.tensor([20, 10, 5])

speed_factors = [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]
speed_perturb = torchaudio.transforms.SpeedPerturbation(16000, speed_factors)


class OOD_Data(Dataset):

    def __init__(self, root,wav_path):
        self.root = root
        # read tab separated file
        self.df = pd.read_csv(self.root)
        self.df['path'] = self.df['file_path'].apply(lambda x: os.path.join(wav_path, x))
        self.length = len(self.df)   

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.
        """
        original_wav, _ = torchaudio.load(self.df['path'][n])
        original_wav = original_wav.view(-1)
        perturbed_wav, _ = speed_perturb(original_wav)
        

        # randomly select a noise
        noise, _ = torchaudio.load(inside_noise_list[random.randint(0, 11)])
        noise = noise.view(-1)
        # randomly select a start point for the noise
        start = random.randint(0, len(noise)-len(perturbed_wav))
        noise = noise[start:start+len(perturbed_wav)]

        # randomly select a snr
        snr = snr_dbs[random.randint(0, 2)]
        perturbed_wav = F.add_noise(perturbed_wav, noise, snr)

        perturbed_wav = perturbed_wav.detach()

        
        # datach if required
        return (perturbed_wav, original_wav)

    
    def collate_fn(self, samples):
        samples_update = []
        for i in range(len(samples)):
            part1 = []
            part2 = []
            part1.append(samples[i][0])
            part2.append(samples[i][1])


            samples_update.append(part1)
            samples_update.append(part2)


        return zip(*samples_update)