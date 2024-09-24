# import packages

import torchaudio
import os
import random
import torch
import torchaudio.functional as F



# Load all the audio files in LibriSpeech set defined by the user:
audio_path = input("Please enter the path to the LibriSpeech dataset, example: /abc/LibriSpeech/dev-clean \n")

set_name = audio_path.split('/')[-1]
noise_set_name = set_name + '_seen_noisy'
# audio files are inside the subdirectories
audio_files = []
for root, dirs, files in os.walk(audio_path):
    for file in files:
        if file.endswith('.flac'):
            audio_files.append(os.path.join(root, file))



# define snr_dbs [20, 10, 5]
snr_dbs = torch.tensor([20, 10, 5])

# set the seed
torch.manual_seed(75)
random.seed(75)

base_dir = input("Please enter the path to the DEMAND dataset, for example: /abc/DEMAND \n")

# Load the noise files
noisy_files =  [base_dir + '/DKITCHEN_16k/DKITCHEN/ch01.wav',
                base_dir + '/DWASHING_16k/DWASHING/ch01.wav',
                base_dir + '/DLIVING_16k/DLIVING/ch01.wav',
                base_dir + '/TMETRO_16k/TMETRO/ch01.wav',
                base_dir + '/TBUS_16k/TBUS/ch01.wav',
                base_dir + '/TCAR_16k/TCAR/ch01.wav',
                base_dir + '/OOFFICE_16k/OOFFICE/ch01.wav',
                base_dir + '/OHALLWAY_16k/OHALLWAY/ch01.wav',
                base_dir + '/OMEETING_16k/OMEETING/ch01.wav',
                base_dir + '/PSTATION_16k/PSTATION/ch01.wav',
                base_dir + '/PCAFETER_16k/PCAFETER/ch01.wav',
                base_dir + '/PRESTO_16k/PRESTO/ch01.wav']


# Add the noise to the audio files
for audio_file in audio_files:
    # Load the audio file
    waveform, _ = torchaudio.load(audio_file)
    
    # Load the noise file
    noise_file = noisy_files[random.randint(0, 11)]

    noise, _ = torchaudio.load(noise_file)
    
    # sample the noise as start and end
    start = random.randint(0, noise.shape[1] - waveform.shape[1])
    noise = noise[:, start:start+waveform.shape[1]]



    # Add the noise to the audio
    snr_db = snr_dbs[random.randint(0, 2)]

    waveform = F.add_noise(waveform, noise, snr_db.unsqueeze(0))

    # Save the noisy audio in test_unseen_noisy
    output_path = audio_file.replace(set_name, noise_set_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torchaudio.save(output_path, waveform, 16000)

    
    


    print(f"Noisy audio saved to {output_path}")
