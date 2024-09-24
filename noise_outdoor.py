import torchaudio
import os
import random
import torch
import torchaudio.functional as F

# set the seed
torch.manual_seed(75)
random.seed(75)


# Load all the audio files in LibriSpeech set defined by the user:
audio_path = input("Please enter the path to the LibriSpeech dataset, example: /abc/LibriSpeech/dev-clean \n")

set_name = audio_path.split('/')[-1]
noise_set_name = set_name + '_unseen_noisy'

audio_files = []
for root, dirs, files in os.walk(audio_path):
    for file in files:
        if file.endswith('.flac'):
            audio_files.append(os.path.join(root, file))


# define snr_dbs [20, 10, 5]
snr_dbs = torch.tensor([20, 10, 5])


# Load the noise files
noise_path = input("Please enter the path to the DEMAND dataset, for example: /abc/DEMAND \n")
noise_files = ['NFIELD_16k/NFIELD/ch01.wav', 'NRIVER_16k/NRIVER/ch01.wav', 'NPARK_16k/NPARK/ch01.wav',
                'STRAFFIC_16k/STRAFFIC/ch01.wav', 'SPSQUARE_16k/SPSQUARE/ch01.wav', 'SCAFE_16k/SCAFE/ch01.wav']
noise_files = [os.path.join(noise_path, f) for f in noise_files]

# Add the noise to the audio files
for audio_file in audio_files:
    # Load the audio file
    waveform, _ = torchaudio.load(audio_file)
    
    # Load the noise file
    noise_file = noise_files[random.randint(0, 5)]

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
