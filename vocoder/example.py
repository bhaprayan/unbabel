import torch
from denoiser import Denoiser
from layers import TacotronSTFT, STFT
from hparams import create_hparams
from vocoder import load_mel, mel_to_wav
import pdb


hparams = create_hparams()
# hparams.mel_fmin = 60

stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)

# pdb.set_trace()
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()

denoiser = Denoiser(waveglow)

mel = load_mel('arctic_a0008.wav', hparams, stft)

for sigma in [0.5, 0.7, 0.8, 0.9, 1.0]:
    mel_to_wav(mel, sigma, waveglow, denoiser, hparams, denoise_strength=0.01, save=True, path='output/audio_sigma{}.wav'.format(str(sigma)))