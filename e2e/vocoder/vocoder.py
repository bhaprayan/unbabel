import torch
from vocoder.denoiser import Denoiser
from scipy.io.wavfile import write
from vocoder.tacotron2.utils import load_wav_to_torch

def load_mel(path, hparams, stft):
    audio = load_wav_to_torch(path, hparams.sampling_rate)
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    return melspec

def mel_to_wav(mel, sigma, waveglow, denoiser, hparams, denoise_strength=0.01, save=False, path='audio.wav'):
    with torch.no_grad():
        audio = waveglow.infer(mel.cuda().half(), sigma=sigma)

    audio_denoised = denoiser(audio, strength=denoise_strength)[:, 0].T
    if save:
        write(path, hparams.sampling_rate, audio_denoised.cpu().numpy())

    return audio_denoised
 