import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchaudio

from datasets import PollyDataset


def generate_sample_sg():
    """
    utility to generate sample sg (for midterm report)
    """
    polly = PollyDataset()
    uk_audio, rate = torchaudio.load("data/uk/treaty_uk.mp3")

    us_audio, rate = torchaudio.load("data/us/treaty_us.mp3")

    us_sg = F.interpolate(polly.sg_transform(us_audio), size=128)
    uk_sg = F.interpolate(polly.sg_transform(uk_audio), size=128)
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(us_sg[0])
    fig.add_subplot(1, 2, 2)
    plt.imshow(uk_sg[0])
    plt.imsave("polly_accent_transfer.png")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
